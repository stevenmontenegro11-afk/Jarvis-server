from __future__ import annotations

import html
import re
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable

import requests
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, HttpUrl, ValidationError

router = APIRouter(tags=["image-search"])

OPENVERSE_URL = "https://api.openverse.org/v1/images/"
WIKIMEDIA_URL = "https://commons.wikimedia.org/w/api.php"
HEADERS = {
    "User-Agent": "JarvisImageSearch/2.0 (FastAPI image search service)",
    "Accept": "application/json, image/avif, image/webp, image/apng, image/svg+xml, image/*, */*;q=0.8",
}
PROVIDER_TIMEOUT = (3.05, 10.0)
VERIFICATION_TIMEOUT = (3.05, 8.0)
MAX_PROVIDER_RESULTS = 40
MAX_VERIFICATION_CANDIDATES = 40
MAX_VERIFICATION_WORKERS = 8
VERIFICATION_BATCH_SIZE = 10

TOKEN_PATTERN = re.compile(r"[^\W_]+", re.UNICODE)
HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
REQUEST_PHRASES = (
    re.compile(r"\b(?:can|could|would|will)\s+you\b", re.IGNORECASE),
    re.compile(r"\b(?:show|find|give|get|send)\s+me\b", re.IGNORECASE),
    re.compile(
        r"\b(?:a\s+|an\s+|the\s+)?(?:picture|photo|photograph|image)s?\s+of\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:show|find|give|get|send|display|search|see|look)\b",
        re.IGNORECASE,
    ),
)
QUERY_FILLER_WORDS = frozenset(
    {
        "a",
        "about",
        "an",
        "and",
        "at",
        "by",
        "can",
        "could",
        "display",
        "for",
        "find",
        "from",
        "get",
        "give",
        "i",
        "image",
        "images",
        "in",
        "look",
        "looking",
        "me",
        "of",
        "on",
        "or",
        "photo",
        "photograph",
        "photographs",
        "photos",
        "picture",
        "pictures",
        "please",
        "search",
        "see",
        "send",
        "show",
        "showing",
        "the",
        "to",
        "want",
        "will",
        "would",
        "you",
        "with",
    }
)
DISPLAYABLE_IMAGE_TYPES = frozenset(
    {
        "image/avif",
        "image/gif",
        "image/jpeg",
        "image/jpg",
        "image/png",
        "image/svg+xml",
        "image/webp",
    }
)


class ImageResult(BaseModel):
    title: str
    image_url: HttpUrl
    thumbnail_url: HttpUrl | None = None
    source_page_url: HttpUrl
    source_name: str
    creator: str | None = None
    license: str | None = None


class ImageSearchResponse(BaseModel):
    query: str
    results: list[ImageResult]


def normalized_token_list(value: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return TOKEN_PATTERN.findall(normalized)


def meaningful_query_terms(query: str) -> list[str]:
    cleaned = query
    for pattern in REQUEST_PHRASES:
        cleaned = pattern.sub(" ", cleaned)

    terms = [
        term
        for term in normalized_token_list(cleaned)
        if term not in QUERY_FILLER_WORDS
    ]

    if not terms:
        terms = [
            term
            for term in normalized_token_list(query)
            if term not in QUERY_FILLER_WORDS
        ]

    unique_terms: list[str] = []
    seen: set[str] = set()
    for term in terms:
        if term in seen:
            continue
        seen.add(term)
        unique_terms.append(term)
    return unique_terms


def meaningful_query_words(query: str) -> set[str]:
    return set(meaningful_query_terms(query))


def normalized_meaningful_query(query: str) -> str:
    return " ".join(meaningful_query_terms(query))


def _metadata_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return HTML_TAG_PATTERN.sub(" ", html.unescape(value)).strip()
    if isinstance(value, dict):
        return " ".join(_metadata_text(item) for item in value.values()).strip()
    if isinstance(value, (list, tuple, set)):
        return " ".join(_metadata_text(item) for item in value).strip()
    return str(value)


def _token_variants(token: str) -> set[str]:
    variants = {token}
    if len(token) > 4 and token.endswith("ies"):
        variants.add(f"{token[:-3]}y")
    if len(token) > 3 and token.endswith("es"):
        variants.add(token[:-2])
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        variants.add(token[:-1])
    return variants


def _term_matches(term: str, metadata_tokens: set[str]) -> bool:
    term_variants = _token_variants(term)
    return any(
        term_variants.intersection(_token_variants(metadata_token))
        for metadata_token in metadata_tokens
    )


def _relevance_score(candidate: dict[str, Any], query_terms: list[str]) -> int:
    title = str(candidate.get("title", ""))
    metadata = str(candidate.get("metadata", ""))
    title_tokens = set(normalized_token_list(title))
    metadata_tokens = set(normalized_token_list(metadata))

    title_matches = sum(_term_matches(term, title_tokens) for term in query_terms)
    metadata_matches = sum(
        _term_matches(term, metadata_tokens) for term in query_terms
    )
    normalized_title = " ".join(normalized_token_list(title))
    normalized_phrase = " ".join(query_terms)

    score = title_matches * 30 + metadata_matches * 10
    if metadata_matches == len(query_terms):
        score += 35
    if normalized_phrase and normalized_phrase in normalized_title:
        score += 60
    score -= min(int(candidate.get("provider_rank", 0)), 40)
    return score


def _metadata_field(extmetadata: Any, name: str) -> Any:
    if not isinstance(extmetadata, dict):
        return None
    value = extmetadata.get(name)
    if isinstance(value, dict):
        return value.get("value")
    return value


def search_wikimedia(query: str, limit: int) -> list[dict[str, Any]]:
    try:
        response = requests.get(
            WIKIMEDIA_URL,
            params={
                "action": "query",
                "format": "json",
                "formatversion": 2,
                "generator": "search",
                "gsrsearch": query,
                "gsrnamespace": 6,
                "gsrlimit": min(max(limit * 6, 24), MAX_PROVIDER_RESULTS),
                "prop": "imageinfo",
                "iiprop": "url|mime|extmetadata",
                "iiurlwidth": 1200,
                "origin": "*",
            },
            headers=HEADERS,
            timeout=PROVIDER_TIMEOUT,
        )
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError):
        return []

    query_payload = payload.get("query", {}) if isinstance(payload, dict) else {}
    pages = query_payload.get("pages", []) if isinstance(query_payload, dict) else []
    if isinstance(pages, dict):
        pages = list(pages.values())
    if not isinstance(pages, list):
        return []

    pages = sorted(
        (page for page in pages if isinstance(page, dict)),
        key=lambda page: page.get("index", 1_000_000),
    )
    results: list[dict[str, Any]] = []

    for fallback_rank, page in enumerate(pages):
        image_info = page.get("imageinfo")
        if not isinstance(image_info, list) or not image_info:
            continue
        info = image_info[0]
        if not isinstance(info, dict):
            continue

        image_url = info.get("url")
        thumbnail_url = info.get("thumburl") or image_url
        source_page_url = info.get("descriptionurl")
        mime_type = str(info.get("mime", "")).lower()
        if not isinstance(image_url, str) or not isinstance(source_page_url, str):
            continue
        if mime_type and mime_type not in DISPLAYABLE_IMAGE_TYPES:
            continue

        title = str(page.get("title") or query).removeprefix("File:")
        extmetadata = info.get("extmetadata")
        creator = _metadata_text(
            [
                _metadata_field(extmetadata, "Artist"),
                _metadata_field(extmetadata, "Credit"),
            ]
        )
        metadata = _metadata_text(
            [
                title,
                _metadata_field(extmetadata, "ImageDescription"),
                _metadata_field(extmetadata, "ObjectName"),
                _metadata_field(extmetadata, "ShortDescription"),
                _metadata_field(extmetadata, "Categories"),
                _metadata_field(extmetadata, "Location"),
                creator,
            ]
        )
        results.append(
            {
                "title": title,
                "image_url": image_url,
                "thumbnail_url": thumbnail_url,
                "source_page_url": source_page_url,
                "source_name": "Wikimedia Commons",
                "creator": creator or None,
                "license": _metadata_text(
                    _metadata_field(extmetadata, "LicenseShortName")
                )
                or None,
                "metadata": metadata,
                "provider_rank": page.get("index", fallback_rank),
                "provider_priority": 0,
            }
        )
    return results


def search_openverse(query: str, limit: int) -> list[dict[str, Any]]:
    try:
        response = requests.get(
            OPENVERSE_URL,
            params={
                "q": query,
                "page_size": min(max(limit * 6, 24), MAX_PROVIDER_RESULTS),
            },
            headers=HEADERS,
            timeout=PROVIDER_TIMEOUT,
        )
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError):
        return []

    provider_results = payload.get("results", []) if isinstance(payload, dict) else []
    if not isinstance(provider_results, list):
        return []

    results: list[dict[str, Any]] = []
    for rank, item in enumerate(provider_results):
        if not isinstance(item, dict):
            continue
        image_url = item.get("url")
        source_page_url = item.get("foreign_landing_url")
        thumbnail_url = item.get("thumbnail")
        if not isinstance(image_url, str) or not isinstance(source_page_url, str):
            continue

        title = str(item.get("title") or query)
        creator = item.get("creator")
        results.append(
            {
                "title": title,
                "image_url": image_url,
                "thumbnail_url": thumbnail_url if isinstance(thumbnail_url, str) else None,
                "source_page_url": source_page_url,
                "source_name": str(item.get("source") or "Openverse"),
                "creator": creator if isinstance(creator, str) else None,
                "license": str(item.get("license")) if item.get("license") else None,
                "metadata": _metadata_text(
                    [
                        title,
                        item.get("tags"),
                        item.get("description"),
                        item.get("attribution"),
                        creator,
                    ]
                ),
                "provider_rank": rank,
                "provider_priority": 1,
            }
        )
    return results


def _run_provider_searches(query: str, limit: int) -> list[dict[str, Any]]:
    providers: tuple[Callable[[str, int], list[dict[str, Any]]], ...] = (
        search_wikimedia,
        search_openverse,
    )
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=len(providers)) as executor:
        futures = [executor.submit(provider, query, limit) for provider in providers]
        for future in as_completed(futures):
            try:
                results.extend(future.result())
            except Exception:
                continue
    return results


def _rank_candidates(
    candidates: list[dict[str, Any]],
    query_terms: list[str],
) -> list[dict[str, Any]]:
    for candidate in candidates:
        candidate["relevance_score"] = _relevance_score(candidate, query_terms)

    candidates.sort(
        key=lambda candidate: (
            -int(candidate.get("relevance_score", 0)),
            int(candidate.get("provider_rank", 1_000_000)),
            int(candidate.get("provider_priority", 10)),
        )
    )

    ranked: list[dict[str, Any]] = []
    seen: set[str] = set()
    for candidate in candidates:
        identity = candidate.get("image_url") or candidate.get("thumbnail_url")
        if not isinstance(identity, str) or identity in seen:
            continue
        seen.add(identity)
        candidate["verification_rank"] = len(ranked)
        ranked.append(candidate)
        if len(ranked) >= MAX_VERIFICATION_CANDIDATES:
            break
    return ranked


def _payload_looks_like_image(payload: bytes) -> bool:
    lowered = payload.lstrip()[:2048].lower()
    return (
        payload.startswith(b"\xff\xd8\xff")
        or payload.startswith(b"\x89PNG\r\n\x1a\n")
        or payload.startswith((b"GIF87a", b"GIF89a"))
        or (payload.startswith(b"RIFF") and payload[8:12] == b"WEBP")
        or b"<svg" in lowered
    )


def _url_is_working_image(url: str) -> bool:
    if not url.lower().startswith(("https://", "http://")):
        return False

    response: requests.Response | None = None
    try:
        response = requests.get(
            url,
            headers={**HEADERS, "Range": "bytes=0-2047"},
            timeout=VERIFICATION_TIMEOUT,
            stream=True,
            allow_redirects=True,
        )
        if response.status_code not in {200, 206}:
            return False

        content_type = response.headers.get("content-type", "").split(";", 1)[0]
        content_type = content_type.lower().strip()
        if content_type in DISPLAYABLE_IMAGE_TYPES:
            return True
        if content_type.startswith("image/"):
            return False

        first_chunk = next(response.iter_content(chunk_size=2048), b"")
        return _payload_looks_like_image(first_chunk)
    except (requests.RequestException, OSError):
        return False
    finally:
        if response is not None:
            response.close()


def _verify_candidate(candidate: dict[str, Any]) -> ImageResult | None:
    image_url = candidate.get("image_url")
    thumbnail_url = candidate.get("thumbnail_url")
    if not isinstance(image_url, str):
        return None

    candidate_urls: list[str] = []
    if isinstance(thumbnail_url, str):
        candidate_urls.append(thumbnail_url)
    if image_url not in candidate_urls:
        candidate_urls.append(image_url)

    working_url = next(
        (url for url in candidate_urls if _url_is_working_image(url)),
        None,
    )
    if working_url is None:
        return None

    try:
        return ImageResult(
            title=str(candidate.get("title") or "Image result"),
            image_url=working_url,
            thumbnail_url=working_url,
            source_page_url=candidate["source_page_url"],
            source_name=str(candidate.get("source_name") or "Image source"),
            creator=candidate.get("creator"),
            license=candidate.get("license"),
        )
    except (KeyError, ValidationError):
        return None


def _verify_candidates(
    candidates: list[dict[str, Any]],
    limit: int,
) -> list[ImageResult]:
    verified: list[tuple[int, ImageResult]] = []

    for batch_start in range(0, len(candidates), VERIFICATION_BATCH_SIZE):
        batch = candidates[batch_start : batch_start + VERIFICATION_BATCH_SIZE]
        workers = min(MAX_VERIFICATION_WORKERS, len(batch))
        if workers < 1:
            break

        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_candidates = {
                executor.submit(_verify_candidate, candidate): candidate
                for candidate in batch
            }
            for future in as_completed(future_candidates):
                candidate = future_candidates[future]
                try:
                    result = future.result()
                except Exception:
                    result = None
                if result is not None:
                    verified.append(
                        (int(candidate["verification_rank"]), result)
                    )

        if len(verified) >= limit:
            break

    verified.sort(key=lambda item: item[0])
    return [result for _, result in verified[:limit]]


def search_images(query: str, limit: int = 5) -> list[ImageResult]:
    query_terms = meaningful_query_terms(query.strip())
    if not query_terms or limit < 1:
        return []

    safe_limit = min(limit, 10)
    provider_query = " ".join(query_terms)
    candidates = _run_provider_searches(provider_query, safe_limit)
    ranked = _rank_candidates(candidates, query_terms)
    results = _verify_candidates(ranked, safe_limit)
    if results or len(query_terms) == 1:
        return results

    fallback_candidates: list[dict[str, Any]] = []
    for term in query_terms:
        fallback_candidates.extend(_run_provider_searches(term, safe_limit))
    fallback_ranked = _rank_candidates(fallback_candidates, query_terms)
    return _verify_candidates(fallback_ranked, safe_limit)


@router.get("/image-search", response_model=ImageSearchResponse)
def image_search(
    q: str = Query(min_length=2, max_length=200),
    limit: int = Query(default=5, ge=1, le=10),
):
    query = normalized_meaningful_query(q.strip())
    if not query:
        raise HTTPException(
            status_code=400,
            detail="Image search query must include a meaningful subject.",
        )

    return ImageSearchResponse(
        query=query,
        results=search_images(query, limit),
    )
