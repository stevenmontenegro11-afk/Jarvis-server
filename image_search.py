from __future__ import annotations

import html
import re
import unicodedata
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable

import requests
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, HttpUrl, ValidationError

router = APIRouter(tags=["image-search"])

OPENVERSE_URL = "https://api.openverse.org/v1/images/"
WIKIMEDIA_URL = "https://commons.wikimedia.org/w/api.php"
HEADERS = {
    "User-Agent": "JarvisImageSearch/1.6 (FastAPI image search service)",
    "Accept": "application/json, image/avif, image/webp, image/apng, image/svg+xml, image/*, */*;q=0.8",
}
PROVIDER_TIMEOUT = (3.05, 6.0)
VERIFICATION_TIMEOUT = (2.5, 4.0)
MAX_PROVIDER_RESULTS = 40
MAX_VERIFICATION_WORKERS = 8
MAX_VERIFICATION_CANDIDATES = 36
VERIFICATION_BATCH_SIZE = 12

TOKEN_PATTERN = re.compile(r"[^\W_]+", re.UNICODE)
HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
REQUEST_PHRASE_PATTERN = re.compile(
    r"\b(?:show|find|give|get|send)\s+me\b"
    r"|\b(?:show|find|give|get|send|display|search)\b"
    r"|\b(?:picture|photo|photograph|image)s?\s+of\b",
    re.IGNORECASE,
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
IMAGE_RESULT_FIELDS = (
    "title",
    "image_url",
    "thumbnail_url",
    "source_page_url",
    "source_name",
    "creator",
    "license",
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


def normalize_tokens(value: str) -> set[str]:
    return set(normalized_token_list(value))


def meaningful_query_terms(query: str) -> list[str]:
    phrase_cleaned_query = REQUEST_PHRASE_PATTERN.sub(" ", query)
    terms = normalized_token_list(phrase_cleaned_query)
    meaningful_terms = [term for term in terms if term not in QUERY_FILLER_WORDS]

    if not meaningful_terms:
        original_terms = normalized_token_list(query)
        meaningful_terms = [
            term for term in original_terms if term not in QUERY_FILLER_WORDS
        ]

    unique_terms: list[str] = []
    seen: set[str] = set()
    for term in meaningful_terms:
        if term in seen:
            continue
        seen.add(term)
        unique_terms.append(term)
    return unique_terms


def meaningful_query_words(query: str) -> set[str]:
    return set(meaningful_query_terms(query))


def normalized_meaningful_query(query: str) -> str:
    return " ".join(meaningful_query_terms(query))


def metadata_values(value: Any) -> Iterable[str]:
    if value is None:
        return
    if isinstance(value, str):
        cleaned = HTML_TAG_PATTERN.sub(" ", html.unescape(value)).strip()
        if cleaned:
            yield cleaned
        return
    if isinstance(value, dict):
        for nested in value.values():
            yield from metadata_values(nested)
        return
    if isinstance(value, (list, tuple, set)):
        for nested in value:
            yield from metadata_values(nested)
        return
    if isinstance(value, (int, float, bool)):
        yield str(value)


def build_metadata_text(*values: Any) -> str:
    return " ".join(text for value in values for text in metadata_values(value))


def token_variants(token: str) -> set[str]:
    variants = {token}
    if len(token) > 4 and token.endswith("ies"):
        variants.add(f"{token[:-3]}y")
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        variants.add(token[:-1])
    return variants


def query_term_matches_metadata(term: str, metadata_words: set[str]) -> bool:
    term_variants = token_variants(term)
    return any(
        term_variants.intersection(token_variants(metadata_word))
        for metadata_word in metadata_words
    )


def matching_term_count(text: str, query_terms: list[str]) -> int:
    metadata_words = normalize_tokens(text)
    return sum(
        query_term_matches_metadata(term, metadata_words) for term in query_terms
    )


def phrase_occurs(text: str, query_terms: list[str]) -> bool:
    if not text or not query_terms:
        return False
    normalized_text = " ".join(normalized_token_list(text))
    return " ".join(query_terms) in normalized_text


def relevance_score(candidate: dict[str, Any], query_terms: list[str]) -> int:
    if not query_terms:
        return 0

    metadata_text = str(candidate.get("metadata_text", ""))
    title_text = str(candidate.get("title_text", ""))
    tag_text = str(candidate.get("tag_text", ""))
    description_text = str(candidate.get("description_text", ""))
    creator_text = str(candidate.get("creator_text", ""))

    metadata_matches = matching_term_count(metadata_text, query_terms)
    title_matches = matching_term_count(title_text, query_terms)
    tag_matches = matching_term_count(tag_text, query_terms)
    description_matches = matching_term_count(description_text, query_terms)
    creator_matches = matching_term_count(creator_text, query_terms)

    score = (
        metadata_matches * 4
        + title_matches * 18
        + tag_matches * 10
        + description_matches * 6
        + creator_matches
    )
    score += round(20 * metadata_matches / len(query_terms))

    if metadata_matches == len(query_terms):
        score += 15
    if phrase_occurs(metadata_text, query_terms):
        score += 15
    if phrase_occurs(title_text, query_terms):
        score += 35
    elif phrase_occurs(tag_text, query_terms):
        score += 20
    elif phrase_occurs(description_text, query_terms):
        score += 10

    return score


def payload_looks_like_image(payload: bytes) -> bool:
    stripped = payload.lstrip()
    lowered = stripped[:2048].lower()
    return (
        payload.startswith(b"\xff\xd8\xff")
        or payload.startswith(b"\x89PNG\r\n\x1a\n")
        or payload.startswith((b"GIF87a", b"GIF89a"))
        or payload.startswith(b"RIFF") and payload[8:12] == b"WEBP"
        or b"<svg" in lowered
    )


def url_is_working_image(url: str) -> bool:
    if not url.lower().startswith(("http://", "https://")):
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
        if response.status_code not in (200, 206):
            return False

        content_type = (
            response.headers.get("content-type", "")
            .split(";", 1)[0]
            .lower()
            .strip()
        )
        if content_type in DISPLAYABLE_IMAGE_TYPES:
            return True
        if content_type.startswith("image/"):
            return False

        first_chunk = next(response.iter_content(chunk_size=2048), b"")
        return payload_looks_like_image(first_chunk)
    except (requests.RequestException, OSError):
        return False
    finally:
        if response is not None:
            response.close()


def search_openverse(query: str, limit: int) -> list[dict[str, Any]]:
    try:
        response = requests.get(
            OPENVERSE_URL,
            params={
                "q": query,
                "page_size": min(max(limit * 4, 20), MAX_PROVIDER_RESULTS),
            },
            headers=HEADERS,
            timeout=PROVIDER_TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()
    except (requests.RequestException, ValueError):
        return []

    provider_results = data.get("results", []) if isinstance(data, dict) else []
    if not isinstance(provider_results, list):
        return []

    results: list[dict[str, Any]] = []
    for provider_rank, item in enumerate(provider_results):
        if not isinstance(item, dict):
            continue

        image_url = item.get("url")
        source_page = item.get("foreign_landing_url")
        thumbnail = item.get("thumbnail")
        if not isinstance(image_url, str) or not isinstance(source_page, str):
            continue

        title = str(item.get("title") or query)
        creator = item.get("creator")
        title_text = build_metadata_text(title)
        tag_text = build_metadata_text(item.get("tags"))
        description_text = build_metadata_text(
            item.get("description"), item.get("attribution")
        )
        creator_text = build_metadata_text(creator)
        results.append(
            {
                "title": title,
                "image_url": image_url,
                "thumbnail_url": thumbnail if isinstance(thumbnail, str) else None,
                "source_page_url": source_page,
                "source_name": str(item.get("source") or "Openverse"),
                "creator": creator if isinstance(creator, str) else None,
                "license": str(item["license"])
                if item.get("license") is not None
                else None,
                "metadata_text": build_metadata_text(
                    title_text, tag_text, description_text, creator_text
                ),
                "title_text": title_text,
                "tag_text": tag_text,
                "description_text": description_text,
                "creator_text": creator_text,
                "provider_rank": provider_rank,
                "provider_priority": 1,
            }
        )
    return results


def metadata_field(extmetadata: Any, field_name: str) -> Any:
    if not isinstance(extmetadata, dict):
        return None
    field = extmetadata.get(field_name)
    if isinstance(field, dict):
        return field.get("value")
    return field


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
                "gsrlimit": min(max(limit * 4, 20), MAX_PROVIDER_RESULTS),
                "prop": "imageinfo",
                "iiprop": "url|mime|extmetadata",
                "iiurlwidth": 1200,
                "origin": "*",
            },
            headers=HEADERS,
            timeout=PROVIDER_TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()
    except (requests.RequestException, ValueError):
        return []

    query_data = data.get("query", {}) if isinstance(data, dict) else {}
    pages = query_data.get("pages", []) if isinstance(query_data, dict) else []
    page_values = pages.values() if isinstance(pages, dict) else pages
    if not isinstance(page_values, Iterable):
        return []

    ordered_pages = sorted(
        (page for page in page_values if isinstance(page, dict)),
        key=lambda page: page.get("index", 1_000_000),
    )
    results: list[dict[str, Any]] = []
    for fallback_rank, page in enumerate(ordered_pages):
        image_info = page.get("imageinfo", [])
        if not isinstance(image_info, list) or not image_info:
            continue
        info = image_info[0]
        if not isinstance(info, dict):
            continue

        image_url = info.get("url")
        thumbnail_url = info.get("thumburl") or image_url
        source_page = info.get("descriptionurl")
        mime_type = str(info.get("mime", "")).lower()
        if not isinstance(image_url, str) or not isinstance(source_page, str):
            continue
        if mime_type and mime_type not in DISPLAYABLE_IMAGE_TYPES:
            continue

        title = str(page.get("title") or query).removeprefix("File:")
        extmetadata = info.get("extmetadata")
        description = build_metadata_text(
            metadata_field(extmetadata, "ImageDescription"),
            metadata_field(extmetadata, "ObjectName"),
            metadata_field(extmetadata, "ShortDescription"),
        )
        tags = build_metadata_text(
            metadata_field(extmetadata, "Categories"),
            metadata_field(extmetadata, "DepictedPeople"),
            metadata_field(extmetadata, "Location"),
        )
        creator = build_metadata_text(
            metadata_field(extmetadata, "Artist"),
            metadata_field(extmetadata, "Credit"),
        )
        title_text = build_metadata_text(title)
        results.append(
            {
                "title": title,
                "image_url": image_url,
                "thumbnail_url": thumbnail_url
                if isinstance(thumbnail_url, str)
                else image_url,
                "source_page_url": source_page,
                "source_name": "Wikimedia Commons",
                "creator": creator or None,
                "license": build_metadata_text(
                    metadata_field(extmetadata, "LicenseShortName")
                )
                or None,
                "metadata_text": build_metadata_text(
                    title_text, tags, description, creator
                ),
                "title_text": title_text,
                "tag_text": tags,
                "description_text": description,
                "creator_text": creator,
                "provider_rank": page.get("index", fallback_rank),
                "provider_priority": 0,
            }
        )
    return results


def run_provider_searches(query: str, limit: int) -> list[dict[str, Any]]:
    providers: tuple[Callable[[str, int], list[dict[str, Any]]], ...] = (
        search_openverse,
        search_wikimedia,
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


def verify_candidate(candidate: dict[str, Any]) -> ImageResult | None:
    original_url = candidate.get("image_url")
    thumbnail_url = candidate.get("thumbnail_url")
    if not isinstance(original_url, str):
        return None

    urls: list[str] = []
    if isinstance(thumbnail_url, str):
        urls.append(thumbnail_url)
    if original_url not in urls:
        urls.append(original_url)

    working_url = next((url for url in urls if url_is_working_image(url)), None)
    if working_url is None:
        return None

    result_data = {field: candidate.get(field) for field in IMAGE_RESULT_FIELDS}
    result_data["image_url"] = working_url
    result_data["thumbnail_url"] = working_url
    try:
        return ImageResult(**result_data)
    except ValidationError:
        return None


def verify_candidate_batch(
    candidates: list[dict[str, Any]],
) -> list[tuple[int, ImageResult]]:
    if not candidates:
        return []

    verified: list[tuple[int, ImageResult]] = []
    worker_count = min(MAX_VERIFICATION_WORKERS, len(candidates))
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        future_candidates = {
            executor.submit(verify_candidate, candidate): candidate
            for candidate in candidates
        }
        for future in as_completed(future_candidates):
            candidate = future_candidates[future]
            try:
                result = future.result()
            except Exception:
                result = None
            if result is not None:
                verified.append((candidate["verification_rank"], result))
    return verified


def search_images(query: str, limit: int = 5) -> list[ImageResult]:
    cleaned_query = query.strip()
    if not cleaned_query or limit < 1:
        return []

    safe_limit = min(limit, 10)
    query_terms = meaningful_query_terms(cleaned_query)
    if not query_terms:
        return []

    provider_query = " ".join(query_terms)
    candidates = run_provider_searches(provider_query, safe_limit)

    ranked: list[dict[str, Any]] = []
    for candidate in candidates:
        candidate["relevance_score"] = relevance_score(candidate, query_terms)
        ranked.append(candidate)

    ranked.sort(
        key=lambda candidate: (
            -candidate["relevance_score"],
            candidate.get("provider_rank", 1_000_000),
            candidate.get("provider_priority", 10),
        )
    )

    unique: list[dict[str, Any]] = []
    seen_urls: set[str] = set()
    for candidate in ranked:
        identity = candidate.get("thumbnail_url") or candidate.get("image_url")
        if not isinstance(identity, str) or identity in seen_urls:
            continue
        seen_urls.add(identity)
        candidate["verification_rank"] = len(unique)
        unique.append(candidate)
        if len(unique) >= MAX_VERIFICATION_CANDIDATES:
            break

    if not unique:
        return []

    verified: list[tuple[int, ImageResult]] = []
    for batch_start in range(0, len(unique), VERIFICATION_BATCH_SIZE):
        batch = unique[batch_start : batch_start + VERIFICATION_BATCH_SIZE]
        verified.extend(verify_candidate_batch(batch))
        if len(verified) >= safe_limit:
            break

    verified.sort(key=lambda item: item[0])
    return [result for _, result in verified[:safe_limit]]


@router.get("/image-search", response_model=ImageSearchResponse)
def image_search(
    q: str = Query(min_length=2, max_length=200),
    limit: int = Query(default=5, ge=1, le=10),
):
    query = q.strip()
    if not query:
        raise HTTPException(
            status_code=400,
            detail="Image search query cannot be empty.",
        )

    meaningful_query = normalized_meaningful_query(query)
    if not meaningful_query:
        raise HTTPException(
            status_code=400,
            detail="Image search query must include a meaningful subject.",
        )

    return ImageSearchResponse(
        query=meaningful_query,
        results=search_images(meaningful_query, limit),
    )
