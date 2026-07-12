from __future__ import annotations

import html
import math
import re
import unicodedata
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

import requests
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, HttpUrl, ValidationError

router = APIRouter(tags=["image-search"])

OPENVERSE_URL = "https://api.openverse.org/v1/images/"
WIKIMEDIA_URL = "https://commons.wikimedia.org/w/api.php"

HEADERS = {
    "User-Agent": "JarvisImageSearch/1.3 (+https://github.com/)"
}

CONNECT_TIMEOUT_SECONDS = 3.05
READ_TIMEOUT_SECONDS = 7
REQUEST_TIMEOUT = (CONNECT_TIMEOUT_SECONDS, READ_TIMEOUT_SECONDS)
MAX_PROVIDER_RESULTS = 50
MAX_VERIFICATION_WORKERS = 10

TOKEN_PATTERN = re.compile(r"[^\W_]+", re.UNICODE)
HTML_TAG_PATTERN = re.compile(r"<[^>]+>")

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
        "show",
        "the",
        "to",
        "want",
        "would",
        "you",
        "with",
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
    terms = normalized_token_list(query)
    meaningful_terms = [
        term for term in terms if term not in QUERY_FILLER_WORDS
    ]
    selected_terms = meaningful_terms or terms

    unique_terms: list[str] = []
    seen: set[str] = set()
    for term in selected_terms:
        if term not in seen:
            seen.add(term)
            unique_terms.append(term)

    return unique_terms


def meaningful_query_words(query: str) -> set[str]:
    return set(meaningful_query_terms(query))


def metadata_values(value: Any) -> Iterable[str]:
    if value is None:
        return

    if isinstance(value, str):
        cleaned = HTML_TAG_PATTERN.sub(" ", html.unescape(value))
        if cleaned.strip():
            yield cleaned
        return

    if isinstance(value, dict):
        for nested_value in value.values():
            yield from metadata_values(nested_value)
        return

    if isinstance(value, (list, tuple, set)):
        for nested_value in value:
            yield from metadata_values(nested_value)
        return

    if isinstance(value, (int, float, bool)):
        yield str(value)


def build_metadata_text(*values: Any) -> str:
    return " ".join(
        text
        for value in values
        for text in metadata_values(value)
    )


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
    words = normalize_tokens(text)
    return sum(
        1
        for term in query_terms
        if query_term_matches_metadata(term, words)
    )


def relevance_score(candidate: dict[str, Any], query_terms: list[str]) -> int | None:
    if not query_terms:
        return None

    metadata_text = str(candidate.get("metadata_text", ""))
    metadata_matches = matching_term_count(metadata_text, query_terms)
    minimum_matches = max(1, math.ceil(len(query_terms) * 0.6))

    if metadata_matches < minimum_matches:
        return None

    title_text = str(candidate.get("title_text", ""))
    tag_text = str(candidate.get("tag_text", ""))
    description_text = str(candidate.get("description_text", ""))
    creator_text = str(candidate.get("creator_text", ""))

    title_matches = matching_term_count(title_text, query_terms)
    tag_matches = matching_term_count(tag_text, query_terms)
    description_matches = matching_term_count(description_text, query_terms)
    creator_matches = matching_term_count(creator_text, query_terms)

    score = metadata_matches * 3
    score += title_matches * 15
    score += tag_matches * 8
    score += description_matches * 5
    score += creator_matches * 2

    normalized_title = normalized_token_list(title_text)
    query_phrase = " ".join(query_terms)
    title_phrase = " ".join(normalized_title)

    if metadata_matches == len(query_terms):
        score += 15
    if query_terms and all(term in normalized_title for term in query_terms):
        score += 15
    if query_phrase and query_phrase in title_phrase:
        score += 30

    return score


def payload_looks_like_image(payload: bytes) -> bool:
    stripped = payload.lstrip()
    lowered = stripped[:256].lower()

    return (
        payload.startswith(b"\xff\xd8\xff")
        or payload.startswith(b"\x89PNG\r\n\x1a\n")
        or payload.startswith((b"GIF87a", b"GIF89a"))
        or payload.startswith((b"II*\x00", b"MM\x00*"))
        or payload.startswith(b"BM")
        or payload.startswith(b"RIFF") and payload[8:12] == b"WEBP"
        or b"<svg" in lowered
        or b"<svg" in stripped[:2048].lower()
    )


def url_is_working_image(url: str) -> bool:
    if not url.lower().startswith(("http://", "https://")):
        return False

    response: requests.Response | None = None
    try:
        response = requests.get(
            url,
            headers=HEADERS,
            timeout=REQUEST_TIMEOUT,
            stream=True,
            allow_redirects=True,
        )
        if response.status_code != 200:
            return False

        content_type = response.headers.get("content-type", "").lower()
        if content_type.startswith("image/"):
            return True

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
                "page_size": min(max(limit * 5, 20), MAX_PROVIDER_RESULTS),
            },
            headers=HEADERS,
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()
    except (requests.RequestException, ValueError):
        return []

    if not isinstance(data, dict):
        return []

    provider_results = data.get("results", [])
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

        title = item.get("title") or query
        tags = item.get("tags")
        description = item.get("description")
        creator = item.get("creator")
        attribution = item.get("attribution")
        license_value = item.get("license")
        source_value = item.get("source")

        title_text = build_metadata_text(title)
        tag_text = build_metadata_text(tags)
        description_text = build_metadata_text(description, attribution)
        creator_text = build_metadata_text(creator)
        metadata_text = build_metadata_text(
            title_text,
            tag_text,
            description_text,
            creator_text,
        )

        results.append(
            {
                "title": str(title),
                "image_url": image_url,
                "thumbnail_url": thumbnail if isinstance(thumbnail, str) else None,
                "source_page_url": source_page,
                "source_name": str(source_value) if source_value else "Openverse",
                "creator": creator if isinstance(creator, str) else None,
                "license": str(license_value) if license_value is not None else None,
                "metadata_text": metadata_text,
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
    if isinstance(field, dict) and "value" in field:
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
                "gsrlimit": min(max(limit * 5, 20), MAX_PROVIDER_RESULTS),
                "prop": "imageinfo",
                "iiprop": "url|mime|extmetadata",
                "iiurlwidth": 1200,
                "origin": "*",
            },
            headers=HEADERS,
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()
    except (requests.RequestException, ValueError):
        return []

    if not isinstance(data, dict):
        return []

    query_data = data.get("query", {})
    if not isinstance(query_data, dict):
        return []

    pages = query_data.get("pages", [])
    if isinstance(pages, dict):
        page_values = pages.values()
    elif isinstance(pages, list):
        page_values = pages
    else:
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
        if mime_type and not mime_type.startswith("image/"):
            continue

        raw_title = page.get("title") or query
        title = str(raw_title).removeprefix("File:")
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
        metadata_text = build_metadata_text(
            title_text,
            tags,
            description,
            creator,
        )

        license_name = build_metadata_text(
            metadata_field(extmetadata, "LicenseShortName")
        ) or None

        results.append(
            {
                "title": title,
                "image_url": image_url,
                "thumbnail_url": thumbnail_url if isinstance(thumbnail_url, str) else image_url,
                "source_page_url": source_page,
                "source_name": "Wikimedia Commons",
                "creator": creator or None,
                "license": license_name,
                "metadata_text": metadata_text,
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

    with ThreadPoolExecutor(max_workers=len(providers)) as executor:
        futures = [executor.submit(provider, query, limit) for provider in providers]
        results: list[dict[str, Any]] = []
        for future in futures:
            try:
                provider_results = future.result()
            except Exception:
                provider_results = []
            results.extend(provider_results)

    return results


def verify_candidate(candidate: dict[str, Any]) -> ImageResult | None:
    original_url = candidate.get("image_url")
    thumbnail_url = candidate.get("thumbnail_url")
    if not isinstance(original_url, str):
        return None

    preferred_url = thumbnail_url if isinstance(thumbnail_url, str) else original_url
    working_url: str | None = None

    if url_is_working_image(preferred_url):
        working_url = preferred_url
    elif preferred_url != original_url and url_is_working_image(original_url):
        working_url = original_url

    if working_url is None:
        return None

    result_data = {field: candidate.get(field) for field in IMAGE_RESULT_FIELDS}
    result_data["image_url"] = working_url
    result_data["thumbnail_url"] = working_url

    try:
        return ImageResult(**result_data)
    except ValidationError:
        return None


def search_images(query: str, limit: int = 5) -> list[ImageResult]:
    cleaned_query = query.strip()
    if not cleaned_query or limit < 1:
        return []

    safe_limit = min(limit, 10)
    query_terms = meaningful_query_terms(cleaned_query)
    provider_query = " ".join(query_terms) or cleaned_query
    candidates = run_provider_searches(provider_query, safe_limit)

    relevant_candidates: list[dict[str, Any]] = []
    for candidate in candidates:
        score = relevance_score(candidate, query_terms)
        if score is None:
            continue
        candidate["relevance_score"] = score
        relevant_candidates.append(candidate)

    relevant_candidates.sort(
        key=lambda candidate: (
            -candidate["relevance_score"],
            candidate.get("provider_priority", 10),
            candidate.get("provider_rank", 1_000_000),
        )
    )

    unique_candidates: list[dict[str, Any]] = []
    seen_urls: set[str] = set()
    verification_limit = max(safe_limit * 3, 12)

    for candidate in relevant_candidates:
        image_url = candidate.get("image_url")
        thumbnail_url = candidate.get("thumbnail_url")
        identity_url = thumbnail_url if isinstance(thumbnail_url, str) else image_url
        if not isinstance(identity_url, str) or identity_url in seen_urls:
            continue

        seen_urls.add(identity_url)
        unique_candidates.append(candidate)
        if len(unique_candidates) >= verification_limit:
            break

    if not unique_candidates:
        return []

    verified: list[ImageResult] = []
    worker_count = min(MAX_VERIFICATION_WORKERS, len(unique_candidates))
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        for result in executor.map(verify_candidate, unique_candidates):
            if result is None:
                continue
            verified.append(result)
            if len(verified) >= safe_limit:
                break

    return verified


@router.get(
    "/image-search",
    response_model=ImageSearchResponse,
)
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

    return ImageSearchResponse(
        query=query,
        results=search_images(query, limit),
    )
