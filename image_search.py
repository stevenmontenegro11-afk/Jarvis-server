from __future__ import annotations

import html
import re
import unicodedata
from collections.abc import Iterable
from typing import Any

import requests
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, HttpUrl, ValidationError


router = APIRouter(tags=["image-search"])

OPENVERSE_URL = "https://api.openverse.org/v1/images/"
WIKIMEDIA_URL = "https://commons.wikimedia.org/w/api.php"

HEADERS = {
    "User-Agent": "JarvisImageSearch/1.1"
}

TIMEOUT_SECONDS = 12

TOKEN_PATTERN = re.compile(r"[^\W_]+", re.UNICODE)
HTML_TAG_PATTERN = re.compile(r"<[^>]+>")

QUERY_FILLER_WORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "at",
        "by",
        "for",
        "find",
        "from",
        "give",
        "image",
        "images",
        "in",
        "me",
        "of",
        "on",
        "or",
        "photo",
        "photos",
        "picture",
        "pictures",
        "please",
        "search",
        "show",
        "the",
        "to",
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

    for metadata_word in metadata_words:
        if term_variants.intersection(token_variants(metadata_word)):
            return True

    return False


def metadata_matches_query(metadata_text: str, query_words: set[str]) -> bool:
    if not query_words:
        return False

    metadata_words = normalize_tokens(metadata_text)
    if not metadata_words:
        return False

    return all(
        query_term_matches_metadata(term, metadata_words)
        for term in query_words
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

    metadata_text = candidate.get("metadata_text", "")
    if not metadata_matches_query(metadata_text, set(query_terms)):
        return None

    title_text = candidate.get("title_text", "")
    tag_text = candidate.get("tag_text", "")
    description_text = candidate.get("description_text", "")
    creator_text = candidate.get("creator_text", "")

    score = len(query_terms)
    score += matching_term_count(title_text, query_terms) * 12
    score += matching_term_count(tag_text, query_terms) * 7
    score += matching_term_count(description_text, query_terms) * 4
    score += matching_term_count(creator_text, query_terms) * 2

    normalized_title = normalized_token_list(title_text)
    if query_terms and all(term in normalized_title for term in query_terms):
        score += 10

    query_phrase = " ".join(query_terms)
    title_phrase = " ".join(normalized_title)
    if query_phrase and query_phrase in title_phrase:
        score += 25

    return score


def url_is_working_image(url: str) -> bool:
    response: requests.Response | None = None

    try:
        response = requests.get(
            url,
            headers=HEADERS,
            timeout=TIMEOUT_SECONDS,
            stream=True,
            allow_redirects=True,
        )
        content_type = response.headers.get("content-type", "").lower()
        return response.status_code == 200 and content_type.startswith("image/")
    except requests.RequestException:
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
                "page_size": min(max(limit * 5, 20), 50),
            },
            headers=HEADERS,
            timeout=TIMEOUT_SECONDS,
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
                "source_name": item.get("source") or "Openverse",
                "creator": creator if isinstance(creator, str) else None,
                "license": item.get("license"),
                "metadata_text": metadata_text,
                "title_text": title_text,
                "tag_text": tag_text,
                "description_text": description_text,
                "creator_text": creator_text,
                "provider_rank": provider_rank,
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
                "generator": "search",
                "gsrsearch": query,
                "gsrnamespace": 6,
                "gsrlimit": min(max(limit * 5, 20), 50),
                "prop": "imageinfo",
                "iiprop": "url|mime|extmetadata",
                "iiurlwidth": 1200,
                "origin": "*",
            },
            headers=HEADERS,
            timeout=TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        data = response.json()
    except (requests.RequestException, ValueError):
        return []

    if not isinstance(data, dict):
        return []

    pages = data.get("query", {}).get("pages", {})
    if not isinstance(pages, dict):
        return []

    ordered_pages = sorted(
        (page for page in pages.values() if isinstance(page, dict)),
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

        if not isinstance(image_url, str) or not isinstance(source_page, str):
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
        creator_name = creator or None

        results.append(
            {
                "title": title,
                "image_url": image_url,
                "thumbnail_url": (
                    thumbnail_url if isinstance(thumbnail_url, str) else image_url
                ),
                "source_page_url": source_page,
                "source_name": "Wikimedia Commons",
                "creator": creator_name,
                "license": license_name,
                "metadata_text": metadata_text,
                "title_text": title_text,
                "tag_text": tags,
                "description_text": description,
                "creator_text": creator,
                "provider_rank": page.get("index", fallback_rank),
            }
        )

    return results


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

    query_terms = meaningful_query_terms(query)
    provider_query = " ".join(query_terms) or query
    candidates = [
        *search_openverse(provider_query, limit),
        *search_wikimedia(provider_query, limit),
    ]

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
            candidate.get("provider_rank", 1_000_000),
        )
    )

    verified: list[ImageResult] = []
    seen_urls: set[str] = set()

    for candidate in relevant_candidates:
        image_url = candidate.get("image_url")
        if not isinstance(image_url, str) or image_url in seen_urls:
            continue

        seen_urls.add(image_url)
        preview_url = candidate.get("thumbnail_url") or image_url

        if not isinstance(preview_url, str) or not url_is_working_image(preview_url):
            continue

        result_data = {
            field: candidate.get(field)
            for field in IMAGE_RESULT_FIELDS
        }

        try:
            verified.append(ImageResult(**result_data))
        except ValidationError:
            continue

        if len(verified) >= limit:
            break

    return ImageSearchResponse(
        query=query,
        results=verified,
    )
