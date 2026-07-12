from __future__ import annotations

from typing import Any

import requests
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, HttpUrl


router = APIRouter(tags=["image-search"])

OPENVERSE_URL = "https://api.openverse.org/v1/images/"
WIKIMEDIA_URL = "https://commons.wikimedia.org/w/api.php"

HEADERS = {
    "User-Agent": "JarvisImageSearch/1.0"
}

TIMEOUT_SECONDS = 12


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


def url_is_working_image(url: str) -> bool:
    try:
        response = requests.get(
            url,
            headers=HEADERS,
            timeout=TIMEOUT_SECONDS,
            stream=True,
            allow_redirects=True,
        )

        content_type = response.headers.get("content-type", "").lower()

        return (
            response.status_code == 200
            and content_type.startswith("image/")
        )

    except requests.RequestException:
        return False


def search_openverse(query: str, limit: int) -> list[dict[str, Any]]:
    try:
        response = requests.get(
            OPENVERSE_URL,
            params={
                "q": query,
                "page_size": min(limit * 3, 20),
            },
            headers=HEADERS,
            timeout=TIMEOUT_SECONDS,
        )

        response.raise_for_status()
        data = response.json()

    except (requests.RequestException, ValueError):
        return []

    results: list[dict[str, Any]] = []

    for item in data.get("results", []):
        image_url = item.get("url")
        source_page = item.get("foreign_landing_url")
        thumbnail = item.get("thumbnail")

        if not image_url or not source_page:
            continue

        results.append(
            {
                "title": item.get("title") or query,
                "image_url": image_url,
                "thumbnail_url": thumbnail,
                "source_page_url": source_page,
                "source_name": item.get("source") or "Openverse",
                "creator": item.get("creator"),
                "license": item.get("license"),
            }
        )

    return results


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
                "gsrlimit": min(limit * 3, 20),
                "prop": "imageinfo",
                "iiprop": "url|mime",
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

    pages = data.get("query", {}).get("pages", {})
    results: list[dict[str, Any]] = []

    for page in pages.values():
        image_info = page.get("imageinfo", [])

        if not image_info:
            continue

        info = image_info[0]
        image_url = info.get("url")
        thumbnail_url = info.get("thumburl") or image_url
        source_page = info.get("descriptionurl")

        if not image_url or not source_page:
            continue

        title = page.get("title", query).replace("File:", "", 1)

        results.append(
            {
                "title": title,
                "image_url": image_url,
                "thumbnail_url": thumbnail_url,
                "source_page_url": source_page,
                "source_name": "Wikimedia Commons",
                "creator": None,
                "license": None,
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

    candidates = [
        *search_openverse(query, limit),
        *search_wikimedia(query, limit),
    ]

    verified: list[ImageResult] = []
    seen_urls: set[str] = set()

    for candidate in candidates:
        image_url = candidate["image_url"]

        if image_url in seen_urls:
            continue

        seen_urls.add(image_url)

        preview_url = (
            candidate.get("thumbnail_url")
            or image_url
        )

        if not url_is_working_image(preview_url):
            continue

        verified.append(ImageResult(**candidate))

        if len(verified) >= limit:
            break

    return ImageSearchResponse(
        query=query,
        results=verified,
    )
