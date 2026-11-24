#!/usr/bin/env python3
"""Two-stage, resumable scraper for thefilmnepal.com.

Stage 1: paginate GET /api/v1/movies?page=N&per_page=24 into
         data2/raw/thefilmnepal_catalog.json (page checkpoints keep it resumable).
Stage 2: for each movie, fetch the detail page (LD+JSON Movie + SSR synopsis fallback),
         optionally the /reviews page (only when total_ratings > 0), and the
         /editor-review page for the 19 slugs listed in the sitemap. Records are
         appended immediately to data2/raw/thefilmnepal_full.jsonl; completed
         slugs are skipped on restart.
Stage 3: print counts (catalog, synopses, reviews, genres).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlparse

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BASE = "https://www.thefilmnepal.com"
API_URL = f"{BASE}/api/v1/movies"
CATALOG_FIELDS = ("description", "genre", "alternateName", "datePublished", "duration", "director", "actor")

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data2" / "raw"
CATALOG_PATH = RAW / "thefilmnepal_catalog.json"
CATALOG_CHECKPOINT = RAW / "thefilmnepal_catalog.pages.jsonl"
OUTPUT_PATH = RAW / "thefilmnepal_full.jsonl"
URLS_PATH = RAW / "movie_urls.jsonl"
SUMMARY_PATH = RAW / "summary.json"

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/157.0.0.0 Safari/537.36"
)

_thread_local = threading.local()
_append_lock = threading.Lock()


# --------------------------------------------------------------------------- #
# HTTP helpers
# --------------------------------------------------------------------------- #
def build_session(retries: int) -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=retries,
        connect=retries,
        read=retries,
        status=retries,
        backoff_factor=0.6,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        respect_retry_after_header=True,
        raise_on_status=False,
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.headers.update(
        {
            "User-Agent": USER_AGENT,
            "Accept": "application/json, text/html;q=0.9, */*;q=0.8",
            "Accept-Language": "en-GB,en-US;q=0.9,en;q=0.8",
            "Referer": f"{BASE}/browse-nepali-movies",
        }
    )
    return session


def worker_session(retries: int) -> requests.Session:
    session = getattr(_thread_local, "session", None)
    if session is None:
        session = build_session(retries)
        _thread_local.session = session
    return session


def get(session: requests.Session, url: str, params: dict[str, Any] | None = None) -> requests.Response:
    response = session.get(url, params=params, timeout=(10, 30))
    response.raise_for_status()
    return response


def pause(low: float, high: float) -> None:
    time.sleep(random.uniform(low, high))


# --------------------------------------------------------------------------- #
# Stage 1 — catalog
# --------------------------------------------------------------------------- #
def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def load_page_checkpoints(path: Path) -> dict[int, dict[str, Any]]:
    pages: dict[int, dict[str, Any]] = {}
    if not path.exists():
        return pages
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                pages[int(record["page"])] = record
            except (json.JSONDecodeError, KeyError, ValueError):
                continue
    return pages


def stage1_catalog(session: requests.Session, args: argparse.Namespace) -> list[dict[str, Any]]:
    pages = load_page_checkpoints(args.catalog_checkpoint)
    last_page = max((int(p["meta"]["last_page"]) for p in pages.values()), default=1)
    print(f"Stage 1: {len(pages)} pages checkpointed, last_page={last_page}")

    def fetch_page(page: int) -> None:
        nonlocal last_page
        response = get(session, API_URL, params={"page": page, "per_page": 24})
        payload = response.json()
        meta = payload["meta"]
        record = {"page": page, "meta": meta, "data": payload["data"]}
        with args.catalog_checkpoint.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        pages[page] = record
        last_page = max(last_page, int(meta["last_page"]))
        if page % 10 == 0 or page == last_page:
            print(
                f"  catalog page {page}/{last_page} "
                f"({sum(len(p['data']) for p in pages.values())} movies)"
            )
        if page < last_page:
            pause(0.5, 1.0)

    # Page 1 reveals meta.last_page; the remaining range is built afterwards.
    if 1 not in pages:
        fetch_page(1)
    for page in range(2, last_page + 1):
        if page not in pages:
            fetch_page(page)

    catalog: list[dict[str, Any]] = []
    for page in sorted(pages):
        catalog.extend(pages[page]["data"])
    # Keep the raw catalog intact (the API repeats a few rows across page boundaries).
    # De-duplication happens only when Stage 2 builds its work list.
    write_json(args.catalog_path, catalog)
    unique_slugs = {movie.get("slug") for movie in catalog}
    print(
        f"Stage 1 done: {len(catalog)} raw rows / {len(unique_slugs)} unique slugs "
        f"-> {args.catalog_path}"
    )
    return catalog


def load_catalog(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list in {path}")
    return data


# --------------------------------------------------------------------------- #
# Stage 2 — detail / reviews / editor reviews
# --------------------------------------------------------------------------- #
def load_completed(path: Path) -> set[str]:
    done: set[str] = set()
    if not path.exists():
        return done
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("status") == "ok" and record.get("slug"):
                done.add(record["slug"])
    return done


def load_editor_slugs(path: Path) -> set[str]:
    slugs: set[str] = set()
    if not path.exists():
        return slugs
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                url = json.loads(line)["url"]
            except (json.JSONDecodeError, KeyError):
                continue
            parts = [p for p in urlparse(url).path.split("/") if p]
            if len(parts) >= 3 and parts[0] == "movie" and parts[2] == "editor-review":
                slugs.add(parts[1])
    return slugs


def parse_movie_html(html: str) -> dict[str, Any]:
    soup = BeautifulSoup(html, "html.parser")
    movie_node: dict[str, Any] | None = None
    for tag in soup.find_all("script", type="application/ld+json"):
        try:
            payload = json.loads(tag.string or tag.get_text() or "")
        except json.JSONDecodeError:
            continue
        nodes = payload.get("@graph", [payload]) if isinstance(payload, dict) else payload
        for node in nodes if isinstance(nodes, list) else [nodes]:
            if isinstance(node, dict) and node.get("@type") == "Movie":
                movie_node = node
                break
        if movie_node:
            break

    result: dict[str, Any] = {}
    if movie_node:
        result["ld_name"] = movie_node.get("name")
        for field in CATALOG_FIELDS:
            result[field] = movie_node.get(field)

    if not result.get("description"):
        fallback = soup.select_one("section.summary .movie-show-synopsis")
        if fallback:
            result["description"] = fallback.get_text(" ", strip=True)
            result["synopsis_source"] = "html-fallback"
        else:
            result["synopsis_source"] = "none"
    else:
        result["synopsis_source"] = "ld+json"
    return result


def parse_reviews(html: str) -> list[dict[str, Any]]:
    soup = BeautifulSoup(html, "html.parser")
    reviews = []
    for card in soup.select(".audience-rating__card"):
        def text(selector: str) -> str | None:
            node = card.select_one(selector)
            return node.get_text(" ", strip=True) if node else None

        reviews.append(
            {
                "author": text(".audience-rating__name"),
                "date": text(".audience-rating__date"),
                "score": text(".audience-rating__user-score"),
                "text": text(".audience-rating__text"),
            }
        )
    return reviews


def parse_editor_review(html: str) -> dict[str, Any]:
    soup = BeautifulSoup(html, "html.parser")
    title = soup.select_one(".movie-editor-review__article-title")
    body = soup.select_one(".movie-editor-review__body")
    return {
        "title": title.get_text(" ", strip=True) if title else None,
        "text": body.get_text(" ", strip=True) if body else None,
    }


def process_movie(
    movie: dict[str, Any], args: argparse.Namespace, editor_slugs: set[str]
) -> tuple[str, str]:
    slug = str(movie["slug"])
    record: dict[str, Any] = {
        "slug": slug,
        "title": movie.get("title"),
        "id": movie.get("id"),
        "catalog_url": movie.get("url"),
        "catalog_year": movie.get("year"),
        "catalog_genres": movie.get("genres"),
        "catalog_total_ratings": movie.get("total_ratings"),
        "status": "error",
    }
    session = worker_session(args.retries)
    base_movie_url = f"{BASE}/movie/{quote(slug, safe='')}"
    try:
        pause(0.8, 1.5)
        detail = get(session, base_movie_url)
        record.update(parse_movie_html(detail.text))

        # Optional sections failing must not discard the detail data.
        reviews: list[dict[str, Any]] = []
        if int(movie.get("total_ratings") or 0) > 0:
            try:
                pause(0.8, 1.5)
                reviews_page = get(session, base_movie_url + "/reviews")
                reviews = parse_reviews(reviews_page.text)
            except Exception as exc:
                record["reviews_error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
        record["reviews"] = reviews
        record["reviews_count"] = len(reviews)

        if slug in editor_slugs:
            try:
                pause(0.8, 1.5)
                editor_page = get(session, base_movie_url + "/editor-review")
                record["editor_review"] = parse_editor_review(editor_page.text)
            except Exception as exc:
                record["editor_review_error"] = f"{type(exc).__name__}: {str(exc)[:200]}"

        record["status"] = "ok"
    except Exception as exc:  # record the failure so it can be retried on resume
        record["error"] = f"{type(exc).__name__}: {str(exc)[:250]}"

    with _append_lock:
        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        with args.output_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
    return slug, record["status"]


def stage2_details(catalog: list[dict[str, Any]], args: argparse.Namespace) -> None:
    editor_slugs = load_editor_slugs(args.urls_path)
    done = load_completed(args.output_path)
    seen_slugs: set[str] = set()
    targets = []
    for movie in catalog:
        slug = str(movie.get("slug"))
        if slug in seen_slugs or slug in done:
            continue
        seen_slugs.add(slug)
        targets.append(movie)
    if args.max_movies is not None:
        targets = targets[: args.max_movies]
    print(
        f"Stage 2: {len(targets)} movies to process "
        f"({len(done)} already complete, {len(editor_slugs)} editor-review slugs)"
    )

    completed = 0
    failures = 0
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(process_movie, movie, args, editor_slugs) for movie in targets]
        for future in as_completed(futures):
            _, status = future.result()
            completed += 1
            if status != "ok":
                failures += 1
            if completed % 25 == 0 or completed == len(targets):
                print(f"  details progress: {completed}/{len(targets)} (failures={failures})")
    print(f"Stage 2 done: {completed} processed, {failures} failures -> {args.output_path}")


# --------------------------------------------------------------------------- #
# Stage 3 — summary
# --------------------------------------------------------------------------- #
def stage3_summary(catalog: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    if args.output_path.exists():
        with args.output_path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

    ok = [r for r in records if r.get("status") == "ok"]
    synopses = sum(1 for r in ok if (r.get("description") or "").strip())
    review_movies = sum(1 for r in ok if int(r.get("reviews_count") or 0) > 0)
    total_reviews = sum(int(r.get("reviews_count") or 0) for r in ok)

    genre_counts: Counter[str] = Counter()
    for movie in catalog:
        names = []
        for genre in movie.get("genres") or []:
            if isinstance(genre, dict):
                names.append(genre.get("title") or genre.get("name"))
            elif isinstance(genre, str):
                names.append(genre)
        for name in set(filter(None, names)):
            genre_counts[str(name)] += 1

    summary = {
        "catalog_movies": len(catalog),
        "detail_records_ok": len(ok),
        "detail_records_error": sum(1 for r in records if r.get("status") == "error"),
        "movies_with_synopsis": synopses,
        "movies_with_reviews": review_movies,
        "total_audience_reviews": total_reviews,
        "genre_distribution": dict(genre_counts.most_common()),
    }
    write_json(args.summary_path, summary)

    print("\n=== Summary ===")
    print(f"Total catalog movies fetched: {summary['catalog_movies']}")
    print(f"Movies with non-empty synopsis: {synopses}")
    print(f"Audience reviews: {total_reviews} across {review_movies} movies")
    print(f"Genre distribution ({len(genre_counts)} genres):")
    for name, count in genre_counts.most_common():
        print(f"  {name}: {count}")
    print(f"Summary saved to {args.summary_path}")
    return summary


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--catalog-path", type=Path, default=CATALOG_PATH)
    parser.add_argument("--catalog-checkpoint", type=Path, default=CATALOG_CHECKPOINT)
    parser.add_argument("--output-path", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--urls-path", type=Path, default=URLS_PATH)
    parser.add_argument("--summary-path", type=Path, default=SUMMARY_PATH)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--max-movies", type=int, default=None, help="limit stage 2 (smoke test)")
    parser.add_argument("--skip-stage1", action="store_true", help="reuse existing catalog file")
    parser.add_argument("--stage1-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    session = build_session(args.retries)

    if args.skip_stage1:
        catalog = load_catalog(args.catalog_path)
        print(f"Loaded catalog: {len(catalog)} movies from {args.catalog_path}")
    else:
        catalog = stage1_catalog(session, args)

    if not args.stage1_only:
        stage2_details(catalog, args)

    stage3_summary(catalog, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
