#Functionality: Read seeds, Fetch web pages, Save to data/raw/
import json
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import requests


USER_AGENT = "COMP64702-RAGProject/1.0 (educational; contact: your_email@example.com)"


def project_root() -> Path:
    # src/ingestion/fetch.py -> project root is 3 levels up
    return Path(__file__).resolve().parents[2]


def slugify(text: str, max_len: int = 80) -> str:
    text = text.strip().lower()
    text = re.sub(r"https?://", "", text)
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    if len(text) > max_len:
        text = text[:max_len].rstrip("_")
    return text


def load_seeds(seeds_path: Path) -> Dict[str, List[str]]:
    if not seeds_path.exists():
        raise FileNotFoundError(f"Seed file not found: {seeds_path}")
    with seeds_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Seed JSON must be a dict with keys like 'wikipedia'.")
    return data


@dataclass
class FetchResult:
    source: str
    url: str
    final_url: str
    status_code: int
    fetched_at_utc: str
    html: str


def is_wikipedia_url(url: str) -> bool:
    try:
        host = urlparse(url).netloc.lower()
    except Exception:
        return False
    return host.endswith("wikipedia.org")


def fetch_url(url: str, timeout: int = 30) -> FetchResult:
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/xhtml+xml",
        "Accept-Language": "en",
    }
    r = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)
    r.raise_for_status()

    fetched_at = datetime.now(timezone.utc).isoformat()

    return FetchResult(
        source="wikipedia",
        url=url,
        final_url=r.url,
        status_code=r.status_code,
        fetched_at_utc=fetched_at,
        html=r.text,
    )


def save_raw_json(out_dir: Path, result: FetchResult) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Use the last part of the wiki path if possible
    parsed = urlparse(result.final_url)
    path = parsed.path  # e.g. /wiki/Mediterranean_cuisine
    title = path.split("/")[-1] if "/wiki/" in path else slugify(result.final_url)
    filename = f"wiki__{slugify(title)}.json"

    payload = {
        "source": result.source,
        "url": result.url,
        "final_url": result.final_url,
        "status_code": result.status_code,
        "fetched_at_utc": result.fetched_at_utc,
        "html": result.html,
    }

    out_path = out_dir / filename
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return out_path


def fetch_wikipedia_from_seeds(
    seeds_path: Optional[Path] = None,
    out_dir: Optional[Path] = None,
    sleep_seconds: float = 0.7,
) -> List[Path]:
    """
    Fetch all Wikipedia URLs from the seeds JSON and save each page as a raw JSON file.
    """
    root = project_root()

    if seeds_path is None:
        seeds_path = root / "data" / "samples" / "mediterranean_seeds.json"
    if out_dir is None:
        out_dir = root / "data" / "raw"

    seeds = load_seeds(seeds_path)
    wiki_urls = seeds.get("wikipedia", [])
    if not wiki_urls:
        raise ValueError("No 'wikipedia' URLs found in seeds file.")

    saved_files: List[Path] = []

    for i, url in enumerate(wiki_urls, start=1):
        if not is_wikipedia_url(url):
            print(f"[SKIP] Not a Wikipedia URL: {url}")
            continue

        try:
            print(f"[{i}/{len(wiki_urls)}] Fetching: {url}")
            result = fetch_url(url)
            out_path = save_raw_json(out_dir, result)
            saved_files.append(out_path)
            print(f"      Saved -> {out_path.relative_to(root)} (HTTP {result.status_code})")
        except requests.HTTPError as e:
            # keep going
            status = getattr(e.response, "status_code", None)
            print(f"[ERROR] HTTP error for {url} (status={status}): {e}")
        except requests.RequestException as e:
            print(f"[ERROR] Request failed for {url}: {e}")
        except Exception as e:
            print(f"[ERROR] Unexpected error for {url}: {e}")

        time.sleep(sleep_seconds)

    return saved_files


def is_wikibooks_url(url: str) -> bool:
    try:
        host = urlparse(url).netloc.lower()
    except Exception:
        return False
    return host.endswith("wikibooks.org")


def extract_wikibooks_recipe_links(index_html: str, base_url: str) -> List[str]:
    """
    From a Wikibooks Cookbook index page, extract recipe links.
    We keep links under /wiki/Cookbook:... and filter out special pages.
    """
    soup = BeautifulSoup(index_html, "html.parser")
    content = soup.find("div", id="mw-content-text") or soup

    links = set()
    for a in content.find_all("a", href=True):
        href = a["href"]
        if not href.startswith("/wiki/"):
            continue
        if "Cookbook:" not in href:
            continue
        # Filter out non-content pages
        if any(prefix in href for prefix in ["/wiki/Special:", "/wiki/Help:", "/wiki/File:", "/wiki/Template:"]):
            continue
        full = urljoin(base_url, href)
        links.add(full)

    return sorted(links)


def fetch_wikibooks_from_seeds(
    seeds_path: Optional[Path] = None,
    out_dir: Optional[Path] = None,
    sleep_seconds: float = 0.7,
    max_pages: Optional[int] = 80,
) -> List[Path]:
    """
    Fetch Wikibooks cookbook index pages from seeds, auto-expand recipe links,
    then fetch recipe pages and save as raw JSON.
    """
    root = project_root()

    if seeds_path is None:
        seeds_path = root / "data" / "samples" / "mediterranean_seeds.json"
    if out_dir is None:
        out_dir = root / "data" / "raw"

    seeds = load_seeds(seeds_path)
    wb_urls = seeds.get("wikibooks", [])
    if not wb_urls:
        print("No 'wikibooks' URLs found in seeds file. Skipping.")
        return []

    saved_files: List[Path] = []

    for idx_url in wb_urls:
        if not is_wikibooks_url(idx_url):
            print(f"[SKIP] Not a Wikibooks URL: {idx_url}")
            continue

        print(f"[WB] Fetching index: {idx_url}")
        headers = {
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml",
            "Accept-Language": "en",
        }

        r = requests.get(idx_url, headers=headers, timeout=30, allow_redirects=True)
        r.raise_for_status()

        fetched_at = datetime.now(timezone.utc).isoformat()

        # Save index page too
        index_payload = {
            "source": "wikibooks",
            "url": idx_url,
            "final_url": r.url,
            "status_code": r.status_code,
            "fetched_at_utc": fetched_at,
            "html": r.text,
            "is_index": True,
        }
        index_name = f"wikibooks__index__{slugify(r.url)}.json"
        index_path = out_dir / index_name
        out_dir.mkdir(parents=True, exist_ok=True)
        with index_path.open("w", encoding="utf-8") as f:
            json.dump(index_payload, f, ensure_ascii=False, indent=2)
        saved_files.append(index_path)
        print(f"      Saved index -> {index_path.relative_to(root)}")

        # Expand recipe links
        recipe_links = extract_wikibooks_recipe_links(r.text, base_url=r.url)
        if max_pages is not None:
            recipe_links = recipe_links[:max_pages]

        print(f"[WB] Found {len(recipe_links)} cookbook pages to fetch.")

        for i, url in enumerate(recipe_links, start=1):
            try:
                rr = requests.get(url, headers=headers, timeout=30, allow_redirects=True)
                rr.raise_for_status()

                payload = {
                    "source": "wikibooks",
                    "url": url,
                    "final_url": rr.url,
                    "status_code": rr.status_code,
                    "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
                    "html": rr.text,
                    "is_index": False,
                }

                # filename uses page title part
                parsed = urlparse(rr.url)
                title = parsed.path.split("/")[-1]
                fname = f"wikibooks__{slugify(title)}.json"
                out_path = out_dir / fname
                with out_path.open("w", encoding="utf-8") as f:
                    json.dump(payload, f, ensure_ascii=False, indent=2)

                saved_files.append(out_path)
                if i % 10 == 0 or i == 1:
                    print(f"      [{i}/{len(recipe_links)}] Saved -> {out_path.relative_to(root)}")

            except Exception as e:
                print(f"[WB][ERROR] {url}: {e}")

            time.sleep(sleep_seconds)

    return saved_files


BLOG_SOURCE = "blog80cuisines"

def is_blog80_url(url: str) -> bool:
    try:
        host = urlparse(url).netloc.lower()
    except Exception:
        return False
    return host.endswith("aroundtheworldin80cuisinesblog.wordpress.com")


def extract_blog_post_links(category_html: str, base_url: str) -> List[str]:
    """
    Extract WordPress post permalinks from a category page.
    Heuristic: links containing /YYYY/MM/DD/ are posts.
    """
    soup = BeautifulSoup(category_html, "html.parser")
    links = set()

    for a in soup.find_all("a", href=True):
        href = a["href"]
        full = urljoin(base_url, href)
        if re.search(r"/\d{4}/\d{2}/\d{2}/", full):
            links.add(full)

    return sorted(links)


def fetch_blog80_from_seeds(
    seeds_path: Optional[Path] = None,
    out_dir: Optional[Path] = None,
    sleep_seconds: float = 0.7,
    max_posts_per_category: int = 15,
) -> List[Path]:
    root = project_root()
    if seeds_path is None:
        seeds_path = root / "data" / "samples" / "mediterranean_seeds.json"
    if out_dir is None:
        out_dir = root / "data" / "raw"

    seeds = load_seeds(seeds_path)
    cat_urls = seeds.get("blog80cuisines", [])
    if not cat_urls:
        print("No 'blog80cuisines' URLs found in seeds file. Skipping.")
        return []

    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/xhtml+xml",
        "Accept-Language": "en",
    }

    saved: List[Path] = []
    out_dir.mkdir(parents=True, exist_ok=True)

    for cat in cat_urls:
        if not is_blog80_url(cat):
            print(f"[SKIP] Not blog80 url: {cat}")
            continue

        print(f"[BLOG] Fetch category: {cat}")
        r = requests.get(cat, headers=headers, timeout=30, allow_redirects=True)
        r.raise_for_status()

        # Save category raw (optional but useful for reproducibility)
        cat_payload = {
            "source": BLOG_SOURCE,
            "url": cat,
            "final_url": r.url,
            "status_code": r.status_code,
            "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
            "html": r.text,
            "is_category": True,
        }
        cat_path = out_dir / f"blog80__category__{slugify(r.url)}.json"
        with cat_path.open("w", encoding="utf-8") as f:
            json.dump(cat_payload, f, ensure_ascii=False, indent=2)
        saved.append(cat_path)

        post_links = extract_blog_post_links(r.text, base_url=r.url)
        post_links = post_links[:max_posts_per_category]
        print(f"[BLOG] Found {len(post_links)} post link(s) from this category.")

        for i, url in enumerate(post_links, start=1):
            try:
                rr = requests.get(url, headers=headers, timeout=30, allow_redirects=True)
                rr.raise_for_status()

                payload = {
                    "source": BLOG_SOURCE,
                    "url": url,
                    "final_url": rr.url,
                    "status_code": rr.status_code,
                    "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
                    "html": rr.text,
                    "is_category": False,
                    "category_url": cat,
                }

                fname = f"blog80__{slugify(rr.url)}.json"
                out_path = out_dir / fname
                with out_path.open("w", encoding="utf-8") as f:
                    json.dump(payload, f, ensure_ascii=False, indent=2)

                saved.append(out_path)
                if i == 1 or i % 10 == 0:
                    print(f"      [{i}/{len(post_links)}] Saved -> {out_path.relative_to(root)}")

            except Exception as e:
                print(f"[BLOG][ERROR] {url}: {e}")

            time.sleep(sleep_seconds)

    return saved

if __name__ == "__main__":
    wiki_files = fetch_wikipedia_from_seeds()
    wb_files = fetch_wikibooks_from_seeds()
    blog_files = fetch_blog80_from_seeds()
    print(
        f"\nDone. Saved {len(wiki_files)} wikipedia, {len(wb_files)} wikibooks, {len(blog_files)} blog80 raw file(s) into data/raw/"
    )