import json
import re
from pathlib import Path
from typing import List, Tuple, Optional
from collections import Counter
from statistics import mean, median
import html
from bs4 import BeautifulSoup, Tag

print("USING UPDATED CLEAN.PY")

STOP_SECTIONS = {
    "references",
    "external links",
    "further reading",
    "see also",
    "notes",
    "sources",
    "bibliography",
}


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def clean_text(text: str) -> str:
    text = html.unescape(text)
    # Remove citation brackets like [1], [ 12 ], [ 3 ]
    text = re.sub(r"\[\s*\d+\s*\]", "", text)

    nav_noise = ["jump to navigation", "jump to search", "edit section", "retrieved from"]
    for noise in nav_noise:
        text = re.sub(f"(?i){noise}", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    # Fix spaces before punctuation: "word ," -> "word,"
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)

    # Fix spaces around apostrophes: "Egypt 's" -> "Egypt's"
    text = re.sub(r"\s+'\s*", "'", text)

    # Fix spaces around parentheses: "( text )" -> "(text)"
    text = re.sub(r"\(\s+", "(", text)
    text = re.sub(r"\s+\)", ")", text)

    return text

BOILERPLATE_PATTERNS = [
    r"^jump to navigation$",
    r"^jump to search$",
    r"^printable version$",
    r"^from wikibooks, open books for an open world$",
    r"^please help improve this.*$",
    r"^this article is a stub.*$",
    r"^advertisements?$",
    r"^share this:?$",
    r"^like this:?$",
    r"^related$",
    r"^cookbook\s*\|.*$",
    r"^recipes\s*\|.*$",
    r"^ingredients\s*\|.*$",
    r"^equipment\s*\|.*$",
    r"^techniques\s*\|.*$",
    r"^please read .* before attempting this recipe[.\"]?$",
r"^please read .* before attempting this .*[.\"]?$",
]

SHORT_TEXT_EXCEPTIONS = {
    "salt", "water", "pepper", "oil", "sugar", "flour", "yeast", "egg", "eggs"
}


def normalize_for_dedup(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text


def looks_like_boilerplate(text: str) -> bool:
    t = text.strip().lower()
    for pat in BOILERPLATE_PATTERNS:
        if re.match(pat, t):
            return True
    return False


def looks_like_noise_block(text: str) -> bool:
    t = text.strip()

    if not t:
        return True

    # only punctuation / symbols
    if re.fullmatch(r"[\W_]+", t):
        return True

    low = t.lower()

    # very short navigation-ish text
    if low in {"home", "next", "previous", "print", "edit", "navigation"}:
        return True

    # repeated title-ish token
    words = low.split()
    if len(words) >= 3 and len(set(words)) == 1:
        return True

    # obvious boilerplate phrases
    if looks_like_boilerplate(t):
        return True

    # low-value recipe template prompts
    if "please read" in low and "before attempting this recipe" in low:
        return True

    # breadcrumb / index-like text with many pipes
    pipe_parts = [p.strip() for p in t.split("|")]
    if len(pipe_parts) >= 4:
        short_parts = sum(1 for p in pipe_parts if 0 < len(p) <= 30)
        if short_parts >= len(pipe_parts) - 1:
            return True

    return False


def in_generic_bad_container(tag: Tag) -> bool:
    bad_parent_names = {"table", "style", "script", "noscript", "aside", "footer", "nav", "form"}
    bad_classes = {
        "infobox", "navbox", "sidebar", "toc", "metadata", "mw-references-wrap",
        "reflist", "thumb", "mw-editsection", "reference", "references",
        "sharedaddy", "jp-relatedposts", "widget", "footer", "post-navigation",
        "sidebar-widget", "comments", "comment", "advertisement"
    }

    for parent in tag.find_parents(True):
        if parent.name in bad_parent_names:
            return True
        cls = parent.get("class") or []
        if any(c in bad_classes for c in cls):
            return True

    return False


def finalize_block(text: str, section_path: List[str]) -> Optional[dict]:
    text = clean_text(text)
    if not text:
        return None
    if looks_like_noise_block(text):
        return None
    return {
        "section_path": [s for s in section_path if s],
        "text": text
    }


def is_stop_section(title: str) -> bool:
    t = title.strip().lower()
    return t in STOP_SECTIONS


def get_heading_text(tag: Tag) -> str:
    # Wikipedia headings often wrap text in <span class="mw-headline">
    headline = tag.find("span", class_="mw-headline")
    if headline:
        return headline.get_text(" ", strip=True)
    return tag.get_text(" ", strip=True)

def extract_wikipedia_blocks(html: str) -> Tuple[str, List[dict]]:
    """
    Return page_title and list of blocks:
      block = {"section_path": [...], "text": paragraph}
    """
    soup = BeautifulSoup(html, "html.parser")

    # Page title
    title_tag = soup.find("h1", id="firstHeading")
    page_title = title_tag.get_text(" ", strip=True) if title_tag else "Unknown"

    content = soup.find("div", id="mw-content-text")
    if content is None:
        return page_title, []

    body = content.find("div", class_="mw-parser-output") or content

    # Track current headings
    section_h2: Optional[str] = None
    section_h3: Optional[str] = None
    section_h4: Optional[str] = None

    blocks: List[dict] = []

    def in_bad_container(tag: Tag) -> bool:
        # Skip paragraphs/headings inside common non-article containers
        bad_parents = tag.find_parents(
            ["table", "style", "script"]
        )
        if bad_parents:
            return True

        # Skip typical Wikipedia boxes and non-content regions
        bad_classes = {
            "infobox",
            "navbox",
            "sidebar",
            "toc",
            "metadata",
            "mw-references-wrap",
            "reflist",
            "thumb",
            "mw-editsection",
        }
        for parent in tag.find_parents(True):
            cls = parent.get("class") or []
            if any(c in bad_classes for c in cls):
                return True
        return False

    # Walk tags in document order (recursive)
    for el in body.find_all(["h2", "h3", "h4", "p"], recursive=True):
        if in_bad_container(el):
            continue

        if el.name == "h2":
            h2 = get_heading_text(el)
            if is_stop_section(h2):
                break
            section_h2, section_h3, section_h4 = h2, None, None
            continue

        if el.name == "h3":
            h3 = get_heading_text(el)
            section_h3, section_h4 = h3, None
            continue

        if el.name == "h4":
            h4 = get_heading_text(el)
            section_h4 = h4
            continue

        if el.name == "p":
            text = clean_text(el.get_text(" ", strip=True))
            if len(text) < 60:
                continue

            block = finalize_block(text, [section_h2, section_h3, section_h4])
            if block:
                blocks.append(block)

    return page_title, blocks

def extract_wikibooks_blocks(html: str) -> Tuple[str, List[dict]]:
    soup = BeautifulSoup(html, "html.parser")

    title_tag = soup.find("h1", id="firstHeading")
    page_title = title_tag.get_text(" ", strip=True) if title_tag else "Unknown"

    content = soup.find("div", id="mw-content-text")
    if content is None:
        return page_title, []

    body = content.find("div", class_="mw-parser-output") or content

    blocks: List[dict] = []
    section_h2: Optional[str] = None
    section_h3: Optional[str] = None
    current_list_items: List[str] = []

    def flush_list_items():
        nonlocal current_list_items
        if not current_list_items:
            return
        combined = "\n".join(f"- {x}" for x in current_list_items if x.strip())
        block = finalize_block(combined, [section_h2, section_h3])
        if block:
            blocks.append(block)
        current_list_items = []

    for el in body.find_all(["h2", "h3", "p", "ul", "ol"], recursive=True):
        if in_generic_bad_container(el):
            continue

        if el.name == "h2":
            flush_list_items()
            heading = get_heading_text(el)
            if is_stop_section(heading):
                break
            section_h2, section_h3 = heading, None
            continue

        if el.name == "h3":
            flush_list_items()
            section_h3 = get_heading_text(el)
            continue

        if el.name == "p":
            flush_list_items()
            text = clean_text(el.get_text(" ", strip=True))
            if len(text) < 40:
                continue
            block = finalize_block(text, [section_h2, section_h3])
            if block:
                blocks.append(block)
            continue

        if el.name in {"ul", "ol"}:
            flush_list_items()
            list_items = []
            for li in el.find_all("li", recursive=False):
                text = clean_text(li.get_text(" ", strip=True))
                if not text:
                    continue
                if looks_like_noise_block(text):
                    continue
                list_items.append(text)

            if list_items:
                current_list_items.extend(list_items)
                flush_list_items()

    flush_list_items()
    return page_title, blocks

def extract_blog80_blocks(html: str) -> Tuple[str, List[dict]]:
    soup = BeautifulSoup(html, "html.parser")

    title_tag = soup.find(["h1", "h2"], class_=re.compile(r"entry-title"))
    if not title_tag:
        title_tag = soup.find("h1")
    page_title = title_tag.get_text(" ", strip=True) if title_tag else "Unknown"

    content = soup.find("div", class_=re.compile(r"entry-content"))
    if content is None:
        return page_title, []

    blocks: List[dict] = []
    current_h2: Optional[str] = None
    current_h3: Optional[str] = None
    current_list_items: List[str] = []

    def flush_list_items():
        nonlocal current_list_items
        if not current_list_items:
            return
        combined = "\n".join(f"- {x}" for x in current_list_items if x.strip())
        block = finalize_block(combined, [current_h2, current_h3])
        if block:
            blocks.append(block)
        current_list_items = []

    for el in content.find_all(["h2", "h3", "p", "ul", "ol"], recursive=True):
        if in_generic_bad_container(el):
            continue

        if el.name == "h2":
            flush_list_items()
            current_h2, current_h3 = clean_text(el.get_text(" ", strip=True)), None
            continue

        if el.name == "h3":
            flush_list_items()
            current_h3 = clean_text(el.get_text(" ", strip=True))
            continue

        if el.name == "p":
            flush_list_items()
            text = clean_text(el.get_text(" ", strip=True))
            if not text:
                continue
            if text.lower().startswith("image:"):
                continue
            if len(text) < 50:
                continue
            block = finalize_block(text, [current_h2, current_h3])
            if block:
                blocks.append(block)
            continue

        if el.name in {"ul", "ol"}:
            flush_list_items()
            list_items = []
            for li in el.find_all("li", recursive=False):
                text = clean_text(li.get_text(" ", strip=True))
                if not text:
                    continue
                if text.lower().startswith("image:"):
                    continue
                if looks_like_noise_block(text):
                    continue
                list_items.append(text)

            if list_items:
                current_list_items.extend(list_items)
                flush_list_items()

    flush_list_items()
    return page_title, blocks




def build_corpus() -> None:
    root = project_root()
    raw_dir = root / "data" / "raw"
    processed_dir = root / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    output_path = processed_dir / "corpus.jsonl"
    stats_path = processed_dir / "corpus_stats.json"

    written = 0
    files = sorted(
        list(raw_dir.glob("wiki__*.json"))
        + list(raw_dir.glob("wikibooks__*.json"))
        + list(raw_dir.glob("blog80__*.json"))
    )

    source_counter = Counter()
    block_lengths = []

    all_records = []

    for file in files:
        with file.open("r", encoding="utf-8") as f:
            data = json.load(f)

        source = data.get("source")
        if data.get("is_index") is True:
            continue
        if data.get("is_category") is True:
            continue

        if source == "wikipedia":
            page_title, blocks = extract_wikipedia_blocks(data["html"])
        elif source == "wikibooks":
            page_title, blocks = extract_wikibooks_blocks(data["html"])
        elif source == "blog80cuisines":
            page_title, blocks = extract_blog80_blocks(data["html"])
        else:
            continue

        # skip Wikibooks directory / index-like pages that slipped through
        if source == "wikibooks":
            normalized_title = page_title.strip().lower().replace(" ", "")
            if normalized_title in {"cookbook:cuisines", "cookbook:cuisine"}:
                continue

        if not blocks:
            continue

        for i, b in enumerate(blocks):
            if source == "wikipedia":
                page_type = "article"
            elif source == "wikibooks":
                page_type = "recipe"
            elif source == "blog80cuisines":
                page_type = "blog_post"
            else:
                page_type = "unknown"

            text = b["text"]

            # simple block type detection
            if "\n-" in text or text.strip().startswith("-"):
                block_type = "list"
            elif len(text) < 80:
                block_type = "short"
            else:
                block_type = "paragraph"

            record = {
                "doc_id": f"{file.stem}__p{i}",
                "source": source,
                "url": data.get("final_url") or data.get("url"),
                "title": page_title,
                "section_path": b["section_path"],
                "text": b["text"],
                "page_type": page_type,
                "char_len": len(b["text"]),
                "block_type": block_type,
            }
            all_records.append(record)

    all_records, duplicates_removed = deduplicate_records(all_records)

    source_counter = Counter()
    block_lengths = []

    with output_path.open("w", encoding="utf-8") as out_f:
        for r in all_records:
            out_f.write(json.dumps(r, ensure_ascii=False) + "\n")
            source_counter[r["source"]] += 1
            block_lengths.append(len(r["text"]))
            written += 1

    stats = {
        "processed_raw_files": len(files),
        "total_blocks_written": written,
        "blocks_per_source": dict(source_counter),
        "exact_duplicates_removed": duplicates_removed,
        "average_block_length_chars": round(mean(block_lengths), 2) if block_lengths else 0,
        "median_block_length_chars": round(median(block_lengths), 2) if block_lengths else 0,
        "max_block_length_chars": max(block_lengths) if block_lengths else 0,
        "blocks_over_1200_chars": sum(1 for x in block_lengths if x > 1200),
    }

    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"Corpus built → {output_path}")
    print(f"Processed {len(files)} raw file(s)")
    print(f"Total blocks written: {written}")
    print(f"Blocks per source: {dict(source_counter)}")
    print(f"Exact duplicates removed: {duplicates_removed}")

    if block_lengths:
        print(f"Average block length: {mean(block_lengths):.2f} chars")
        print(f"Median block length: {median(block_lengths):.2f} chars")
        print(f"Max block length: {max(block_lengths)} chars")
        print(f"Blocks > 1200 chars: {sum(1 for x in block_lengths if x > 1200)}")

def deduplicate_records(records: List[dict]) -> Tuple[List[dict], int]:
    seen = set()
    deduped = []
    removed = 0

    for r in records:
        key = (
            r["source"],
            r["title"].strip().lower(),
            tuple(r["section_path"]),
            r["text"].strip().lower(),
        )
        if key in seen:
            removed += 1
            continue
        seen.add(key)
        deduped.append(r)

    return deduped, removed

if __name__ == "__main__":
    build_corpus()