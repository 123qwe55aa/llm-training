#!/usr/bin/env python3
"""Crawl public MIT 6.036 courseware into local Markdown and a course manifest."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import html
import json
import re
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

BASE = "https://openlearninglibrary.mit.edu"
COURSE = "course-v1:MITx+6.036+1T2019"
OUTLINE_URL = f"{BASE}/courses/{COURSE}/course/"


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-") or "item"


def get(session: requests.Session, url: str) -> requests.Response:
    response = session.get(url, timeout=120)
    response.raise_for_status()
    return response


def parse_outline(text: str) -> list[dict]:
    soup = BeautifulSoup(text, "lxml")
    chapters = []
    for chapter_node in soup.select("li.outline-item.section"):
        title_node = chapter_node.select_one("h3.section-title")
        chapter_button = chapter_node.select_one("button.section-name")
        if not title_node or not chapter_button:
            continue
        chapter_id = chapter_button.get("id", "").split("@")[-1]
        sections = []
        for anchor in chapter_node.select("a.subsection-text"):
            section_title = anchor.select_one("h4.subsection-title")
            section_id = anchor.get("id", "").split("@")[-1]
            if section_title and section_id:
                sections.append(
                    {
                        "id": section_id,
                        "title": section_title.get_text(" ", strip=True),
                        "url": anchor.get("href"),
                        "graded": "graded" in (anchor.parent.get("class") or []),
                    }
                )
        chapters.append(
            {
                "id": chapter_id,
                "title": title_node.get_text(" ", strip=True),
                "sections": sections,
            }
        )
    return chapters


def clean_text(node: BeautifulSoup) -> str:
    for bad in node.select("script, style, button, input, textarea, .site-search"):
        bad.decompose()
    text = node.get_text("\n", strip=True)
    text = text.replace("ﬁ", "fi").replace("ﬂ", "fl")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def extract_section(session: requests.Session, chapter: dict, section: dict, root: Path) -> dict:
    response = get(session, section["url"])
    page_html = response.text
    section_dir = root / "raw" / slug(chapter["id"])
    section_dir.mkdir(parents=True, exist_ok=True)
    (section_dir / f"{slug(section['id'])}.html").write_text(page_html, encoding="utf-8")

    decoded = html.unescape(page_html)
    soup = BeautifulSoup(decoded, "lxml")
    units = []
    for unit in soup.select("div.xblock-public_view-vertical"):
        title_node = unit.select_one("h2.unit-title")
        title = title_node.get_text(" ", strip=True) if title_node else "Untitled unit"
        block_types = sorted(
            {
                block.get("data-block-type")
                for block in unit.select("[data-block-type]")
                if block.get("data-block-type")
            }
        )
        text = clean_text(BeautifulSoup(str(unit), "lxml"))
        units.append({"title": title, "types": block_types, "text": text})

    videos = []
    video_ids = sorted(set(re.findall(r"type@video\+block@([A-Za-z0-9_-]+)", decoded)))
    for video_id in video_ids:
        endpoint = (
            f"{BASE}/courses/{COURSE}/xblock/block-v1:MITx+6.036+1T2019"
            f"+type@video+block@{video_id}/handler/transcript/translation/en"
        )
        try:
            data = get(session, endpoint).json()
            transcript = "\n".join(data.get("text", []))
        except Exception:
            transcript = ""
        title_match = re.search(
            rf"<h3[^>]*>([^<]+)</h3>.*?type@video\+block@{re.escape(video_id)}",
            decoded,
            re.S,
        )
        videos.append(
            {
                "id": video_id,
                "title": html.unescape(title_match.group(1)).strip() if title_match else video_id,
                "transcript": transcript,
            }
        )

    pdf_urls = sorted(
        {
            urljoin(BASE, html.unescape(match))
            for match in re.findall(
                r'((?:/assets/courseware/|/asset-v1:)[^"<>& ]+?\.pdf)', decoded, re.I
            )
        }
    )
    lti_blocks = sorted(set(re.findall(r"type@lti\+block@([A-Za-z0-9_-]+)", decoded)))

    article_dir = root / "content" / slug(chapter["id"])
    article_dir.mkdir(parents=True, exist_ok=True)
    article_path = article_dir / f"{slug(section['id'])}.md"
    lines = [
        f"# {section['title']}",
        "",
        f"> {chapter['title']} · MIT 6.036 courseware archive",
        "",
    ]
    for unit in units:
        lines.extend([f"## {unit['title']}", "", unit["text"], ""])
    if videos:
        lines.extend(["## Video transcripts", ""])
        for video in videos:
            lines.extend([f"### {video['title']}", "", video["transcript"], ""])
    if pdf_urls:
        lines.extend(["## Source PDFs", ""])
        lines.extend([f"- {url}" for url in pdf_urls])
        lines.append("")
    article_path.write_text("\n".join(lines), encoding="utf-8")

    return {
        **section,
        "article": str(article_path.relative_to(root)),
        "units": [{"title": u["title"], "types": u["types"]} for u in units],
        "videos": [{"id": v["id"], "title": v["title"]} for v in videos],
        "video_count": len(videos),
        "pdf_urls": pdf_urls,
        "lti_blocks": lti_blocks,
        "characters": len(article_path.read_text(encoding="utf-8")),
    }


def download_pdf(session: requests.Session, url: str, target_dir: Path) -> dict:
    response = get(session, url)
    digest = hashlib.sha256(response.content).hexdigest()
    filename = Path(urlparse(url).path).name or f"{digest[:12]}.pdf"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / filename
    if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() != digest:
        target = target_dir / f"{target.stem}-{digest[:8]}{target.suffix}"
    target.write_bytes(response.content)
    return {"url": url, "path": str(target), "sha256": digest, "bytes": len(response.content)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("course-data"))
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    root = args.output
    root.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers["User-Agent"] = "Mozilla/5.0 course-archive/1.0"
    outline_text = get(session, OUTLINE_URL).text
    (root / "raw").mkdir(exist_ok=True)
    (root / "raw" / "course-outline.html").write_text(outline_text, encoding="utf-8")
    chapters = parse_outline(outline_text)

    work = [(chapter, section) for chapter in chapters for section in chapter["sections"]]
    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_map = {
            executor.submit(extract_section, session, chapter, section, root): section["id"]
            for chapter, section in work
        }
        for future in concurrent.futures.as_completed(future_map):
            section_id = future_map[future]
            try:
                results[section_id] = future.result()
                print(f"section {section_id}", flush=True)
            except Exception as error:
                results[section_id] = {"id": section_id, "error": str(error)}
                print(f"ERROR {section_id}: {error}", flush=True)

    for chapter in chapters:
        chapter["sections"] = [results.get(section["id"], section) for section in chapter["sections"]]

    pdf_urls = sorted(
        {
            url
            for chapter in chapters
            for section in chapter["sections"]
            for url in section.get("pdf_urls", [])
        }
    )
    pdfs = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(download_pdf, session, url, root / "pdfs") for url in pdf_urls]
        for future in concurrent.futures.as_completed(futures):
            try:
                pdfs.append(future.result())
            except Exception as error:
                pdfs.append({"error": str(error)})

    manifest = {
        "course": {"id": COURSE, "title": "MIT 6.036 Introduction to Machine Learning", "url": OUTLINE_URL},
        "chapters": chapters,
        "pdfs": sorted(pdfs, key=lambda item: item.get("url", "")),
    }
    (root / "course.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    summary = {
        "chapters": len(chapters),
        "sections": len(work),
        "videos": sum(s.get("video_count", 0) for c in chapters for s in c["sections"]),
        "lti_blocks": sum(len(s.get("lti_blocks", [])) for c in chapters for s in c["sections"]),
        "pdf_urls": len(pdf_urls),
    }
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
