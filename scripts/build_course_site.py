#!/usr/bin/env python3
"""Build the local MIT 6.036 website manifest from the crawled course archive."""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COURSE_DATA = ROOT / "course-data"
SITE = ROOT / "week1-blog"


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-") or "item"


def activity_type(title: str, block: str) -> str:
    value = f"{title} {block}".lower()
    if "homework" in value:
        return "homework"
    if "lab" in value:
        return "lab"
    if "exercise" in value or "assessment" in value:
        return "exercise"
    return "guide"


def main() -> int:
    course = json.loads((COURSE_DATA / "course.json").read_text(encoding="utf-8"))
    lti = json.loads((COURSE_DATA / "lti.json").read_text(encoding="utf-8"))["items"]
    lti_by_section: dict[tuple[str, str], list[dict]] = {}
    for item in lti:
        lti_by_section.setdefault((item["chapterId"], item["sectionId"]), []).append(item)

    chapters = []
    articles = {}
    for chapter_index, chapter in enumerate(course["chapters"]):
        chapter_slug = slug(chapter["id"])
        site_chapter = {
            "id": chapter_slug,
            "number": chapter_index,
            "title": chapter["title"],
            "sections": [],
        }
        for section in chapter["sections"]:
            article_id = f"{chapter_slug}-{slug(section['id'])}"
            activities = []
            for item in lti_by_section.get((chapter["id"], section["id"]), []):
                activity_id = f"activity-{slug(item['block'])}"
                kind = activity_type(section["title"], item["block"])
                text_path = item.get("text", f"lti/{item['block']}.txt")
                text_file = COURSE_DATA / text_path
                activity = {
                    "id": activity_id,
                    "title": item["block"].replace("_", " "),
                    "type": kind,
                    "url": f"../course-data/{text_path}",
                    "markdown": False,
                    "characters": item.get(
                        "characters",
                        len(text_file.read_text(encoding="utf-8")) if text_file.exists() else 0,
                    ),
                    "source_url": item.get("url", item["preview"]),
                }
                activities.append(activity)
                articles[activity_id] = {
                    **activity,
                    "kicker": f"{chapter['title']} · {kind.upper()}",
                    "deck": f"{section['title']} 的原版交互活动题面。网站仅整理题目与代码模板，不包含答案。",
                }

            public_article = {
                "id": article_id,
                "title": section["title"],
                "type": "notes" if not section["graded"] else "overview",
                "url": f"../course-data/{section['article']}",
                "markdown": True,
                "video_count": section.get("video_count", 0),
                "pdf_count": len(section.get("pdf_urls", [])),
                "characters": section.get("characters", 0),
                "activities": activities,
            }
            articles[article_id] = {
                **public_article,
                "kicker": f"{chapter['title']} · COURSE NOTES",
                "deck": (
                    f"融合课程正文、PDF 内容与 {section.get('video_count', 0)} 段视频 transcript "
                    f"的网页讲义。"
                ),
            }
            site_chapter["sections"].append(public_article)
        chapters.append(site_chapter)

    manifest = {
        "course": course["course"],
        "stats": {
            "chapters": len(chapters),
            "sections": sum(len(chapter["sections"]) for chapter in chapters),
            "videos": sum(
                section["video_count"] for chapter in chapters for section in chapter["sections"]
            ),
            "activities": len(lti),
            "pdfs": len(course["pdfs"]),
        },
        "chapters": chapters,
        "articles": articles,
    }
    target = SITE / "content" / "course.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest["stats"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
