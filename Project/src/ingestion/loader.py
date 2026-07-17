"""
Data source loader: arXiv API (humanoid robotics / legged locomotion papers).

API details worth knowing (from the official user manual —
https://info.arxiv.org/help/api/user-manual.html):
- No auth required. Responses are Atom 1.0 XML, not JSON — that's why this
  file parses XML instead of calling resp.json() like most REST APIs you'll
  hit.
- Terms of Use require a delay between paged requests and cap page size at
  2000 results. We page in smaller batches (100) and sleep 3s between
  requests — non-negotiable if you want your ingestion script to not get
  your IP throttled or blocked mid-run.
- search_query supports field prefixes (ti:, abs:, cat:, au:) combined with
  AND/OR/ANDNOT. SEARCH_QUERY below is the one thing you'll want to edit
  for a different topic.
"""
from __future__ import annotations

import logging
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Iterator

import requests

logger = logging.getLogger(__name__)

ARXIV_API_URL = "http://export.arxiv.org/api/query"
ATOM_NS = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}

# cs.RO = Robotics category. Edit this query for your own topic —
# see https://info.arxiv.org/help/api/user-manual.html#query_details
SEARCH_QUERY = 'cat:cs.RO AND (abs:"humanoid" OR abs:"legged locomotion" OR abs:"bipedal")'
PAGE_SIZE = 100                # results per HTTP request
MAX_RESULTS = 2000             # total papers to pull across all pages
REQUEST_DELAY_SECONDS = 3.0    # arXiv Terms of Use minimum between paged requests — do not lower this


@dataclass
class RawDocument:
    """One unit of source content before chunking."""
    source_id: str
    text: str
    metadata: dict


def _fetch_page(start: int, max_results: int) -> str:
    params = {
        "search_query": SEARCH_QUERY,
        "start": start,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    resp = requests.get(ARXIV_API_URL, params=params, timeout=30)
    resp.raise_for_status()
    return resp.text


def _text(entry: ET.Element, tag: str) -> str | None:
    el = entry.find(f"atom:{tag}", ATOM_NS)
    return el.text.strip() if el is not None and el.text else None


def _parse_entries(xml_text: str) -> list[RawDocument]:
    """
    Parse one page of Atom XML into RawDocuments.

    Kept as a standalone function (not inlined into the pagination loop)
    specifically so it's unit-testable against a saved XML fixture without
    hitting the live API — see tests/test_arxiv_loader.py.
    """
    root = ET.fromstring(xml_text)
    docs: list[RawDocument] = []

    for entry in root.findall("atom:entry", ATOM_NS):
        entry_id = _text(entry, "id")
        if entry_id is None:
            continue  # malformed entry; skip rather than crash the whole page

        # entry_id looks like "http://arxiv.org/abs/2401.01234v2" -- the
        # short id (with version) is the stable identifier we upsert on.
        # We deliberately keep the version suffix: a v1 -> v2 revision is a
        # real content change and should re-embed, not silently overwrite
        # under a shared unversioned id.
        short_id = entry_id.rsplit("/abs/", 1)[-1]

        title = (_text(entry, "title") or "").replace("\n", " ").strip()
        summary = (_text(entry, "summary") or "").replace("\n", " ").strip()
        published = _text(entry, "published")

        authors = [
            a.find("atom:name", ATOM_NS).text
            for a in entry.findall("atom:author", ATOM_NS)
            if a.find("atom:name", ATOM_NS) is not None
        ]

        primary_cat_el = entry.find("arxiv:primary_category", ATOM_NS)
        primary_category = primary_cat_el.get("term") if primary_cat_el is not None else None

        pdf_url = next(
            (link.get("href") for link in entry.findall("atom:link", ATOM_NS) if link.get("title") == "pdf"),
            None,
        )

        if not title or not summary:
            logger.warning("Skipping entry %s: missing title or abstract", short_id)
            continue

        # Text body is title + abstract, not full text. Full text would
        # mean downloading and parsing each PDF (see the pdf-reading skill
        # for that pattern) -- a reasonable v2 extension, but abstracts
        # alone give a reliably clean corpus without PDF-parsing failure
        # modes (scanned pages, broken columns, etc.) polluting your chunks.
        text = f"{title}\n\n{summary}"

        docs.append(
            RawDocument(
                source_id=short_id,
                text=text,
                metadata={
                    "title": title,
                    "authors": authors,
                    "published": published,
                    "primary_category": primary_category,
                    "pdf_url": pdf_url,
                    "arxiv_url": entry_id,
                },
            )
        )

    return docs


def load_raw_documents() -> Iterator[RawDocument]:
    """
    Page through arXiv API results for SEARCH_QUERY, yielding one
    RawDocument per paper. Generator, not a list — keeps memory flat and
    lets the ingestion pipeline start embedding/upserting the first page
    while later pages are still being fetched.
    """
    start = 0
    total_yielded = 0

    while total_yielded < MAX_RESULTS:
        page_size = min(PAGE_SIZE, MAX_RESULTS - total_yielded)
        logger.info("Fetching arXiv results %d-%d", start, start + page_size)

        try:
            xml_text = _fetch_page(start, page_size)
        except requests.RequestException:
            # Network hiccup mid-run shouldn't crash the whole ingestion job
            # with a partial corpus and no explanation — log and stop
            # cleanly; the idempotent upsert means rerunning later is safe.
            logger.exception("arXiv API request failed at start=%d, stopping pagination", start)
            return

        entries = _parse_entries(xml_text)
        if not entries:
            logger.info("No more results at start=%d — pagination complete", start)
            return

        yield from entries
        total_yielded += len(entries)
        start += len(entries)

        time.sleep(REQUEST_DELAY_SECONDS)

    logger.info("Reached MAX_RESULTS=%d, stopping", MAX_RESULTS)
