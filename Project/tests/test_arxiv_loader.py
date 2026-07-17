"""
Tests for the arXiv loader's XML parsing, run against a saved fixture
rather than the live API. Keeps the test fast, offline, and immune to
arXiv API downtime -- and lets you verify the parser survives edge cases
(missing fields, multiple authors) without needing 3s-delayed live calls.
"""
from src.ingestion.loader import _parse_entries

SAMPLE_ATOM_XML = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/2401.01234v2</id>
    <published>2024-01-15T10:00:00Z</published>
    <updated>2024-02-01T12:00:00Z</updated>
    <title>Robust Bipedal Locomotion via Reinforcement Learning</title>
    <summary>
      We present a reinforcement learning approach for humanoid bipedal
      locomotion that generalizes across terrain types without retraining.
    </summary>
    <author><name>Jane Researcher</name></author>
    <author><name>John Coauthor</name></author>
    <arxiv:primary_category term="cs.RO" scheme="http://arxiv.org/schemas/atom"/>
    <link href="http://arxiv.org/abs/2401.01234v2" rel="alternate" type="text/html"/>
    <link title="pdf" href="http://arxiv.org/pdf/2401.01234v2" rel="related" type="application/pdf"/>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2402.05678v1</id>
    <published>2024-02-10T08:30:00Z</published>
    <title>Whole-Body Control for Humanoid Robots: A Survey</title>
    <summary>A survey of whole-body control methods for humanoid platforms.</summary>
    <author><name>Alex Author</name></author>
    <arxiv:primary_category term="cs.RO" scheme="http://arxiv.org/schemas/atom"/>
    <link title="pdf" href="http://arxiv.org/pdf/2402.05678v1" rel="related" type="application/pdf"/>
  </entry>
</feed>
"""

SAMPLE_ATOM_MALFORMED_ENTRY = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/2403.09999v1</id>
    <title>Paper With No Abstract</title>
  </entry>
</feed>
"""


def test_parses_expected_number_of_entries():
    docs = _parse_entries(SAMPLE_ATOM_XML)
    assert len(docs) == 2


def test_source_id_keeps_version_suffix():
    docs = _parse_entries(SAMPLE_ATOM_XML)
    assert docs[0].source_id == "2401.01234v2"


def test_text_combines_title_and_abstract():
    docs = _parse_entries(SAMPLE_ATOM_XML)
    assert "Robust Bipedal Locomotion" in docs[0].text
    assert "generalizes across terrain types" in docs[0].text


def test_multiple_authors_captured():
    docs = _parse_entries(SAMPLE_ATOM_XML)
    assert docs[0].metadata["authors"] == ["Jane Researcher", "John Coauthor"]


def test_single_author_captured():
    docs = _parse_entries(SAMPLE_ATOM_XML)
    assert docs[1].metadata["authors"] == ["Alex Author"]


def test_metadata_fields_present():
    docs = _parse_entries(SAMPLE_ATOM_XML)
    meta = docs[0].metadata
    assert meta["primary_category"] == "cs.RO"
    assert meta["pdf_url"] == "http://arxiv.org/pdf/2401.01234v2"
    assert meta["published"] == "2024-01-15T10:00:00Z"


def test_entry_missing_abstract_is_skipped_not_crashed():
    # A paper entry with no <summary> shouldn't take down the whole page
    # of results -- it should be skipped and logged.
    docs = _parse_entries(SAMPLE_ATOM_MALFORMED_ENTRY)
    assert docs == []


def test_empty_feed_returns_empty_list():
    empty_feed = '<feed xmlns="http://www.w3.org/2005/Atom"></feed>'
    assert _parse_entries(empty_feed) == []
