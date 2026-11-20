"""Tests for src.osint_collector (strictly offline, no network calls)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import requests

from src.osint_collector import (
    INCIDENT_CATEGORIES,
    build_google_news_rss_url,
    build_google_news_rss_urls_for_window,
    build_incident_dataset,
    classify_incident,
    geocode_location,
    load_incidents_file,
    parse_rss_atom_xml,
)


@pytest.fixture(autouse=True)
def block_all_network_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    def _no_network(*args, **kwargs):
        raise AssertionError("Network call attempted during offline OSINT test")

    monkeypatch.setattr(requests, "get", _no_network)
    monkeypatch.setattr(requests.Session, "get", _no_network)


def test_bilingual_classifier_categories_and_severity() -> None:
    cat1, sev1, lang1 = classify_incident(
        "Choque grave y accidente vehicular en Libramiento Norte"
    )
    assert cat1 == "traffic_accident"
    assert lang1 == "es"
    assert 0.5 <= sev1 <= 1.0

    cat2, _, lang2 = classify_incident("Road closed due to major construction detour")
    assert cat2 == "road_closure"
    assert lang2 == "en"

    cat3, _, _ = classify_incident("Operativo de seguridad policial en zona centro")
    assert cat3 == "public_safety"

    cat4, _, _ = classify_incident("Tormenta con lluvia intensa e inundacion en avenida")
    assert cat4 == "weather_disruption"

    cat5, _, _ = classify_incident("Feria gastronomica de cafe y musica")
    assert cat5 == "other"
    assert set(INCIDENT_CATEGORIES) == {
        "traffic_accident",
        "road_closure",
        "public_safety",
        "weather_disruption",
    }


def test_parse_rss_and_atom_feeds_offline() -> None:
    rss_xml = Path("tests/fixtures/sample_feed.xml").read_text(encoding="utf-8")
    rss_df = parse_rss_atom_xml(rss_xml, source_name="SilaoRSS", relevant_only=True)

    assert len(rss_df) == 4
    assert set(rss_df["category"]) == set(INCIDENT_CATEGORIES)
    # Row 3 carried explicit georss:point
    weather = rss_df.loc[rss_df["category"] == "weather_disruption"].iloc[0]
    assert weather["lat"] == pytest.approx(20.9960)
    assert weather["lon"] == pytest.approx(-101.4820)
    # Row 1 resolved via offline Silao gazetteer
    acc = rss_df.loc[rss_df["category"] == "traffic_accident"].iloc[0]
    assert acc["geocode_method"] == "gazetteer"
    assert 20.90 <= acc["lat"] <= 21.02

    atom_xml = """<?xml version="1.0" encoding="utf-8"?>
    <feed xmlns="http://www.w3.org/2005/Atom">
      <title>Silao Atom</title>
      <entry>
        <id>atom-1</id>
        <title>Cierre vial en Boulevard Raul Bailleres</title>
        <summary>Obras de pavimentacion desde las 07:00</summary>
        <published>2026-09-10T07:30:00-06:00</published>
        <link href="https://example.local/atom/1"/>
      </entry>
    </feed>
    """
    atom_df = parse_rss_atom_xml(atom_xml, source_name="SilaoAtom")
    assert len(atom_df) == 1
    assert atom_df.iloc[0]["category"] == "road_closure"
    assert atom_df.iloc[0]["hour_int"] == 7


def test_offline_gazetteer_and_csv_json_ingestion(tmp_path: Path) -> None:
    lat, lon, method, matched = geocode_location("Accidente cerca de Puerto Interior")
    assert method == "gazetteer"
    assert matched == "puerto interior"
    assert 20.90 <= lat <= 21.05
    assert -101.60 <= lon <= -101.35

    csv_path = tmp_path / "inc.csv"
    csv_path.write_text(
        "date,hour_int,title,summary,location_name\n"
        "2026-09-12,9,Choque en Libramiento Norte,Trafico lento,Libramiento Norte\n",
        encoding="utf-8",
    )
    json_path = tmp_path / "inc.json"
    json_path.write_text(
        json.dumps(
            [
                {
                    "timestamp": "2026-09-14T18:20:00",
                    "title": "Heavy rain and flooding",
                    "summary": "Weather disruption near Zona Centro",
                    "location_name": "Zona Centro",
                }
            ]
        ),
        encoding="utf-8",
    )

    df_csv = load_incidents_file(csv_path)
    df_json = load_incidents_file(json_path)
    assert len(df_csv) == 1
    assert df_csv.iloc[0]["category"] == "traffic_accident"
    assert len(df_json) == 1
    assert df_json.iloc[0]["category"] == "weather_disruption"

    combined = build_incident_dataset(
        xml_texts=[Path("tests/fixtures/sample_feed.xml").read_text(encoding="utf-8")],
        file_paths=[csv_path, json_path],
        relevant_only=True,
    )
    assert len(combined) == 6
    assert "url" in build_google_news_rss_url("Silao accidente") or "news.google.com" in build_google_news_rss_url("Silao accidente")


def test_windowed_rss_url_builder_and_resilient_fetch_errors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    urls = build_google_news_rss_urls_for_window(
        start_date="2026-06-01",
        end_date="2026-08-15",
        chunk_days=30,
    )
    assert len(urls) == 3
    assert "after%3A2026-06-01" in urls[0]
    assert "before%3A2026-07-01" in urls[0]
    assert "after%3A2026-07-31" in urls[-1]

    sample_xml = Path("tests/fixtures/sample_feed.xml").read_text(encoding="utf-8")

    def _mock_fetch(url: str, **kwargs):
        if "2026-06-01" in url:
            raise requests.RequestException("simulated chunk failure")
        return parse_rss_atom_xml(sample_xml, source_name="MockRSS", relevant_only=True)

    import src.osint_collector as oc

    monkeypatch.setattr(oc, "fetch_rss_feed", _mock_fetch)

    with pytest.raises(requests.RequestException):
        build_incident_dataset(feed_urls=urls, ignore_fetch_errors=False)

    resilient = build_incident_dataset(
        feed_urls=urls, ignore_fetch_errors=True, relevant_only=True
    )
    assert len(resilient) == 4
