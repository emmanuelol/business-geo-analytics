"""Geo-fenced OSINT incident collector, bilingual classifier, and offline Silao geocoder."""

from __future__ import annotations

import email.utils
import html
import json
import re
import unicodedata
import urllib.parse
import xml.etree.ElementTree as ET
from collections.abc import Iterable, Sequence
from datetime import date, datetime, timedelta
from pathlib import Path
import warnings
from typing import Any

import pandas as pd
import requests

from src.config import DEFAULT_LAT, DEFAULT_LON

INCIDENT_CATEGORIES: tuple[str, ...] = (
    "traffic_accident",
    "road_closure",
    "public_safety",
    "weather_disruption",
)

_CATEGORY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "traffic_accident": (
        "accidente",
        "choque",
        "volcadura",
        "colision",
        "atropellamiento",
        "carambola",
        "congestion",
        "trafico",
        "embotellamiento",
        "accident",
        "crash",
        "collision",
        "pileup",
        "overturned",
        "gridlock",
        "traffic jam",
    ),
    "road_closure": (
        "cierre",
        "cerrado",
        "cerrada",
        "bloqueo",
        "desvio",
        "obras",
        "pavimentacion",
        "bacheo",
        "manifestacion",
        "marcha",
        "socavon",
        "reparacion",
        "closure",
        "road closed",
        "detour",
        "construction",
        "blocked",
        "roadwork",
        "lane closed",
    ),
    "public_safety": (
        "seguridad",
        "operativo",
        "balacera",
        "detencion",
        "asalto",
        "robo",
        "policia",
        "policial",
        "guardia nacional",
        "acordonamiento",
        "incendio",
        "security",
        "police",
        "shooting",
        "robbery",
        "lockdown",
        "evacuation",
        "fire",
    ),
    "weather_disruption": (
        "lluvia",
        "tormenta",
        "inundacion",
        "granizo",
        "encharcamiento",
        "viento",
        "frente frio",
        "desbordamiento",
        "rain",
        "storm",
        "thunderstorm",
        "flood",
        "flooding",
        "hail",
        "downpour",
        "weather disruption",
    ),
}

_HIGH_SEVERITY_TERMS: tuple[str, ...] = (
    "grave",
    "severo",
    "severa",
    "total",
    "fatal",
    "multiple",
    "bloqueo total",
    "cierre total",
    "inundacion",
    "severe",
    "major",
    "flooding",
    "heavy",
    "gridlock",
    "emergency",
)

_ES_MARKERS = (" en ", " de ", " la ", " el ", " por ", " con ", " del ", " vial", " cierre", " choque")
_EN_MARKERS = (" the ", " near ", " due to ", " and ", " with ", " road ", " traffic", " closed")

# Approximate Silao, Guanajuato gazetteer (longest key match wins)
SILAO_GAZETTEER: dict[str, tuple[float, float]] = {
    "paseo ms": (DEFAULT_LAT, DEFAULT_LON),
    "el progreso": (20.9510, -101.4215),
    "libramiento norte": (20.9545, -101.4240),
    "blvd raul bailleres": (20.9438, -101.4265),
    "boulevard raul bailleres": (20.9438, -101.4265),
    "raul bailleres": (20.9438, -101.4265),
    "blvd emiliano zapata": (20.9412, -101.4302),
    "zona centro": (20.9434, -101.4272),
    "centro silao": (20.9434, -101.4272),
    "alameda": (20.9461, -101.4258),
    "puerto interior": (20.9960, -101.4820),
    "parque bicentenario": (20.9718, -101.3905),
    "aeropuerto del bajio": (20.9935, -101.4809),
    "carretera silao leon": (20.9780, -101.4560),
    "autopista salamanca leon": (20.9350, -101.4100),
    "corredor industrial": (20.9850, -101.4680),
    "silao": (DEFAULT_LAT, DEFAULT_LON),
}

STANDARD_INCIDENT_COLUMNS: tuple[str, ...] = (
    "incident_id",
    "timestamp",
    "date",
    "hour_int",
    "title",
    "summary",
    "source",
    "category",
    "severity",
    "lat",
    "lon",
    "location_name",
    "language",
    "geocode_method",
    "link",
)


def _normalize_text(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", str(text))
    ascii_only = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", ascii_only.lower()).strip()


def _strip_html(text: str) -> str:
    unescaped = html.unescape(str(text or ""))
    no_tags = re.sub(r"<[^>]+>", " ", unescaped)
    return re.sub(r"\s+", " ", no_tags).strip()


def _local_tag(tag: str) -> str:
    return tag.split("}", 1)[-1].lower()


def classify_incident(text: str) -> tuple[str, float, str]:
    """Classify Spanish/English incident text into (category, severity, language)."""
    norm = f" {_normalize_text(text)} "

    es_hits = sum(1 for tok in _ES_MARKERS if tok in norm)
    en_hits = sum(1 for tok in _EN_MARKERS if tok in norm)
    language = "en" if en_hits > es_hits else "es"

    scores: dict[str, int] = {}
    for category, keywords in _CATEGORY_KEYWORDS.items():
        hits = sum(1 for kw in keywords if kw in norm)
        if hits > 0:
            scores[category] = hits

    if not scores:
        return "other", 0.3, language

    best_cat = max(scores, key=lambda k: scores[k])
    sev_boost = sum(1 for term in _HIGH_SEVERITY_TERMS if term in norm)
    severity = min(1.0, round(0.45 + 0.12 * scores[best_cat] + 0.12 * sev_boost, 2))
    return best_cat, severity, language


def geocode_location(
    text_or_place: str,
    default_lat: float = DEFAULT_LAT,
    default_lon: float = DEFAULT_LON,
    use_nominatim: bool = False,
    timeout: float = 5.0,
) -> tuple[float, float, str, str]:
    """Resolve location via offline Silao gazetteer, with optional opt-in Nominatim."""
    norm = _normalize_text(text_or_place)
    for key in sorted(SILAO_GAZETTEER.keys(), key=len, reverse=True):
        if key != "silao" and key in norm:
            lat, lon = SILAO_GAZETTEER[key]
            return lat, lon, "gazetteer", key

    if use_nominatim and norm:
        try:
            resp = requests.get(
                "https://nominatim.openstreetmap.org/search",
                params={"q": f"{text_or_place}, Silao, Guanajuato, Mexico", "format": "json", "limit": 1},
                headers={"User-Agent": "business-geo-analytics/0.1 (research)"},
                timeout=timeout,
            )
            resp.raise_for_status()
            payload = resp.json()
            if payload:
                return float(payload[0]["lat"]), float(payload[0]["lon"]), "nominatim", text_or_place
        except Exception:
            pass

    if "silao" in norm:
        lat, lon = SILAO_GAZETTEER["silao"]
        return lat, lon, "gazetteer", "silao"

    return float(default_lat), float(default_lon), "default_centroid", "default"


def _parse_timestamp(raw_ts: str | None) -> tuple[str, str, int]:
    if not raw_ts:
        return "", "", 12
    raw = str(raw_ts).strip()
    dt: datetime | None = None
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        try:
            dt = email.utils.parsedate_to_datetime(raw)
        except (TypeError, ValueError):
            dt = None
    if dt is not None:
        return dt.strftime("%Y-%m-%dT%H:%M:%S"), dt.strftime("%Y-%m-%d"), int(dt.hour)

    m = re.search(r"(\d{4}-\d{2}-\d{2})", raw)
    date_part = m.group(1) if m else ""
    return raw, date_part, 12


def _extract_child_text(parent: ET.Element, *names: str) -> str:
    targets = {n.lower() for n in names}
    for child in parent:
        if _local_tag(child.tag) in targets:
            if child.text and child.text.strip():
                return child.text.strip()
            href = child.attrib.get("href")
            if href:
                return href.strip()
    return ""


def _extract_georss(parent: ET.Element) -> tuple[float | None, float | None]:
    for child in parent:
        tag = _local_tag(child.tag)
        if tag == "point" and child.text:
            parts = child.text.strip().split()
            if len(parts) >= 2:
                try:
                    return float(parts[0]), float(parts[1])
                except ValueError:
                    pass
    lat_txt = _extract_child_text(parent, "lat")
    lon_txt = _extract_child_text(parent, "long", "lon")
    if lat_txt and lon_txt:
        try:
            return float(lat_txt), float(lon_txt)
        except ValueError:
            pass
    return None, None


def _normalize_record(
    raw: dict[str, Any],
    idx: int,
    use_nominatim: bool = False,
) -> dict[str, Any]:
    title = _strip_html(str(raw.get("title") or ""))
    summary = _strip_html(str(raw.get("summary") or raw.get("description") or ""))
    combined_text = f"{title} {summary} {raw.get('location_name') or ''}".strip()

    auto_cat, auto_sev, auto_lang = classify_incident(combined_text)
    category = str(raw.get("category") or "").strip() or auto_cat
    severity_val = raw.get("severity")
    severity = float(severity_val) if severity_val not in (None, "", "nan") else auto_sev
    language = str(raw.get("language") or "").strip() or auto_lang

    ts_iso, date_str, hour_val = _parse_timestamp(
        raw.get("timestamp") or raw.get("pubDate") or raw.get("published") or raw.get("date")
    )
    if raw.get("date") and not date_str:
        date_str = str(raw["date"])[:10]
    if raw.get("hour_int") not in (None, "", "nan") and not pd.isna(raw.get("hour_int")):
        hour_val = int(float(raw["hour_int"]))
    if not ts_iso and date_str:
        ts_iso = f"{date_str}T{hour_val:02d}:00:00"

    raw_lat = raw.get("lat")
    raw_lon = raw.get("lon")
    if (
        raw_lat not in (None, "", "nan")
        and raw_lon not in (None, "", "nan")
        and not pd.isna(raw_lat)
        and not pd.isna(raw_lon)
    ):
        lat, lon, method = float(raw_lat), float(raw_lon), "explicit"
        loc_name = str(raw.get("location_name") or "explicit_coords")
    else:
        place_hint = str(raw.get("location_name") or combined_text)
        lat, lon, method, matched = geocode_location(
            place_hint, use_nominatim=use_nominatim
        )
        loc_name = str(raw.get("location_name") or matched)

    return {
        "incident_id": str(raw.get("incident_id") or raw.get("guid") or f"OSINT-{idx + 1:04d}"),
        "timestamp": ts_iso,
        "date": date_str,
        "hour_int": hour_val,
        "title": title,
        "summary": summary,
        "source": str(raw.get("source") or "unknown"),
        "category": category,
        "severity": round(float(severity), 2),
        "lat": round(float(lat), 6),
        "lon": round(float(lon), 6),
        "location_name": loc_name,
        "language": language,
        "geocode_method": method,
        "link": str(raw.get("link") or ""),
    }


def parse_rss_atom_xml(
    xml_text: str,
    source_name: str | None = None,
    use_nominatim: bool = False,
    relevant_only: bool = False,
) -> pd.DataFrame:
    """Parse RSS 2.0 or Atom XML string into normalized incident DataFrame."""
    root = ET.fromstring(xml_text)
    root_tag = _local_tag(root.tag)
    feed_title = _extract_child_text(root, "title")
    items: list[ET.Element] = []

    if root_tag == "rss":
        channel = next((c for c in root if _local_tag(c.tag) == "channel"), None)
        if channel is not None:
            feed_title = _extract_child_text(channel, "title") or feed_title
            items = [c for c in channel if _local_tag(c.tag) == "item"]
    elif root_tag == "feed":
        items = [c for c in root if _local_tag(c.tag) == "entry"]
    else:
        items = [el for el in root.iter() if _local_tag(el.tag) in {"item", "entry"}]

    source = source_name or feed_title or "rss_feed"
    records: list[dict[str, Any]] = []
    for idx, elem in enumerate(items):
        lat, lon = _extract_georss(elem)
        raw = {
            "incident_id": _extract_child_text(elem, "guid", "id") or f"{source}-{idx + 1:03d}",
            "title": _extract_child_text(elem, "title"),
            "summary": _extract_child_text(elem, "description", "summary", "content"),
            "link": _extract_child_text(elem, "link"),
            "timestamp": _extract_child_text(elem, "pubdate", "published", "updated"),
            "source": source,
            "lat": lat,
            "lon": lon,
        }
        norm = _normalize_record(raw, idx=idx, use_nominatim=use_nominatim)
        if relevant_only and norm["category"] not in INCIDENT_CATEGORIES:
            continue
        records.append(norm)

    if not records:
        return pd.DataFrame(columns=STANDARD_INCIDENT_COLUMNS)
    return pd.DataFrame(records, columns=STANDARD_INCIDENT_COLUMNS)


def build_google_news_rss_url(
    query: str = "Silao Guanajuato (accidente OR cierre OR trafico OR lluvia OR seguridad)",
    hl: str = "es-419",
    gl: str = "MX",
) -> str:
    """Build a localized Google News RSS search URL for opt-in OSINT collection."""
    params = urllib.parse.urlencode({"q": query, "hl": hl, "gl": gl, "ceid": f"{gl}:es-419"})
    return f"https://news.google.com/rss/search?{params}"


def build_google_news_rss_urls_for_window(
    start_date: str | date | datetime,
    end_date: str | date | datetime,
    query: str = "Silao Guanajuato (accidente OR cierre OR trafico OR lluvia OR seguridad)",
    chunk_days: int = 30,
    hl: str = "es-419",
    gl: str = "MX",
) -> list[str]:
    """Generate date-bounded Google News RSS query URLs across a historical window."""
    if chunk_days <= 0:
        raise ValueError(f"chunk_days must be positive, got {chunk_days}")
    start_d = (
        start_date.date()
        if isinstance(start_date, datetime)
        else start_date
        if isinstance(start_date, date)
        else date.fromisoformat(str(start_date)[:10])
    )
    end_d = (
        end_date.date()
        if isinstance(end_date, datetime)
        else end_date
        if isinstance(end_date, date)
        else date.fromisoformat(str(end_date)[:10])
    )
    if end_d < start_d:
        raise ValueError(f"end_date ({end_d}) must be >= start_date ({start_d})")

    urls: list[str] = []
    cur = start_d
    limit_end = end_d + timedelta(days=1)
    while cur <= end_d:
        nxt = min(cur + timedelta(days=chunk_days), limit_end)
        chunk_query = f"{query} after:{cur.isoformat()} before:{nxt.isoformat()}"
        urls.append(build_google_news_rss_url(query=chunk_query, hl=hl, gl=gl))
        cur = nxt
    return urls


def fetch_rss_feed(
    url: str,
    source_name: str | None = None,
    timeout: float = 10.0,
    use_nominatim: bool = False,
    relevant_only: bool = True,
) -> pd.DataFrame:
    """Opt-in network fetcher for RSS/Atom feeds."""
    resp = requests.get(
        url,
        headers={"User-Agent": "business-geo-analytics/0.1"},
        timeout=timeout,
    )
    resp.raise_for_status()
    return parse_rss_atom_xml(
        resp.text,
        source_name=source_name,
        use_nominatim=use_nominatim,
        relevant_only=relevant_only,
    )


def load_incidents_file(
    path: Path | str,
    use_nominatim: bool = False,
    relevant_only: bool = False,
) -> pd.DataFrame:
    """Load incidents from CSV or JSON and normalize categories, timestamps, and coords."""
    file_path = Path(path)
    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        raw_df = pd.read_csv(file_path)
        raw_records = raw_df.to_dict(orient="records")
    elif suffix == ".json":
        payload = json.loads(file_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and "incidents" in payload:
            raw_records = list(payload["incidents"])
        elif isinstance(payload, list):
            raw_records = payload
        else:
            raise ValueError(f"Unsupported JSON structure in {file_path}")
    else:
        raise ValueError(f"Unsupported file format {suffix!r}; expected .csv or .json")

    rows: list[dict[str, Any]] = []
    for idx, rec in enumerate(raw_records):
        norm = _normalize_record(rec, idx=idx, use_nominatim=use_nominatim)
        if relevant_only and norm["category"] not in INCIDENT_CATEGORIES:
            continue
        rows.append(norm)

    if not rows:
        return pd.DataFrame(columns=STANDARD_INCIDENT_COLUMNS)
    return pd.DataFrame(rows, columns=STANDARD_INCIDENT_COLUMNS)


def build_incident_dataset(
    xml_texts: Iterable[str] = (),
    file_paths: Iterable[Path | str] = (),
    feed_urls: Sequence[str] = (),
    use_nominatim: bool = False,
    relevant_only: bool = True,
    ignore_fetch_errors: bool = False,
) -> pd.DataFrame:
    """Combine XML feeds, CSV/JSON files, and optional live URLs into one clean DataFrame."""
    frames: list[pd.DataFrame] = []
    for idx, xml_text in enumerate(xml_texts):
        frames.append(
            parse_rss_atom_xml(
                xml_text,
                source_name=f"xml_feed_{idx + 1}",
                use_nominatim=use_nominatim,
                relevant_only=relevant_only,
            )
        )
    for fp in file_paths:
        frames.append(
            load_incidents_file(
                fp, use_nominatim=use_nominatim, relevant_only=relevant_only
            )
        )
    for url in feed_urls:
        try:
            frames.append(
                fetch_rss_feed(
                    url, use_nominatim=use_nominatim, relevant_only=relevant_only
                )
            )
        except Exception as exc:
            if not ignore_fetch_errors:
                raise
            warnings.warn(f"Skipping failed RSS URL {url}: {exc}", RuntimeWarning, stacklevel=2)

    non_empty = [f for f in frames if not f.empty]
    if not non_empty:
        return pd.DataFrame(columns=STANDARD_INCIDENT_COLUMNS)

    combined = pd.concat(non_empty, ignore_index=True)
    combined["_dedup_title"] = combined["title"].map(_normalize_text)
    combined = (
        combined.drop_duplicates(subset=["date", "_dedup_title"])
        .drop(columns=["_dedup_title"])
        .sort_values(["date", "hour_int", "incident_id"])
        .reset_index(drop=True)
    )
    return combined
