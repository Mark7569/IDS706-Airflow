"""First learning checkpoint. Independent source tasks, no third-party packages.

Run: python3 scripts/prepare_data.py
Functions will be reused by the Airflow DAG in the next checkpoint.
"""
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path
import csv
import hashlib
import io
import json
import urllib.parse
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
BIKE_URL = "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip"
WEATHER_URL = "https://archive-api.open-meteo.com/v1/archive?" + urllib.parse.urlencode({
    "latitude": 38.9072, "longitude": -77.0369,
    "start_date": "2011-01-01", "end_date": "2012-12-31",
    "daily": "temperature_2m_mean,precipitation_sum,wind_speed_10m_max",
    "timezone": "America/New_York", "models": "era5",
})


def download(url, filename):
    RAW.mkdir(parents=True, exist_ok=True)
    path = RAW / filename
    if not path.exists():
        request = urllib.request.Request(url, headers={"User-Agent": "BikeWeatherClassProject/1.0"})
        with urllib.request.urlopen(request, timeout=120) as response:
            payload = response.read()
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_bytes(payload)
        temporary.replace(path)
    return path


def save_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def clean_bikes():
    source = download(BIKE_URL, "bikes.zip")
    with zipfile.ZipFile(source) as archive:
        names = [n for n in archive.namelist() if n.split("/")[-1] == "day.csv"]
        if len(names) != 1:
            raise ValueError("Expected exactly one day.csv")
        original = list(csv.DictReader(io.StringIO(archive.read(names[0]).decode("utf-8-sig"))))
    rows, seen, duplicates = [], set(), 0
    for item in original:
        day = date.fromisoformat(item["dteday"])
        if day in seen:
            duplicates += 1
            continue
        count = int(item["cnt"])
        if count < 0:
            raise ValueError("Negative rental count")
        # casual + registered = cnt: excluding both prevents target leakage.
        rows.append({"date": day.isoformat(), "rentals": count,
                     "holiday": int(item["holiday"]), "workingday": int(item["workingday"])})
        seen.add(day)
    rows.sort(key=lambda row: row["date"])
    save_csv(PROCESSED / "bikes_clean.csv", rows)
    return rows, {"input_rows": len(original), "output_rows": len(rows), "duplicates_removed": duplicates}


def clean_weather():
    source = download(WEATHER_URL, "weather.json")
    data = json.loads(source.read_text())
    daily = data["daily"]
    expected_units = {"temperature_2m_mean": "°C", "precipitation_sum": "mm", "wind_speed_10m_max": "km/h"}
    for field, unit in expected_units.items():
        if data["daily_units"][field] != unit:
            raise ValueError(f"Unexpected units for {field}")
    rows, seen = [], set()
    for index, stamp in enumerate(daily["time"]):
        day = date.fromisoformat(stamp)
        if day in seen:
            raise ValueError("Duplicate weather date")
        row = {"date": day.isoformat()}
        for source_field, field in [("temperature_2m_mean", "temperature_c"),
                                    ("precipitation_sum", "precipitation_mm"),
                                    ("wind_speed_10m_max", "wind_kmh")]:
            value = daily[source_field][index]
            if value is None:
                raise ValueError(f"Missing {field} at {day}; do not silently invent weather")
            row[field] = float(value)
        if not (-60 <= row["temperature_c"] <= 60) or min(row["precipitation_mm"], row["wind_kmh"]) < 0:
            raise ValueError(f"Implausible weather at {day}")
        rows.append(row)
        seen.add(day)
    rows.sort(key=lambda row: row["date"])
    save_csv(PROCESSED / "weather_clean.csv", rows)
    return rows, {"input_rows": len(daily["time"]), "output_rows": len(rows), "missing_values": 0}


def merge_rows(bikes, weather):
    lookup = {row["date"]: row for row in weather}
    if {row["date"] for row in bikes} != set(lookup):
        raise ValueError("Source date coverage differs; inspect before merging")
    result = []
    for row in bikes:
        day = date.fromisoformat(row["date"])
        result.append({**row, **lookup[row["date"]], "month": day.month,
                       "weekday": day.weekday(), "day_index": (day - date(2011, 1, 1)).days})
    expected_dates = {(date(2011, 1, 1) + timedelta(days=i)).isoformat() for i in range(731)}
    if {row["date"] for row in result} != expected_dates:
        raise ValueError("Expected all 731 dates in 2011–2012")
    save_csv(PROCESSED / "bike_weather.csv", result)
    return result


def main():
    with ThreadPoolExecutor(max_workers=2) as pool:
        bike_task = pool.submit(clean_bikes)
        weather_task = pool.submit(clean_weather)
        bikes, bike_audit = bike_task.result()
        weather, weather_audit = weather_task.result()
    merged = merge_rows(bikes, weather)
    audit = {"sources": {"bikes": BIKE_URL, "weather": WEATHER_URL},
             "sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(RAW.iterdir()) if p.is_file() and not p.name.startswith(".")},
             "bikes": bike_audit, "weather": weather_audit,
             "merged_rows": len(merged), "join": "one-to-one on local date",
             "date_range": [merged[0]["date"], merged[-1]["date"]]}
    (PROCESSED / "quality_report.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
