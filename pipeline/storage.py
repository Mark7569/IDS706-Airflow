"""Transactional, idempotent loading and explicit database reads."""
import os
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text

ROOT = Path(__file__).resolve().parents[1]


def engine():
    # SQLite is an explicit local verification mode; Compose uses PostgreSQL.
    url = os.environ.get("DATA_DATABASE_URL")
    if not url:
        raise RuntimeError("Set DATA_DATABASE_URL; see README")
    return create_engine(url)


def load(rows):
    with engine().begin() as connection:
        connection.execute(text("""CREATE TABLE IF NOT EXISTS bike_weather (
            date DATE PRIMARY KEY, rentals INTEGER NOT NULL CHECK (rentals >= 0),
            holiday INTEGER NOT NULL, workingday INTEGER NOT NULL,
            temperature_c FLOAT NOT NULL, precipitation_mm FLOAT NOT NULL,
            wind_kmh FLOAT NOT NULL, month INTEGER NOT NULL,
            weekday INTEGER NOT NULL, day_index INTEGER NOT NULL)"""))
        # A single transaction ensures readers see either the old or new snapshot.
        connection.execute(text("DELETE FROM bike_weather"))
        connection.execute(text("""INSERT INTO bike_weather
            (date,rentals,holiday,workingday,temperature_c,precipitation_mm,wind_kmh,month,weekday,day_index)
            VALUES (:date,:rentals,:holiday,:workingday,:temperature_c,:precipitation_mm,:wind_kmh,:month,:weekday,:day_index)"""), rows)
    return len(rows)


def read():
    with engine().connect() as connection:
        result = connection.execute(text("SELECT * FROM bike_weather ORDER BY date"))
        return pd.DataFrame(result.fetchall(), columns=list(result.keys()))

