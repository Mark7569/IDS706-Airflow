from datetime import datetime, timedelta
import csv

from airflow.decorators import dag, task, task_group
from scripts.prepare_data import download, clean_bikes, clean_weather, merge_rows, BIKE_URL, WEATHER_URL, PROCESSED


@dag(dag_id="bike_weather_daily", start_date=datetime(2024, 1, 1),
     schedule="0 6 * * *", catchup=False, max_active_runs=1,
     default_args={"retries": 2, "retry_delay": timedelta(seconds=30)},
     tags=["assignment", "regression"])
def bike_weather_daily():
    @task_group
    def rentals():
        @task
        def ingest():
            return str(download(BIKE_URL, "bikes.zip"))

        @task
        def transform(_source):
            clean_bikes()
            return str(PROCESSED / "bikes_clean.csv")
        return transform(ingest())

    @task_group
    def weather():
        @task
        def ingest():
            return str(download(WEATHER_URL, "weather.json"))

        @task
        def transform(_source):
            clean_weather()
            return str(PROCESSED / "weather_clean.csv")
        return transform(ingest())

    @task
    def merge_and_load(bike_path, weather_path):
        from pipeline.storage import load
        with open(bike_path) as source:
            bikes = list(csv.DictReader(source))
        with open(weather_path) as source:
            observations = list(csv.DictReader(source))
        rows = merge_rows(bikes, observations)
        for row in rows:
            for key in ["rentals", "holiday", "workingday", "month", "weekday", "day_index"]:
                row[key] = int(row[key])
            for key in ["temperature_c", "precipitation_mm", "wind_kmh"]:
                row[key] = float(row[key])
        return load(rows)

    @task
    def analyze(_loaded_rows):
        from pipeline.analyze import run
        return run()

    analyze(merge_and_load(rentals(), weather()))


dag = bike_weather_daily()
