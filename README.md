# Bike Weather: Airflow Pipeline and Agent ML Workflow

Predict daily Capital Bikeshare rentals using calendar features and independent
historical weather data. This repository contains both workflows, their Docker
setup, evaluation outputs, and four screenshots from successful execution.

## Data and pipeline

- **Rentals:** [UCI Bike Sharing](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset),
  `day.csv`, 2011–2012. Attribution: Hadi Fanaee-T (2013), DOI
  `10.24432/C5W894`, CC BY 4.0.
- **Weather:** [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api),
  ERA5 mean temperature, precipitation and maximum wind speed for Washington DC
  (38.9072, -77.0369), using `America/New_York` daily dates. Attribution:
  Open-Meteo and ERA5/Copernicus data providers.

The two source tables are downloaded and cleaned in independent TaskGroups.
Cleaning parses dates/types, handles rental duplicates, checks weather units and
ranges, and rejects duplicate weather dates or missing weather. The tables are
joined one-to-one by date, requiring complete coverage: **731 daily rows**.
Calendar features are added, then the result is loaded transactionally into
PostgreSQL's `bike_weather` table. Repeated loads do not duplicate rows.
Analysis reads the cleaned table from the database, trains a mean baseline and
Ridge regression, and saves a plot and metrics in `artifacts/`.

`dags/bike_weather.py` schedules `bike_weather_daily` at **06:00 America/New_York**
every day. LocalExecutor permits the independent source tasks to run in parallel;
merging waits for both branches. `catchup=False` avoids historical backfills,
`max_active_runs=1` prevents overlapping runs, and failed tasks retry twice after
30 seconds. This refreshes a cached historical snapshot, not new daily observations.

```text
rentals: ingest → transform ─┐
                            ├→ merge_and_load → analyze (reads PostgreSQL)
weather: ingest → transform ─┘
```

## Modified agent workflow

Adapted from the [classroom research-loop design](https://github.com/Kedar-V/automl-research-lab/tree/refactor/lab-solution-layout),
reference commit `a294145ea9aa4225908ce701c06160c33213335d`.
`agents/graph.py` changes classification/F1 to rental regression/MAE, replaces CSV
input with SQL, uses chronological splits, and adds a database-quality agent.

```text
database_quality → data_scientist → hypothesis_generator → ml_engineer → judge
                                     ↑                                  │
                                     └─ context_builder ← memory_writer ┘ (revise)
judge → holdout_eval → END (accept or experiment budget exhausted)
```

The hypothesis agent selects Ridge, random forest or gradient boosting; the
engineer trains and validates; the judge routes according to validation MAE.
Memory records previous experiments and context prepares feedback. The loop stops
at MAE ≤ 700 or after three experiments. The Streamlit dashboard renders the
compiled LangGraph and updates it from actual execution events. These agents use
**deterministic policies without LLM calls**, corresponding to the classroom's
heuristic mode; no API key is required.

## Evaluation and key findings

Data is split in date order: 438 training days, 146 validation days, and 147 final
test days. Preprocessing is fit on training rows during selection. `casual` and
`registered` are excluded because their sum equals the target. Final model selection
uses validation only; the selected family is refit on train+validation and tested.

| Model | Validation MAE | Test MAE | Test RMSE | Test R² |
|---|---:|---:|---:|---:|
| Training-mean baseline | — | 2290.4 | 2560.6 | -0.866 |
| Ridge (selected) | 998.5 | 896.2 | 1227.2 | 0.571 |
| Random forest | 1853.4 | Not tested | Not tested | Not tested |
| Gradient boosting | 1519.1 | Not tested | Not tested | Not tested |

Ridge reduced test MAE by **60.9%** versus the mean baseline. None of the candidates
met the validation MAE threshold of 700, so the completed loop ended with
`budget_exhausted`. The nonlinear models did worse on the later validation period.
This is a retrospective experiment with observed historical weather, not a
validated next-day forecast or causal analysis. Two years from one city limit
generalization; Ridge can also produce negative count estimates.

## Run both workflows

Prerequisites: Docker Desktop running with Docker Compose; allow at least 4 GB
memory. From this repository directory:

```bash
mkdir -p data/raw data/processed artifacts
docker compose build
docker compose up airflow-init
# After initialization exits successfully:
docker compose up -d airflow-webserver airflow-scheduler agents
docker compose exec airflow-scheduler airflow dags list-import-errors
```

1. Open **http://localhost:8088**, login `admin` / `admin`. First web startup can
   take several minutes. Unpause `bike_weather_daily`, trigger it, and wait for
   all six tasks to succeed. Expand both TaskGroups in Graph view.
2. Open **http://localhost:8501** and click **Run research loop**. Run the DAG first
   so the PostgreSQL table exists. The agent graph and evaluation update live.
3. Alternatively run the agent workflow through the CLI:
   `docker compose run --rm agents python -m agents.graph`.
4. Outputs appear in `artifacts/`. Stop containers while preserving database data
   with `docker compose down`.

Airflow uses its own metadata database; the cleaned data lives in a separate
PostgreSQL warehouse. Raw data downloads are cached in `data/raw/` on first run.
Both Dockerfiles and dependency lists are included.

## Execution evidence

The scheduled run `scheduled__2026-11-11T11:00:00+00:00` and manual run
`assignment_verified` both succeeded on November 12, 2026; all six tasks succeeded
in each run. The agent dashboard also completed all three experiments against
PostgreSQL. Screenshots were captured from those real interfaces before the
project was moved into this submission folder; the workflow code is unchanged.

- [Successful Airflow execution](screenshots/01_airflow_success.jpg)
- [Airflow DAG graph](screenshots/02_airflow_graph.jpg)
- [Modified live agent graph](screenshots/03_agent_graph.jpg)
- [Model evaluation results](screenshots/04_agent_metrics.jpg)

Exact results: `artifacts/pipeline_metrics.json` and `artifacts/agent_metrics.json`.
The analysis plot and compiled agent graph are included in the same directory.
