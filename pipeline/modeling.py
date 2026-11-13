"""Chronological model selection. Only final scoring sees holdout labels."""
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC = ["temperature_c", "precipitation_mm", "wind_kmh", "day_index"]
CATEGORICAL = ["month", "weekday", "holiday", "workingday"]
FEATURES = NUMERIC + CATEGORICAL


def split(df):
    df = df.sort_values("date").reset_index(drop=True)
    first, second = int(len(df) * .6), int(len(df) * .8)
    if first < 30 or second == first:
        raise ValueError("Insufficient rows for temporal evaluation")
    return df.iloc[:first], df.iloc[first:second], df.iloc[second:]


def estimator(family):
    models = {"mean": DummyRegressor(strategy="mean"), "ridge": Ridge(alpha=10),
              "forest": RandomForestRegressor(n_estimators=200, min_samples_leaf=3, random_state=42, n_jobs=2),
              "boosting": HistGradientBoostingRegressor(max_iter=150, max_leaf_nodes=15, random_state=42)}
    transformer = ColumnTransformer([
        ("numeric", StandardScaler(), NUMERIC),
        ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL)])
    return make_pipeline(transformer, models[family])


def metrics(actual, predicted):
    return {"mae": float(mean_absolute_error(actual, predicted)),
            "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
            "r2": float(r2_score(actual, predicted))}


def validate(df, family):
    train, validation, _ = split(df)
    model = estimator(family).fit(train[FEATURES], train["rentals"])
    return metrics(validation["rentals"], model.predict(validation[FEATURES]))


def holdout(df, family):
    train, validation, test = split(df)
    import pandas as pd
    development = pd.concat([train, validation])
    model = estimator(family).fit(development[FEATURES], development["rentals"])
    predictions = model.predict(test[FEATURES])
    return metrics(test["rentals"], predictions), test, predictions

