import json
import operator
from pathlib import Path
from typing import Annotated, TypedDict

import pandas as pd
from langgraph.graph import StateGraph, START, END

from pipeline.storage import read
from pipeline.modeling import validate, holdout, split, FEATURES

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"


class State(TypedDict, total=False):
    data: pd.DataFrame
    cycle: int
    max_cycles: int
    target_mae: float
    family: str
    validation: dict
    best_family: str
    best_mae: float
    memory: Annotated[list, operator.add]
    brief: str
    verdict: str
    profile: dict
    holdout_metrics: dict
    baseline_metrics: dict
    steps: Annotated[list, operator.add]


def database_quality(state):
    df = read()
    if df.empty or df.date.duplicated().any() or df[FEATURES + ["rentals"]].isna().any().any():
        raise ValueError("Database quality gate failed")
    if len(df) != 731 or df.rentals.lt(0).any():
        raise ValueError("Expected 731 nonnegative daily targets")
    return {"data": df, "steps": [{"node": "database_quality", "rows": len(df)}]}


def data_scientist(state):
    train, validation, test = split(state["data"])
    return {"profile": {"features": FEATURES, "train_rows": len(train),
                        "validation_rows": len(validation), "holdout_rows": len(test),
                        "task": "regression", "agent_mode": "deterministic policies; no LLM"},
            "steps": [{"node": "data_scientist", "note": "Temporal split; leakage columns excluded"}]}


def hypothesis_generator(state):
    # Memory excludes already-tried families; context explains the next trial.
    tried = {entry["family"] for entry in state.get("memory", [])}
    family = next((candidate for candidate in ["ridge", "forest", "boosting"] if candidate not in tried), "ridge")
    return {"family": family, "cycle": state.get("cycle", 0) + 1,
            "steps": [{"node": "hypothesis_generator", "family": family,
                       "context": state.get("brief", "Start with a regularized linear model")}]}


def ml_engineer(state):
    score = validate(state["data"], state["family"])
    patch = {"validation": score, "steps": [{"node": "ml_engineer", "family": state["family"], "validation": score}]}
    if score["mae"] < state.get("best_mae", float("inf")):
        patch.update(best_mae=score["mae"], best_family=state["family"])
    return patch


def judge(state):
    verdict = "accept" if state["best_mae"] <= state["target_mae"] else "revise"
    if verdict == "revise" and state["cycle"] >= state["max_cycles"]:
        verdict = "budget_exhausted"
    return {"verdict": verdict, "steps": [{"node": "judge", "verdict": verdict, "best_validation_mae": state["best_mae"]}]}


def memory_writer(state):
    entry = {"cycle": state["cycle"], "family": state["family"], "validation": state["validation"]}
    return {"memory": [entry], "steps": [{"node": "memory_writer", "note": "Recorded validation evidence"}]}


def context_builder(state):
    return {"brief": f"Best validation MAE is {state['best_mae']:.1f}; try the next untested nonlinear family.",
            "steps": [{"node": "context_builder", "note": "Rebuilt experiment context from memory"}]}


def holdout_eval(state):
    score, _, _ = holdout(state["data"], state["best_family"])
    baseline, _, _ = holdout(state["data"], "mean")
    return {"holdout_metrics": score, "baseline_metrics": baseline,
            "steps": [{"node": "holdout_eval", "note": "Final selected model scored once on sealed holdout", "metrics": score}]}


def route(state):
    return "memory_writer" if state["verdict"] == "revise" else "holdout_eval"


def build_graph():
    graph = StateGraph(State)
    for name, function in [("database_quality", database_quality), ("data_scientist", data_scientist),
                           ("hypothesis_generator", hypothesis_generator), ("ml_engineer", ml_engineer),
                           ("judge", judge), ("memory_writer", memory_writer),
                           ("context_builder", context_builder), ("holdout_eval", holdout_eval)]:
        graph.add_node(name, function)
    graph.add_edge(START, "database_quality")
    graph.add_edge("database_quality", "data_scientist")
    graph.add_edge("data_scientist", "hypothesis_generator")
    graph.add_edge("hypothesis_generator", "ml_engineer")
    graph.add_edge("ml_engineer", "judge")
    graph.add_conditional_edges("judge", route, {"memory_writer": "memory_writer", "holdout_eval": "holdout_eval"})
    graph.add_edge("memory_writer", "context_builder")
    graph.add_edge("context_builder", "hypothesis_generator")
    graph.add_edge("holdout_eval", END)
    return graph.compile()


def initial(max_cycles=3, target_mae=700.0):
    if not 1 <= max_cycles <= 3 or target_mae <= 0:
        raise ValueError("Use 1–3 cycles and positive target MAE")
    return {"cycle": 0, "max_cycles": max_cycles, "target_mae": target_mae,
            "best_mae": float("inf"), "steps": [], "memory": []}


def save(final):
    ARTIFACTS.mkdir(exist_ok=True)
    report = {key: value for key, value in final.items() if key != "data"}
    (ARTIFACTS / "agent_metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    (ARTIFACTS / "agent_graph.mmd").write_text(build_graph().get_graph().draw_mermaid())
    return report


if __name__ == "__main__":
    graph = build_graph()
    final = graph.invoke(initial(), config={"recursion_limit": 40})
    print(json.dumps(save(final), indent=2))

