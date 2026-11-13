"""Live graph is generated from the compiled workflow, never a separate mock."""
import graphviz
import streamlit as st

from agents.graph import build_graph, initial, save

st.set_page_config(page_title="Bike Weather Research Loop", layout="wide")
st.title("Bike Weather · Research Loop")
st.caption("LangGraph · deterministic agent policies · temporal model validation · PostgreSQL in Docker")
cycles = st.sidebar.slider("Maximum experiment cycles", 1, 3, 3)
target = st.sidebar.number_input("Validation MAE acceptance threshold (rentals/day)", min_value=1., value=700.)
st.sidebar.info("The holdout is scored after model selection. Adjust settings using validation results only.")
graph = build_graph()
left, right = st.columns([1, 1])
canvas = left.empty()
events = right.empty()


def draw(active=None, completed=()):
    dot = graphviz.Digraph()
    dot.attr(rankdir="TB")
    compiled = graph.get_graph()
    for name in compiled.nodes:
        color = "#fde68a" if name == active else "#bbf7d0" if name in completed else "#e2e8f0"
        dot.node(name, name.replace("_", " "), style="filled", fillcolor=color)
    for edge in compiled.edges:
        dot.edge(edge.source, edge.target, style="dashed" if edge.conditional else "solid")
    canvas.graphviz_chart(dot)


draw()
if st.button("Run research loop", type="primary"):
    final = initial(cycles, target)
    completed = []
    try:
        # Values stream contains accumulated state, including the actual node trace.
        for state in graph.stream(final, stream_mode="values", config={"recursion_limit": 40}):
            final = state
            if state.get("steps"):
                active = state["steps"][-1]["node"]
                completed.append(active)
                draw(active, completed)
                events.dataframe(state["steps"], use_container_width=True)
        st.session_state["report"] = save(final)
        draw(completed=completed)
    except Exception as error:
        st.error(f"Workflow failed: {error}")

if "report" in st.session_state:
    result = st.session_state["report"]
    st.success(f"Finished {result['cycle']} cycle(s) · selected {result['best_family']} · {result['verdict']}")
    a, b, c = st.columns(3)
    a.metric("Holdout MAE (rentals/day)", f"{result['holdout_metrics']['mae']:.1f}")
    b.metric("Holdout RMSE", f"{result['holdout_metrics']['rmse']:.1f}")
    c.metric("Holdout R²", f"{result['holdout_metrics']['r2']:.3f}")
    st.json(result)

