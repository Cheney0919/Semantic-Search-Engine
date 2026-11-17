"""F1 Insight Hub: semantic discovery and lightweight historical analysis."""

from pathlib import Path
import shutil

import chromadb
import pandas as pd
import plotly.express as px
import streamlit as st
from sentence_transformers import SentenceTransformer


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "f1_semantic_documents.csv"
CHROMA_PATH = ROOT / ".chroma"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH, keep_default_na=False)


@st.cache_resource(show_spinner="Loading the semantic search model…")
def load_model() -> SentenceTransformer:
    return SentenceTransformer(MODEL_NAME)


@st.cache_resource(show_spinner="Preparing the F1 semantic index…")
def load_collection(record_count: int):
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    collection = client.get_or_create_collection(name="f1_race_insights", metadata={"hnsw:space": "cosine"})
    if collection.count() != record_count:
        if collection.count():
            client.delete_collection("f1_race_insights")
            collection = client.create_collection(name="f1_race_insights", metadata={"hnsw:space": "cosine"})
        data = load_data()
        model = load_model()
        batch_size = 256
        for start in range(0, len(data), batch_size):
            batch = data.iloc[start : start + batch_size]
            embeddings = model.encode(batch["document"].tolist(), show_progress_bar=False).tolist()
            metadata = [
                {
                    "record_type": row.record_type,
                    "year": int(row.year),
                    "driver": str(row.driver),
                    "constructor": str(row.constructor),
                    "country": str(row.country),
                }
                for row in batch.itertuples(index=False)
            ]
            collection.add(ids=batch["id"].tolist(), documents=batch["document"].tolist(), embeddings=embeddings, metadatas=metadata)
    return collection


def reset_index() -> None:
    if CHROMA_PATH.exists():
        shutil.rmtree(CHROMA_PATH)
    load_collection.clear()


def make_where(year_range, driver, constructor):
    filters = [{"year": {"$gte": int(year_range[0])}}, {"year": {"$lte": int(year_range[1])}}]
    if driver != "All drivers":
        filters.append({"driver": driver})
    if constructor != "All constructors":
        filters.append({"constructor": constructor})
    return {"$and": filters}


def ordinal(value: str) -> str:
    try:
        number = int(float(value))
    except (TypeError, ValueError):
        return value or "Not classified"
    suffix = "th" if 10 < number % 100 < 14 else {1: "st", 2: "nd", 3: "rd"}.get(number % 10, "th")
    return f"{number}{suffix}"


def main() -> None:
    st.set_page_config(page_title="F1 Insight Hub", page_icon="🏁", layout="wide")
    st.title("🏁 F1 Insight Hub")
    st.caption("Explore Formula 1 history through semantic search and performance data.")

    if not DATA_PATH.exists():
        st.error("Search data is missing. Run `python scripts/prepare_data.py` after placing the supplied CSV files in `data/raw/`.")
        st.stop()

    data = load_data()
    performance = data[data["record_type"] == "driver_race_result"].copy()
    years = sorted(data["year"].astype(int).unique())

    with st.sidebar:
        st.header("Explore")
        year_range = st.slider("Season range", int(min(years)), int(max(years)), (max(int(min(years)), 2000), int(max(years))))
        drivers = ["All drivers"] + sorted(driver for driver in performance["driver"].unique() if driver)
        constructors = ["All constructors"] + sorted(team for team in performance["constructor"].unique() if team)
        driver = st.selectbox("Driver", drivers)
        constructor = st.selectbox("Constructor", constructors)
        if st.button("Rebuild local search index", help="Use after replacing the prepared CSV."):
            reset_index()
            st.rerun()

    filtered = performance[performance["year"].between(*year_range)]
    if driver != "All drivers":
        filtered = filtered[filtered["driver"] == driver]
    if constructor != "All constructors":
        filtered = filtered[filtered["constructor"] == constructor]

    st.subheader("Ask F1 history in plain language")
    query = st.text_input(
        "Try a question",
        placeholder="e.g., Which Ferrari drivers gained positions at Monaco?",
        label_visibility="collapsed",
    )
    st.caption("Try: “pole-to-win races”, “Hamilton at Silverstone”, or “drivers who scored points after starting deep in the field.”")

    if query:
        collection = load_collection(len(data))
        model = load_model()
        result = collection.query(
            query_embeddings=model.encode([query]).tolist(),
            n_results=8,
            where=make_where(year_range, driver, constructor),
            include=["documents", "metadatas", "distances"],
        )
        documents = result.get("documents", [[]])[0]
        distances = result.get("distances", [[]])[0]
        if not documents:
            st.info("No matching records for these filters. Broaden the filters and try again.")
        else:
            st.subheader("Most relevant historical records")
            for i, (document, distance) in enumerate(zip(documents, distances), start=1):
                relevance = max(0, round((1 - float(distance)) * 100))
                with st.expander(f"{i}. Match score: {relevance}%", expanded=i == 1):
                    st.write(document)

    st.divider()
    st.subheader("Performance snapshot")
    wins = int((filtered["finish_position"] == "1").sum())
    points = filtered["points"].sum()
    classified = filtered[filtered["finish_position"].str.isnumeric()].copy()
    average_finish = pd.to_numeric(classified["finish_position"], errors="coerce").mean()
    dnfs = int((~filtered["finish_position"].str.isnumeric()).sum())
    metrics = st.columns(4)
    metrics[0].metric("Race entries", f"{len(filtered):,}")
    metrics[1].metric("Wins", f"{wins:,}")
    metrics[2].metric("Points scored", f"{points:,.0f}")
    metrics[3].metric("Average classified finish", f"{average_finish:.1f}" if pd.notna(average_finish) else "—")
    st.caption(f"DNF / unclassified entries in the selection: {dnfs:,}")

    left, right = st.columns((1.25, 1))
    with left:
        chart_data = filtered.groupby(["year", "constructor"], as_index=False)["points"].sum()
        top_teams = chart_data.groupby("constructor")["points"].sum().nlargest(6).index
        chart_data = chart_data[chart_data["constructor"].isin(top_teams)]
        st.plotly_chart(
            px.line(chart_data, x="year", y="points", color="constructor", markers=True, title="Points trend by constructor"),
            use_container_width=True,
        )
    with right:
        drivers_table = (
            filtered.groupby("driver", as_index=False)
            .agg(entries=("id", "count"), points=("points", "sum"), wins=("finish_position", lambda values: (values == "1").sum()))
            .sort_values(["points", "wins"], ascending=False)
            .head(10)
        )
        st.markdown("#### Top drivers in this view")
        st.dataframe(drivers_table, hide_index=True, use_container_width=True)

    st.divider()
    st.caption("Data is historical and may be incomplete. This project provides performance exploration, not live timing, weather, audience, sponsorship-exposure, or ROI analytics.")


if __name__ == "__main__":
    main()
