# 🏁 F1 Insight Hub

**Explore Formula 1 history through plain-language search and performance data.**

F1 Insight Hub turns a relational archive of Formula 1 races into an interactive research tool for fans. Instead of requiring users to understand a database schema, it supports questions such as *“Which Ferrari drivers gained positions at Monaco?”* and pairs the answers with a filtered performance dashboard.

> Built as a semantic-search application for AIPI 510: Data Sourcing for Analytics.

## Live Demo

🚀 **[Explore F1 Insight Hub](https://cheney0919-semantic-search-engine-app-rlmon4.streamlit.app/)**

## Why this project?

Historical race data is usually distributed across tables for races, drivers, constructors, circuits, results, and status codes. This project joins those tables into readable race narratives, embeds each narrative with `all-MiniLM-L6-v2`, and stores the vectors in ChromaDB for similarity search.

The result is a fan-facing exploration product with a realistic analytical use case:

- **Discover stories** — Find races, drivers, and performance patterns using natural language.
- **Compare performance** — Filter by era, driver, and constructor, then examine points and race-entry trends.
- **Make historical data approachable** — Translate relational data into understandable, traceable F1 records.

## Features

- Semantic retrieval over race summaries and driver-level race results
- Natural-language queries such as `pole-to-win races`, `Hamilton at Silverstone`, or `drivers who scored points after starting deep in the field`
- Season, driver, and constructor filters
- Performance KPIs: race entries, wins, points, average classified finish, and unclassified entries
- Interactive constructor points trend and driver leaderboard
- Local, persistent ChromaDB vector index

## Product scope

The app is an **historical performance explorer**, not a live motorsport analytics service. The supplied data does not include live timing, weather, social engagement, broadcast exposure, sponsorship amounts, or fan demographics. It should not be used to estimate sponsorship ROI.

## Architecture

```text
F1 relational CSV files
        │
        ▼
scripts/prepare_data.py
  joins + transforms tables into readable race records
        │
        ▼
SentenceTransformers embeddings ──► ChromaDB persistent vector index
        │                                      │
        └──────────────► Streamlit UI ◄────────┘
                    semantic search + dashboard
```

## Data

The project uses the provided F1 CSV archive. It includes race, result, circuit, driver, constructor, standing, qualifying, lap-time, pit-stop, practice-session, sprint, season, and status tables, spanning 1950–2026 according to the `races.csv` file.

The first version uses these tables to create its primary corpus:

- `races.csv`
- `results.csv`
- `drivers.csv`
- `constructors.csv`
- `circuits.csv`
- `status.csv`

`lap_times.csv`, `pit_stops.csv`, and qualifying data are intentionally reserved for a future advanced analysis module. This keeps the first semantic index focused, responsive, and interpretable.

## Run locally

### 1. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Prepare the search corpus

Place the CSV files from the F1 archive inside `data/raw/`, then run:

```bash
python scripts/prepare_data.py
```

This creates `data/f1_semantic_documents.csv`, a searchable collection of race summaries and driver race-result narratives.

### 3. Build the persistent vector index

```bash
python scripts/build_index.py
```

This embeds the records and saves the ChromaDB index in `.chroma/`.

### 4. Launch the app

```bash
streamlit run app.py
```

The app reuses `.chroma/` on later launches. It can also build the index automatically on the first search if it does not exist, or rebuild it when you select **Rebuild local search index**.

## Deploy to Hugging Face Spaces

1. Create a new **Streamlit** Space.
2. Push this repository to the Space, or connect the GitHub repository.
3. Ensure `data/f1_semantic_documents.csv` is available in the repository or generate it during the Space build.
4. Hugging Face installs `requirements.txt` automatically and runs `app.py`.

## Repository structure

```text
.
├── app.py                         # Streamlit application
├── data/
│   └── f1_semantic_documents.csv  # Generated semantic-search corpus
├── scripts/
│   ├── build_index.py              # Persistent ChromaDB builder
│   └── prepare_data.py             # Relational-data transformation pipeline
├── .streamlit/config.toml          # App theme
├── requirements.txt
└── README.md
```

## Future improvements

- Add pit-stop and lap-time analysis for strategy exploration.
- Add a circuit map and country-level race-history view.
- Add retrieval evaluation with a labeled set of fan questions.
- Add live, licensed data sources for current-season analytics.

## Tech stack

`Python` · `Pandas` · `SentenceTransformers` · `ChromaDB` · `Streamlit` · `Plotly`

