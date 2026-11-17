"""Create the persistent ChromaDB index used by the Streamlit app."""

from pathlib import Path
import shutil
import chromadb
import pandas as pd
from sentence_transformers import SentenceTransformer


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "f1_semantic_documents.csv"
CHROMA_PATH = ROOT / ".chroma"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def main() -> None:
    data = pd.read_csv(DATA_PATH, keep_default_na=False)
    if CHROMA_PATH.exists():
        shutil.rmtree(CHROMA_PATH)
    model = SentenceTransformer(MODEL_NAME)
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    collection = client.create_collection("f1_race_insights", metadata={"hnsw:space": "cosine"})

    batch_size = 256
    for start in range(0, len(data), batch_size):
        batch = data.iloc[start : start + batch_size]
        collection.add(
            ids=batch["id"].tolist(),
            documents=batch["document"].tolist(),
            embeddings=model.encode(batch["document"].tolist(), show_progress_bar=False).tolist(),
            metadatas=[
                {
                    "record_type": row.record_type,
                    "year": int(row.year),
                    "driver": str(row.driver),
                    "constructor": str(row.constructor),
                    "country": str(row.country),
                }
                for row in batch.itertuples(index=False)
            ],
        )
        print(f"Indexed {min(start + batch_size, len(data)):,}/{len(data):,} records")
    print(f"Saved ChromaDB index with {collection.count():,} records to {CHROMA_PATH}")


if __name__ == "__main__":
    main()
