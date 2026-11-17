"""Turn the supplied relational F1 CSV files into searchable text records."""

from pathlib import Path
import argparse
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def clean(value: object, fallback: str = "not recorded") -> str:
    if pd.isna(value) or str(value) == "\\N":
        return fallback
    return str(value)


def ordinal(value: object) -> str:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return clean(value)
    suffix = "th" if 10 < number % 100 < 14 else {1: "st", 2: "nd", 3: "rd"}.get(number % 10, "th")
    return f"{number}{suffix}"


def main(source_dir: Path, output_path: Path) -> None:
    races = pd.read_csv(source_dir / "races.csv", na_values="\\N")
    results = pd.read_csv(source_dir / "results.csv", na_values="\\N")
    drivers = pd.read_csv(source_dir / "drivers.csv", na_values="\\N")
    constructors = pd.read_csv(source_dir / "constructors.csv", na_values="\\N")
    circuits = pd.read_csv(source_dir / "circuits.csv", na_values="\\N")
    statuses = pd.read_csv(source_dir / "status.csv", na_values="\\N")

    drivers["driver"] = drivers["forename"].fillna("") + " " + drivers["surname"].fillna("")
    records = (
        results.merge(races[["raceId", "year", "round", "circuitId", "name", "date", "url"]], on="raceId")
        .merge(drivers[["driverId", "driver", "nationality"]], on="driverId")
        .merge(constructors[["constructorId", "name"]].rename(columns={"name": "constructor"}), on="constructorId")
        .merge(circuits[["circuitId", "name", "location", "country"]].rename(columns={"name": "circuit"}), on="circuitId")
        .merge(statuses, on="statusId", how="left")
    )
    records = records.rename(columns={"name": "race_name", "status": "race_status"})
    records["year"] = records["year"].astype(int)

    performance_rows = []
    for row in records.itertuples(index=False):
        finish = ordinal(row.position) if pd.notna(row.position) else clean(row.positionText)
        start = ordinal(row.grid) if pd.notna(row.grid) and int(row.grid) > 0 else "the pit lane or an unrecorded grid position"
        fastest = clean(row.fastestLapTime, "no fastest-lap time recorded")
        text = (
            f"{row.year} {row.race_name} at {row.circuit} in {row.location}, {row.country}. "
            f"{row.driver} of {row.constructor} started from {start} and finished {finish}. "
            f"They completed {clean(row.laps, 'an unrecorded number of')} laps, scored {clean(row.points, '0')} points, "
            f"and their race status was {clean(row.race_status)}. "
            f"Fastest lap: {fastest}."
        )
        performance_rows.append(
            {
                "id": f"result-{row.resultId}", "record_type": "driver_race_result", "document": text,
                "year": row.year, "round": row.round, "race_name": row.race_name, "circuit": row.circuit,
                "country": row.country, "date": row.date, "driver": row.driver, "constructor": row.constructor,
                "finish_position": clean(row.positionText), "grid": clean(row.grid), "points": float(row.points),
                "laps": clean(row.laps), "status": clean(row.race_status), "url": row.url,
            }
        )

    race_rows = []
    for race_id, group in records.groupby("raceId", sort=False):
        first = group.iloc[0]
        winner = group.loc[group["positionOrder"] == 1].head(1)
        winner_text = "Winner not recorded"
        if not winner.empty:
            w = winner.iloc[0]
            winner_text = f"Winner: {w['driver']} ({w['constructor']}) from grid {clean(w['grid'])}."
        finishers = group[group["position"].notna()].shape[0]
        text = (
            f"Race summary: {first['year']} {first['race_name']} at {first['circuit']} in "
            f"{first['location']}, {first['country']}. {winner_text} "
            f"The round had {len(group)} classified entries and {finishers} recorded finishers."
        )
        race_rows.append(
            {
                "id": f"race-{race_id}", "record_type": "race_summary", "document": text,
                "year": int(first.year), "round": first.round, "race_name": first.race_name,
                "circuit": first.circuit, "country": first.country, "date": first.date,
                "driver": "", "constructor": "", "finish_position": "", "grid": "", "points": 0.0,
                "laps": "", "status": "", "url": first.url,
            }
        )

    output = pd.DataFrame(performance_rows + race_rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    print(f"Wrote {len(output):,} searchable records to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "f1_semantic_documents.csv")
    args = parser.parse_args()
    main(args.source, args.output)
