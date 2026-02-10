#!/usr/bin/env python3
"""
09_assess_quality_capture_rates.py

Analyze how many high-quality images (pelage_probability >= 0.5) are accumulated
per individual per camera trap event and per season.

Key Definitions:
- Camera trap event: Unique ymdh timestamp per individual
- Season: Year-specific, September to August (e.g., "2021_2022" = Sept 2021 through Aug 2022)
- High quality: pelage_probability >= 0.5
"""

import json
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path


QUALITY_THRESHOLD = 0.5


def get_season(ymdh: int) -> str:
    """
    Assign season year label.

    Sept 2021 - Aug 2022 → '2021_2022'

    Args:
        ymdh: Timestamp in YYYYMMDDHHMM format

    Returns:
        Season string like '2021_2022'
    """
    month = (ymdh // 1000000) % 100
    year = ymdh // 100000000

    if month >= 9:  # Sept-Dec
        return f"{year}_{year + 1}"
    else:  # Jan-Aug
        return f"{year - 1}_{year}"


def main():
    # Paths
    script_dir = Path(__file__).parent
    data_dir = script_dir.parent / "data"
    input_file = data_dir / "inference" / "pelage_inference_results.csv"
    output_file = data_dir / "quality_capture_rates.json"

    # Load data
    print(f"Loading data from {input_file}")
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} records")

    # Add season column
    df["season"] = df["ymdh"].apply(get_season)

    # Flag high quality images
    df["is_high_quality"] = df["pelage_probability"] >= QUALITY_THRESHOLD

    # Group by (id, ymdh) to get per-event statistics
    event_stats = df.groupby(["id", "ymdh", "season"]).agg(
        n_images=("crop_filename", "count"),
        high_quality_count=("is_high_quality", "sum")
    ).reset_index()

    # Core metric: high-quality images per event
    hq_per_event = event_stats["high_quality_count"]
    median_hq_per_event = float(np.median(hq_per_event))
    mean_hq_per_event = float(np.mean(hq_per_event))
    q25_hq_per_event = float(np.percentile(hq_per_event, 25))
    q75_hq_per_event = float(np.percentile(hq_per_event, 75))

    # Build output structure
    result = {
        "key_finding": {
            "median_high_quality_per_event": median_hq_per_event,
            "mean_high_quality_per_event": round(mean_hq_per_event, 1),
            "q25_high_quality_per_event": q25_hq_per_event,
            "q75_high_quality_per_event": q75_hq_per_event,
            "description": f"Across the study, the median count of high-quality images was {median_hq_per_event:.0f} per camera trap event (IQR: {q25_hq_per_event:.0f}-{q75_hq_per_event:.0f})."
        },
        "metadata": {
            "quality_threshold": QUALITY_THRESHOLD,
            "season_months": "all months included",
            "generated_at": datetime.now().isoformat(),
            "source_file": "pelage_inference_results.csv"
        },
        "summary": {
            "total_individuals": event_stats["id"].nunique(),
            "total_events": len(event_stats),
            "total_images": int(event_stats["n_images"].sum()),
            "total_high_quality": int(event_stats["high_quality_count"].sum())
        },
        "by_season": {},
        "by_individual": {}
    }

    # Process by season (aggregate across all individuals)
    for season in sorted(event_stats["season"].unique()):
        season_data = event_stats[event_stats["season"] == season]
        season_hq = season_data["high_quality_count"]
        # Get unique stations from the filtered dataframe
        n_stations = df[df["season"] == season]["station"].nunique()
        result["by_season"][season] = {
            "n_stations": n_stations,
            "n_individuals": season_data["id"].nunique(),
            "n_events": len(season_data),
            "n_images": int(season_data["n_images"].sum()),
            "high_quality_count": int(season_hq.sum()),
            "median_hq_per_event": float(np.median(season_hq))
        }

    # Process by individual (simplified - no event-level detail)
    for individual_id in sorted(event_stats["id"].unique()):
        ind_events = event_stats[event_stats["id"] == individual_id]
        ind_hq = ind_events["high_quality_count"]

        result["by_individual"][individual_id] = {
            "total_events": len(ind_events),
            "total_images": int(ind_events["n_images"].sum()),
            "high_quality_count": int(ind_hq.sum()),
            "median_hq_per_event": float(np.median(ind_hq))
        }

    # Save output
    print(f"\nSaving results to {output_file}")
    with open(output_file, "w") as f:
        json.dump(result, f, indent=2)

    # Print key finding prominently
    print("\n" + "=" * 60)
    print("KEY FINDING")
    print("=" * 60)
    print(result["key_finding"]["description"])
    print("=" * 60)

    # Print summary
    print(f"\nTotal: {result['summary']['total_individuals']} individuals, "
          f"{result['summary']['total_events']} events, "
          f"{result['summary']['total_images']} images, "
          f"{result['summary']['total_high_quality']} high quality")

    print("\nBy Season:")
    for season, stats in result["by_season"].items():
        print(f"  {season}: {stats['n_events']} events, median {stats['median_hq_per_event']:.0f} HQ/event")

    print("\nDone!")


if __name__ == "__main__":
    main()
