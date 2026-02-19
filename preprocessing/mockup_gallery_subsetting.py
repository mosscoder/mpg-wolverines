"""
Mockup: explore gallery subsetting strategy for test evaluation.

For filtered configs (threshold=0.25, 0.5), the gallery is manageable.
For unfiltered (threshold=0.0), the full training set is too large.

Strategy: for each event, count how many images pass the quality filter,
then randomly sample that same count from the full (unfiltered) event.
This guarantees exact image count match and preserves event-level distribution.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from utils.reid_data import (
    load_feasibility_config, load_reidentification_dataset,
    build_metadata_cache, filter_training_pool_by_quality,
    get_event_index_map, subsample_to_match_filtered,
)


def main():
    print("Loading dataset and config...")
    dataset = load_reidentification_dataset()
    metadata_cache = build_metadata_cache(dataset)
    config = load_feasibility_config()

    if not config:
        sys.exit(1)

    feasible_individuals = config.get('qualified_individuals', [])
    id_to_indices = metadata_cache['id_to_indices']

    thresholds = [0.0, 0.25, 0.5]

    print(f"\n{'='*90}")
    print(f"Gallery subsetting analysis for {len(feasible_individuals)} individuals")
    print(f"{'='*90}")

    for ind_id in sorted(feasible_individuals):
        all_indices = list(id_to_indices.get(ind_id, []))
        all_events = get_event_index_map(metadata_cache, all_indices)

        print(f"\n{ind_id}:")
        print(f"  Total: {len(all_indices)} images, {len(all_events)} events")

        filtered_by_thresh = {}
        for thresh in thresholds:
            if thresh == 0.0:
                filtered = all_indices
            else:
                filtered = filter_training_pool_by_quality(
                    metadata_cache, all_indices, thresh
                )
            filtered_events = get_event_index_map(metadata_cache, filtered)
            filtered_by_thresh[thresh] = filtered
            print(f"  threshold>={thresh}: {len(filtered):4d} images, "
                  f"{len(filtered_events):3d} events")

        for target_thresh in [0.25, 0.5]:
            subsampled = subsample_to_match_filtered(
                metadata_cache, all_indices, filtered_by_thresh[target_thresh]
            )
            sub_events = get_event_index_map(metadata_cache, subsampled)
            print(f"  unfiltered subsampled to match >={target_thresh}: "
                  f"{len(subsampled):4d} images, {len(sub_events):3d} events")

    # Summary
    print(f"\n{'='*90}")
    print(f"Summary across all individuals")
    print(f"{'='*90}")
    print(f"{'Condition':<20} {'Total Images':>14} {'Total Events':>14} "
          f"{'Mean Img/Ind':>14} {'Mean Evt/Ind':>14}")
    print(f"{'-'*76}")

    n_ind = len(feasible_individuals)

    for thresh in thresholds:
        total_images = 0
        total_events = 0
        for ind_id in feasible_individuals:
            indices = list(id_to_indices.get(ind_id, []))
            if thresh == 0.0:
                filtered = indices
            else:
                filtered = filter_training_pool_by_quality(
                    metadata_cache, indices, thresh
                )
            total_images += len(filtered)
            total_events += len(get_event_index_map(metadata_cache, filtered))

        print(f">={thresh:<18} {total_images:>14d} {total_events:>14d} "
              f"{total_images/n_ind:>14.1f} {total_events/n_ind:>14.1f}")

    for target_thresh in [0.25, 0.5]:
        total_images = 0
        total_events = 0
        for ind_id in feasible_individuals:
            indices = list(id_to_indices.get(ind_id, []))
            filtered = filter_training_pool_by_quality(
                metadata_cache, indices, target_thresh
            )
            subsampled = subsample_to_match_filtered(
                metadata_cache, indices, filtered
            )
            total_images += len(subsampled)
            total_events += len(get_event_index_map(metadata_cache, subsampled))

        print(f"unfilt→{target_thresh:<13} {total_images:>14d} {total_events:>14d} "
              f"{total_images/n_ind:>14.1f} {total_events/n_ind:>14.1f}")


if __name__ == "__main__":
    main()
