"""
Standardized result loading utilities for wolverines experiments.
Provides generic JSON/CSV loaders and ResultsCollection class for querying results.
"""

import os
import json
import glob
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from collections import defaultdict


def load_json_results(results_dir: str, pattern: str = "*.json") -> List[Dict[str, Any]]:
    """
    Load all JSON result files from a directory matching a pattern.

    Args:
        results_dir: Directory containing result files
        pattern: Glob pattern for files (default: "*.json")

    Returns:
        List of dictionaries loaded from JSON files
    """
    results = []
    results_path = Path(results_dir)

    if not results_path.exists():
        print(f"Results directory not found: {results_dir}")
        return results

    json_files = list(results_path.glob(pattern))

    if not json_files:
        print(f"No files matching '{pattern}' found in {results_dir}")
        return results

    print(f"Found {len(json_files)} result files")

    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                result = json.load(f)
                result['_filename'] = json_file.name
                result['_filepath'] = str(json_file)
                results.append(result)
        except json.JSONDecodeError as e:
            print(f"Warning: Could not load {json_file.name}: {e}")
        except Exception as e:
            print(f"Warning: Error loading {json_file.name}: {e}")

    print(f"Successfully loaded {len(results)} results")
    return results


def load_csv_results(results_dir: str, pattern: str = "*.csv") -> List[Dict[str, Any]]:
    """
    Load all CSV result files from a directory matching a pattern.
    Each CSV is loaded as a dict with 'data' containing the DataFrame and metadata from filename.

    Args:
        results_dir: Directory containing result files
        pattern: Glob pattern for files (default: "*.csv")

    Returns:
        List of dictionaries with 'data' key containing pandas DataFrame
    """
    results = []
    results_path = Path(results_dir)

    if not results_path.exists():
        print(f"Results directory not found: {results_dir}")
        return results

    csv_files = list(results_path.glob(pattern))

    if not csv_files:
        print(f"No files matching '{pattern}' found in {results_dir}")
        return results

    print(f"Found {len(csv_files)} CSV files")

    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            result = {
                'data': df,
                '_filename': csv_file.name,
                '_filepath': str(csv_file)
            }
            results.append(result)
        except Exception as e:
            print(f"Warning: Could not load {csv_file.name}: {e}")

    print(f"Successfully loaded {len(results)} CSV files")
    return results


class ResultsCollection:
    """
    Collection of experiment results with querying and filtering capabilities.
    Supports both JSON and CSV results.
    """

    def __init__(self, results: Optional[List[Dict[str, Any]]] = None):
        """
        Initialize ResultsCollection.

        Args:
            results: Optional list of result dictionaries
        """
        self.results = results or []

    @classmethod
    def from_json_dir(cls, results_dir: str, pattern: str = "*.json") -> 'ResultsCollection':
        """Load results from a directory of JSON files."""
        results = load_json_results(results_dir, pattern)
        return cls(results)

    @classmethod
    def from_csv_dir(cls, results_dir: str, pattern: str = "*.csv") -> 'ResultsCollection':
        """Load results from a directory of CSV files."""
        results = load_csv_results(results_dir, pattern)
        return cls(results)

    def __len__(self) -> int:
        return len(self.results)

    def __iter__(self):
        return iter(self.results)

    def __getitem__(self, idx):
        return self.results[idx]

    def filter(self, **kwargs) -> 'ResultsCollection':
        """
        Filter results by attribute values.

        Args:
            **kwargs: Key-value pairs to filter on. Supports nested keys with dot notation.

        Returns:
            New ResultsCollection with filtered results

        Example:
            collection.filter(alpha=0, samples_per_class=16)
            collection.filter(config__alpha=0)  # Nested access using __
        """
        filtered = []

        for result in self.results:
            match = True
            for key, value in kwargs.items():
                # Support nested key access with double underscore
                keys = key.split('__')
                obj = result

                try:
                    for k in keys:
                        if isinstance(obj, dict):
                            obj = obj.get(k)
                        else:
                            obj = getattr(obj, k, None)
                        if obj is None:
                            match = False
                            break

                    if obj != value:
                        match = False
                except (KeyError, AttributeError, TypeError):
                    match = False

                if not match:
                    break

            if match:
                filtered.append(result)

        return ResultsCollection(filtered)

    def group_by(self, *keys) -> Dict[tuple, 'ResultsCollection']:
        """
        Group results by one or more keys.

        Args:
            *keys: Keys to group by (supports nested keys with __ notation)

        Returns:
            Dict mapping key tuples to ResultsCollection instances

        Example:
            groups = collection.group_by('alpha', 'samples_per_class')
        """
        groups = defaultdict(list)

        for result in self.results:
            key_values = []
            for key in keys:
                # Support nested key access
                key_parts = key.split('__')
                obj = result

                try:
                    for k in key_parts:
                        if isinstance(obj, dict):
                            obj = obj.get(k)
                        else:
                            obj = getattr(obj, k, None)
                    key_values.append(obj)
                except (KeyError, AttributeError, TypeError):
                    key_values.append(None)

            groups[tuple(key_values)].append(result)

        return {k: ResultsCollection(v) for k, v in groups.items()}

    def get_unique(self, key: str) -> List[Any]:
        """
        Get unique values for a key across all results.

        Args:
            key: Key to get unique values for (supports __ notation for nested)

        Returns:
            Sorted list of unique values
        """
        values = set()
        key_parts = key.split('__')

        for result in self.results:
            obj = result
            try:
                for k in key_parts:
                    if isinstance(obj, dict):
                        obj = obj.get(k)
                    else:
                        obj = getattr(obj, k, None)
                if obj is not None:
                    values.add(obj)
            except (KeyError, AttributeError, TypeError):
                continue

        return sorted(values)

    def to_dataframe(self, keys: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Convert results to a pandas DataFrame.

        Args:
            keys: Optional list of keys to include. If None, flattens all top-level keys.

        Returns:
            DataFrame with results
        """
        if not self.results:
            return pd.DataFrame()

        rows = []
        for result in self.results:
            if keys:
                row = {}
                for key in keys:
                    key_parts = key.split('__')
                    obj = result
                    try:
                        for k in key_parts:
                            if isinstance(obj, dict):
                                obj = obj.get(k)
                            else:
                                obj = getattr(obj, k, None)
                        row[key] = obj
                    except (KeyError, AttributeError, TypeError):
                        row[key] = None
                rows.append(row)
            else:
                # Flatten top-level dict keys only
                row = {k: v for k, v in result.items()
                       if not isinstance(v, (dict, list)) or k.startswith('_')}
                rows.append(row)

        return pd.DataFrame(rows)

    def get_metric(self, metric_key: str, aggregation: str = 'mean') -> Optional[float]:
        """
        Get aggregated metric value across all results.

        Args:
            metric_key: Key to the metric (supports __ notation)
            aggregation: How to aggregate ('mean', 'std', 'min', 'max', 'median')

        Returns:
            Aggregated metric value or None if no valid values found
        """
        import numpy as np

        values = []
        key_parts = metric_key.split('__')

        for result in self.results:
            obj = result
            try:
                for k in key_parts:
                    if isinstance(obj, dict):
                        obj = obj.get(k)
                    else:
                        obj = getattr(obj, k, None)
                if obj is not None and isinstance(obj, (int, float)):
                    values.append(obj)
            except (KeyError, AttributeError, TypeError):
                continue

        if not values:
            return None

        if aggregation == 'mean':
            return np.mean(values)
        elif aggregation == 'std':
            return np.std(values, ddof=1) if len(values) > 1 else 0.0
        elif aggregation == 'min':
            return np.min(values)
        elif aggregation == 'max':
            return np.max(values)
        elif aggregation == 'median':
            return np.median(values)
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")

    def get_metrics(self, metric_key: str) -> List[float]:
        """
        Get list of metric values across all results.

        Args:
            metric_key: Key to the metric (supports __ notation)

        Returns:
            List of metric values
        """
        values = []
        key_parts = metric_key.split('__')

        for result in self.results:
            obj = result
            try:
                for k in key_parts:
                    if isinstance(obj, dict):
                        obj = obj.get(k)
                    else:
                        obj = getattr(obj, k, None)
                if obj is not None and isinstance(obj, (int, float)):
                    values.append(obj)
            except (KeyError, AttributeError, TypeError):
                continue

        return values
