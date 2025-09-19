"""
Feature extraction and data handling for VampRust Python bindings.
"""

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional, Union

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd


@dataclass
class Feature:
    """Represents a single feature extracted by a Vamp plugin."""

    values: List[float]
    has_timestamp: bool = False
    sec: int = 0
    nsec: int = 0
    label: Optional[str] = None

    @property
    def timestamp(self) -> float:
        """Get timestamp as floating point seconds."""
        return self.sec + self.nsec * 1e-9

    def to_dict(self) -> Dict[str, Any]:
        """Convert feature to dictionary."""
        result = {
            "values": self.values,
            "has_timestamp": self.has_timestamp,
        }

        if self.has_timestamp:
            result["timestamp"] = self.timestamp
            result["sec"] = self.sec
            result["nsec"] = self.nsec

        if self.label:
            result["label"] = self.label

        return result


class FeatureSet:
    """Container for multiple features with analysis utilities."""

    def __init__(self, features: List[Union[Feature]]):
        self.features: List[Feature] = []

        for f in features:
            if isinstance(f, Feature):
                self.features.append(f)
            elif isinstance(f, dict):
                # Convert dict to Feature
                feature = Feature(
                    values=f.get("values", []),
                    has_timestamp=f.get("has_timestamp", False),
                    sec=f.get("sec", 0),
                    nsec=f.get("nsec", 0),
                    label=f.get("label", None),
                )
                self.features.append(feature)

    def __len__(self) -> int:
        return len(self.features)

    def __iter__(self) -> Iterator[Feature]:
        return iter(self.features)

    def __getitem__(self, index: int) -> Feature:
        return self.features[index]

    @property
    def timestamps(self) -> "np.ndarray":
        """Get array of timestamps for all features."""
        return np.array([f.timestamp for f in self.features if f.has_timestamp])

    @property
    def values_matrix(self) -> "np.ndarray":
        """Get matrix of all feature values (features x dimensions)."""
        if not self.features:
            return np.array([])

        max_dims = max(len(f.values) for f in self.features)
        matrix = np.zeros((len(self.features), max_dims))

        for i, feature in enumerate(self.features):
            matrix[i, : len(feature.values)] = feature.values

        return matrix

    def filter_by_timestamp(self, start_time: float, end_time: float) -> "FeatureSet":
        """Filter features by timestamp range."""
        filtered = [
            f
            for f in self.features
            if f.has_timestamp and start_time <= f.timestamp <= end_time
        ]
        return FeatureSet(filtered)

    def filter_by_label(self, label: str) -> "FeatureSet":
        """Filter features by label."""
        filtered = [f for f in self.features if f.label == label]
        return FeatureSet(filtered)

    def group_by_label(self) -> Dict[str, "FeatureSet"]:
        """Group features by their labels."""
        groups = defaultdict(list)

        for feature in self.features:
            key = feature.label if feature.label else "unlabeled"
            groups[key].append(feature)

        return {label: FeatureSet(features) for label, features in groups.items()}

    def to_numpy(self) -> "np.ndarray":
        """Convert to numpy array (features x dimensions)."""
        return self.values_matrix

    def to_dict_list(self) -> List[Dict[str, Any]]:
        """Convert to list of dictionaries."""
        return [f.to_dict() for f in self.features]

    def to_pandas(self) -> "pd.DataFrame":
        """Convert to pandas DataFrame (requires pandas)."""

        import pandas as pd

        data = []
        for i, feature in enumerate(self.features):
            row: Dict[str, Any] = {"feature_index": i}
            print(feature)

            if feature.has_timestamp:
                row["timestamp"] = feature.timestamp
                row["sec"] = feature.sec
                row["nsec"] = feature.nsec

            if feature.label:
                row["label"] = feature.label

            # Add feature values as separate columns
            for j, value in enumerate(feature.values):
                row[f"value_{j}"] = value

            data.append(row)

        return pd.DataFrame(data)

    def summary(self) -> Dict[str, Any]:
        """Get summary statistics of the feature set."""
        if not self.features:
            return {"count": 0}

        values_matrix = self.values_matrix

        summary = {
            "count": len(self.features),
            "dimensions": values_matrix.shape[1],
            "has_timestamps": any(f.has_timestamp for f in self.features),
            "labels": list(set(f.label for f in self.features if f.label)),
        }

        if values_matrix.size > 0:
            summary.update(
                {
                    "mean_values": values_matrix.mean(axis=0).tolist(),
                    "std_values": values_matrix.std(axis=0).tolist(),
                    "min_values": values_matrix.min(axis=0).tolist(),
                    "max_values": values_matrix.max(axis=0).tolist(),
                }
            )

        if summary["has_timestamps"]:
            timestamps = self.timestamps
            if len(timestamps) > 0:
                summary.update(
                    {
                        "duration": float(timestamps.max() - timestamps.min()),
                        "start_time": float(timestamps.min()),
                        "end_time": float(timestamps.max()),
                        "time_resolution": float(np.mean(np.diff(timestamps)))
                        if len(timestamps) > 1
                        else 0.0,
                    }
                )

        return summary

    def __repr__(self) -> str:
        summary = self.summary()
        return f"FeatureSet(count={summary['count']}, dimensions={summary.get('dimensions', 0)})"
