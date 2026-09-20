"""Natural inference must use local statistics and cover bounded node groups."""

from pathlib import Path

import numpy as np
import pytest
import torch

from giano.spatiotemporal_dataset import (
    NaturalGapWindowDataset,
    SpatiotemporalPanel,
    SpatiotemporalWindowDataset,
)


def _panel():
    length, nodes = 24, 5
    values = (
        np.arange(length, dtype=np.float32)[:, None]
        + np.arange(nodes, dtype=np.float32)[None, :]
    )
    values[4:6] = np.nan
    return SpatiotemporalPanel(
        variable="temperature",
        times=np.datetime64("2020-01-01T00", "ns")
        + np.arange(length) * np.timedelta64(1, "h"),
        station_ids=tuple(f"S{node}" for node in range(nodes)),
        coordinates=np.array(
            [[46 + node / 10, 11] for node in range(nodes)], dtype=np.float32
        ),
        values=values,
        auxiliary=None,
        coverage_bounds=np.tile([0, length - 1], (nodes, 1)),
        source_paths=tuple(Path(f"S{node}.nc") for node in range(nodes)),
    )


def _windows(panel):
    return SpatiotemporalWindowDataset(
        panel,
        "all",
        seq_len=12,
        stride=12,
        max_nodes=2,
        min_context_points=2,
        mask_mode="natural",
    )


def test_every_target_covered_once_per_window_with_training_sized_groups():
    dataset = NaturalGapWindowDataset(_windows(_panel()))
    assert len(dataset) == 3
    counts = np.zeros(5, dtype=int)
    for sample in dataset:
        assert sample["features"].shape[1] == 2
        assert sample["node_mask"].sum() <= 2
        for local, global_node in enumerate(sample["node_indices"]):
            if sample["evaluation_mask"][:, local].any():
                counts[int(global_node)] += 1
                assert sample["evaluation_mask"][:, local].sum() == 2
    np.testing.assert_array_equal(counts, np.ones(5))


def test_natural_inputs_ignore_values_outside_the_window():
    original, changed = _panel(), _panel()
    changed.values[12:] += 10000
    first = NaturalGapWindowDataset(_windows(original))
    second = NaturalGapWindowDataset(_windows(changed))
    for index in range(len(first)):
        for key in (
            "features",
            "baseline",
            "baseline_physical",
            "center",
            "scale",
            "evaluation_mask",
            "node_indices",
        ):
            assert torch.equal(first[index][key], second[index][key]), key


@pytest.mark.parametrize("max_nodes", [2, 8])
def test_target_mask_matches_numpy_with_context_and_padding(max_nodes):
    windows = SpatiotemporalWindowDataset(
        _panel(),
        "all",
        seq_len=12,
        stride=12,
        max_nodes=max_nodes,
        min_context_points=2,
        mask_mode="natural",
    )
    dataset = NaturalGapWindowDataset(windows)
    for index, (window_index, targets) in enumerate(dataset.groups):
        original = windows._window(window_index, targets)
        expected = original["evaluation_mask"].numpy().copy()
        expected[:, ~np.isin(original["node_indices"].numpy(), targets)] = False
        sample = dataset[index]
        assert sample["evaluation_mask"].dtype == torch.bool
        np.testing.assert_array_equal(sample["evaluation_mask"].numpy(), expected)
        for key in original.keys() - {"evaluation_mask"}:
            assert torch.equal(sample[key], original[key]), key


def test_targets_without_two_local_observations_are_left_to_fallback():
    panel = _panel()
    panel.values[:12, 0] = np.nan
    panel.values[0, 0] = 10
    dataset = NaturalGapWindowDataset(_windows(panel))
    for sample in dataset:
        for local, node in enumerate(sample["node_indices"]):
            if node == 0:
                assert not sample["evaluation_mask"][:, local].any()


def test_no_neural_windows_is_valid_for_complete_or_unobserved_panel():
    for value in (10.0, np.nan):
        panel = _panel()
        panel.values[:] = value
        assert len(NaturalGapWindowDataset(_windows(panel))) == 0
