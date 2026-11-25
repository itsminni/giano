import numpy as np
import torch
from evaluate_temperature import DATA, compare, scores

from giano.data import load_input


def test_paired_temperature_benchmark_reproduces_published_results():
    torch.set_num_threads(2)
    network = load_input(DATA / "observations.csv", DATA / "stations.csv")
    rows = compare(network)
    assert rows.groupby("hours").size().to_dict() == {6: 54, 168: 1512}
    result = scores(rows)
    np.testing.assert_array_equal(result["episodes"], [9, 9])
    np.testing.assert_array_equal(result["cells"], [54, 1499])
    np.testing.assert_allclose(result["giano_mae"], [2.9398, 3.2189], atol=0.001)
    np.testing.assert_allclose(result["linear_mae"], [0.3741, 2.6840], atol=0.001)
