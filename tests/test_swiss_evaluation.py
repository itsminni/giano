import numpy as np
import pandas as pd
from swiss.evaluate import ORIGINAL, summarize


def test_network_comparison_pairs_by_identity_and_uses_angular_error():
    time = pd.date_range("2024-01-15", periods=3, freq="h")
    narrow = pd.DataFrame(
        dict(
            station=ORIGINAL[0],
            start=time[0],
            hours=6,
            field="wind_direction",
            time=time,
            truth=[359, 20, 90],
            giano=[1, np.nan, 10],
            source=3,
            linear=1,
            spatial=1,
        )
    )
    wide = narrow.iloc[::-1].copy()
    wide["giano"] = [np.nan, 25, 358]
    summary = summarize(narrow, wide)
    groups = summary.query("station=='all' and start=='all'").set_index("network")
    assert groups.loc["3-original", "giano"] == 2
    assert groups.loc["8-original", "giano"] == 1
    assert groups.loc["8-all", "giano"] == 3
    assert groups["paired"].to_dict() == {"3-original": 1, "8-original": 1, "8-all": 2}
