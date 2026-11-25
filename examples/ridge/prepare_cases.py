"""Export the original Ridge fit, masks and controls using the Giano 1.0 environment."""

import argparse
from importlib import import_module
from importlib.util import find_spec
from pathlib import Path

import numpy as np

CASES = (("point", 0.3), ("block", 12), ("terminal", 12), ("empirical", 0.95))
VARIABLES = (
    "temperature",
    "humidity",
    "pressure",
    "precipitation",
    "wind_direction",
    "wind_speed",
)


def prepare(data, output, variable):
    ridge_forecaster = import_module("giano.downstream.ridge_forecaster")
    protocol = import_module("giano.evaluation.forecasting.protocol")
    selection = import_module("giano.evaluation.forecasting.run")
    cases = import_module("giano.evaluation.gapfill.cases")
    gap_masks = import_module("giano.evaluation.gapfill.masks")
    dataset = import_module("giano.spatiotemporal_dataset")
    panel = dataset.build_spatiotemporal_panel(data / "2-processed-v2", variable)
    model = ridge_forecaster.AutoregressiveRidgeForecaster()
    size = len(panel.times)
    end_train = int(size * 0.7)
    args = dict(history_length=72, horizons=model.config.horizons)
    try:
        node, _ = selection.select_station(panel, **args, stride=24, requested=None)
    except selection.NoEligibleStationError:
        if variable != "wind_speed":
            raise
        print(f"{variable}: no eligible station", flush=True)
        return
    series = panel.values[:, node]
    origins = ridge_forecaster.complete_origins(
        series, **args, start=int(size * 0.85), end=size, stride=24
    )[:200]
    model.fit_series(series, panel.times, end_train)
    lengths = gap_masks.empirical_block_lengths(
        panel.values, panel.coverage_bounds, end_train, quantile_cap=0.95, max_length=71
    )
    masks = []
    predictions = []

    def capture(corrupted, visible, times):
        masks.append(~visible)
        return corrupted

    for origin in origins:
        for seed in range(42, 47):
            for kind, parameter in CASES:
                histories = protocol.prepare_origin_histories(
                    series,
                    panel.times,
                    origin,
                    72,
                    cases.BenchmarkCase(kind, parameter),
                    seed=seed,
                    direction=variable == "wind_direction",
                    empirical_lengths=lengths,
                    repairers={"capture": capture},
                )
                predictions.append(
                    [
                        model.predict_history(histories[key], panel.times[origin])
                        for key in ("clean", "corrupted", "interpolation")
                    ]
                )
    ridge = model.regressor
    np.savez_compressed(
        output / f"{variable}.npz",
        station=panel.station_ids[node],
        times=panel.times[origins],
        indices=origins,
        masks=np.reshape(masks, (-1, 20, 72)),
        history=series[origins[:, None] + np.arange(-71, 1)],
        targets=series[origins[:, None] + np.array(model.config.horizons)],
        controls=np.reshape(predictions, (-1, 20, 3, 4)),
        calendar=np.stack(
            [
                model.design_row(series[i - 71 : i + 1], panel.times[i])[-4:]
                for i in origins
            ]
        ),
        x_mean=ridge.x_mean,
        x_scale=ridge.x_scale,
        y_mean=ridge.y_mean,
        coefficients=ridge.coefficients,
    )
    print(f"{variable}: {panel.station_ids[node]}, {len(origins)} origins", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if find_spec("giano.spatiotemporal_dataset") is None:
        parser.error(
            "Run this exporter in the Giano v1.0.0 environment (see https://github.com/itsminni/giano/wiki/Downstream-Utility)"
        )
    args.output.mkdir(parents=True, exist_ok=True)
    for variable in VARIABLES:
        prepare(args.data, args.output, variable)
