# External replication on German weather stations

This study compares the frozen Giano ensemble with its optional local-context mode on hourly observations from Deutscher Wetterdienst (DWD), Climate Data Center. The data come from the historical, quality-checked products for air temperature and relative humidity (`TT_TU`, `RF_TU`) and station-level barometric pressure (`P0`, hPa). Missing codes and readings outside the input bounds are excluded. Data attribution: Deutscher Wetterdienst, CDC, CC BY 4.0. [Temperature archive](https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/hourly/air_temperature/historical/) · [Pressure field definitions and licence](https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/hourly/pressure/DESCRIPTION_obsgermany_climate_hourly_pressure_en.pdf)

The eight closest station IDs with both historical products spanning January 2023 through December 2025 were selected by great-circle distance from Hamburg (53.55° N, 10.00° E) and Munich (48.14° N, 11.58° E). Each region has its own eight-station network. A frozen manifest records the coordinates, elevation, archive filenames and SHA-256 digests. Both modes use the same Trentino-trained weights, no external weather fields, and local reference statistics fitted on 2023. For long temperature gaps, local context fits a Ridge regression on up to a year of paired pre-gap measurements from the target and nearby stations. The weekly donor offsets remain available for humidity, pressure and temperature when regression is unsupported. [Station and source manifest](sources.csv)

## Reconstruction

Each station is the target on 5 January at 03 UTC, 5 April at 09 UTC, 5 July at 15 UTC and 30 September at 21 UTC in both 2024 and 2025. Hide all target fields for 1, 6, 168, 720 or 2,160 hours. In two additional scenarios lasting 6 and 720 hours, withhold every station during the gap. The baseline interpolates between the nearest measured target values on both sides. Score both modes and the baseline on identical finite target hours, keeping the hours where local context fell back to the default. The counts distinguish measured, supported, paired and locally replaced values.

The replication criterion was fixed before inspecting predictions. For each variable and gap duration with donors, local-context MAE had to improve in **each of four region–year cells**, with equal coverage. Tables pool absolute errors by the number of paired hours within each cell. Multiple durations at the same start overlap, and stations in one region share weather events. [Region–year scores](scores.csv)

## Downstream check

A secondary study uses 72-hour station histories ending at the forecast origin. Fit one common Ridge per variable and region on complete 2023 training histories and targets from all eight stations, with standardised inputs, an unpenalised intercept and `alpha=1`. Forecasts use the same four calendar features and 1, 6, 12 and 24 hour horizons as the Trentino study. Schedule origins on the 10th and 24th of February (03 UTC), May (09 UTC), August (15 UTC) and November (21 UTC) in both test years. Skip an origin for a variable when its history or any target is missing. Apply the same point, middle-12, terminal-12 and terminal-48 masks to both modes. Interpolation only sees the visible history through the origin. Both modes receive the same regression coefficients. Average forecast errors within station first. This secondary comparison was specified after inspecting the reconstruction results and before computing forecast errors.

## Reproduce

Run from the project root. The download uses the filenames in the frozen manifest. A future DWD release may rotate historical filenames. The evaluator verifies the downloaded bytes against the manifest.

```sh
mkdir -p runs/dwd/raw
python3 - <<'PY'
import csv
import subprocess
from pathlib import Path

base = "https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/hourly"
with Path("examples/dwd/sources.csv").open() as f:
    for station in csv.DictReader(f):
        for product in ("air_temperature", "pressure"):
            name = station[product]
            url = f"{base}/{product}/historical/{name}"
            subprocess.run(["curl", "-fLsS", "--retry", "3", url, "-o", f"runs/dwd/raw/{name}"], check=True)
PY
uv run --locked python examples/dwd/evaluate.py --raw runs/dwd/raw --output runs/dwd/reconstruction.csv
PYTHONPATH=examples uv run --locked python -m dwd.forecast --raw runs/dwd/raw --output runs/dwd/forecast.csv
```

Reconstruction output rows hold the counts and sums of absolute errors for each episode and variable. Divide summed errors by summed paired counts to obtain the reported MAEs. The forecast file records the original target and one prediction from each method for each eligible station, origin, mask and horizon.

## Final release test on 2026 observations

We froze the checkout's implementation, distributed weights and this protocol before scoring the 2026 targets. The test retains the 16 stations from the earlier German study and reserves four UTC start dates between January and September 2026. It covers 432 simulated station outages. Durations are 1 hour, 6 hours, 7 days, 30 days and 90 days with neighbours reporting, plus 6-hour and 30-day outages with all stations masked. The last date has no 90-day case because the recent feed did not yet span that interval. All six variables are scored where observed. Four stations lacked a recent precipitation file.

We compare the release with both station options enabled, the standard configuration, temporal interpolation and offset-calibrated neighbours. The spatial control uses the preceding week to estimate offsets and is available when neighbours report. Giano's station reference uses observations before each outage. Retrospective interpolation can also use the visible boundary after an outage. The test uses DWD's provisional recent products as downloaded on 25 November 2026 and includes no external weather fields. The underlying weights and rules were fixed. The test provides a later-year replication on the same German stations.

### Thirty-day outages with neighbours reporting

| Variable | Paired hours | Giano | Standard | Interpolation | Calibrated neighbours |
|---|---:|---:|---:|---:|---:|
| Temperature (°C) | 45,794 | 0.799 | 1.685 | 5.175 | 1.111 |
| Relative humidity (percentage points) | 45,792 | 5.282 | 6.979 | 16.223 | 5.282 |
| Station pressure (hPa) | 45,773 | 0.727 | 6.252 | 6.875 | 0.569 |
| Precipitation (mm) | 34,304 | 0.066 | 0.066 | 0.086 | 0.061 |
| Wind speed (m/s) | 45,512 | 0.616 | 0.836 | 1.574 | 0.772 |
| Wind direction (°) | 45,385 | 23.441 | 30.187 | 77.369 | 29.215 |

Each row reports MAE on the same hidden measurements. Giano covered 1,182,019 of 1,182,021 measured target occurrences. The two exclusions are wind direction in overlapping January gaps at one Munich station. Cases at different durations and regimes can reuse hours. Wind direction uses the shortest angular distance.

Giano beat interpolation in all 18 long-gap comparisons with neighbours reporting. Calibrated neighbours scored better on pressure and precipitation in the 30- and 90-day comparisons. In 30-day network-wide blackouts, Giano's wind errors exceeded interpolation. Giano also underestimated the precipitation volume in those blackouts and detected none of the observed wet hours at the 0.1 mm threshold. The CSV reports RMSE, coverage, wet-hour F1 and accumulated precipitation alongside MAE. The run used Python 3.11, PyTorch 2.13.0 and two CPU threads on Apple M4 Pro. [Full scores](holdout-scores.csv) · [Frozen protocol](holdout-protocol.json) · [Source checksums and missing files](holdout-sources.csv)

To reproduce this test, put the ZIPs identified in the source receipt in a local raw-data folder. The recent files are updated by DWD. Use copies with the recorded hashes.

```sh
uv run --locked python -m examples.dwd.evaluate_holdout \
  --freeze examples/dwd/holdout-protocol.json \
  --sources examples/dwd/holdout-sources.csv --raw path/to/dwd-2026 \
  --output runs/dwd-2026
```
