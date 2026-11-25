# Giano

Giano reconstructs missing hourly measurements in weather station records. Give it a station network and a date range, and it writes a NetCDF file containing the original measurements, the reconstructed series and the source of each estimate. It is designed for retrospective gap filling, when observations from before and after an outage are available.

This is the official implementation of Giano. The project began at Fondazione Bruno Kessler’s WebValley 2025 summer school using the Meteotrentino network and reached the finals of the **Premio G.F. Marilli 2026**. [Project website](https://itsminni.github.io/giano/) · [Project wiki](https://github.com/itsminni/giano/wiki/Home)

**Data and model availability.** The trained weights and a small Swiss station dataset come with the repository. For the full Trentino analysis, training and evaluation, download the two folders named `1-raw/` and `2-processed-v2/`. [Download the Meteotrentino dataset on Google Drive](https://drive.google.com/drive/folders/1DPsYf6dVRtCqxE3mDYAx4evTj0WASXK7?usp=sharing)

## Fill gaps in a station record

With Git and uv installed, run the following commands in a terminal. The example runs on CPU with Python 3.11 and the packaged weights. [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/)

```bash
git clone https://github.com/itsminni/giano.git
cd giano
uv sync --locked --python 3.11
uv run --locked giano --input examples/noaa-2024/observations.csv \
  --stations examples/noaa-2024/stations.csv \
  --start 2024-03-01 --end-exclusive 2024-04-01 --local-context --station-reference --output march.nc
```

The input contains a year of observations from Locarno-Monti, Magadino-Cadenazzo and Lugano. The output covers March. Across the three stations, Giano preserves all 1,803 measured temperatures and estimates the 429 missing ones. It leaves 161 wind directions undefined because the measured wind speed was zero. [Example data and provenance](examples/noaa-2024/README.md)

To inspect the values estimated for one station, open `march.nc` in Python with the project environment:

```python
import xarray as xr

with xr.open_dataset("march.nc") as ds:
    station = ds.sel(station="067600-99999")
    estimates = station.temperature.isnull() & station.temperature_filled.notnull()
    fields = ["temperature_filled", "temperature_method", "temperature_reference"]
    print(station[fields].where(estimates, drop=True).to_dataframe().head())
```

## What gets filled

Giano preserves the measured values. For each missing hour inside an instrument’s record it looks for support in neighbouring stations, external weather fields or station history. The output records the method used in `*_method`. Hours without sufficient support stay empty. The model reads six variables together and learns corrections for temperature, humidity, pressure and rain. Wind starts from a physical estimate. For outages of at least a week, local context can estimate wind speed and direction from nearby stations with enough shared history.

The example enables `--local-context --station-reference`, the configuration reported below. It can use observations on either side of a short gap, learn how nearby stations track the target during longer outages, and check pressure against the station’s recent history. Both options are off by default. To use your own data, supply an hourly NetCDF or the same pair of CSVs as the example. [Input formats, units and reconstruction rules](https://github.com/itsminni/giano/wiki/Input-and-output) · [Architecture](https://github.com/itsminni/giano/wiki/Giano-Architecture)

## Explore, train, reconstruct and evaluate

Start with the four notebooks. Their example data are included, so you can work through the entire sequence before downloading the full Trentino dataset.

```bash
uv sync --locked --group notebooks
uv run --locked --group notebooks jupyter lab notebooks/
```

| Notebook | What you will do |
|---|---|
| [01 · Station coverage](notebooks/01_station_coverage.ipynb) | Explore missing observations across 497 instrument series at 171 Trentino stations. |
| [02 · Train Giano](notebooks/02_train_giano.ipynb) | Train both members on a small sample of real windows and use the exported weights for inference. |
| [03 · Reconstruct gaps](notebooks/03_reconstruct_gaps.ipynb) | Fill natural gaps in Swiss station records and inspect the method recorded for each hour. |
| [04 · Evaluate reconstructions](notebooks/04_paired_temperature.ipynb) | Hide known temperatures, score Giano and linear interpolation on the same hours, and inspect the Trentino results. |

### Reproduce the official weights and evaluation

Place the downloaded folders at `data/1-raw/` and `data/2-processed-v2/`. From the repository root, run:

```bash
uv run --locked giano-prepare --data data --output runs/windows
uv run --locked giano-train --cache runs/windows --output runs/training --fixed-rounds
uv run --locked giano-evaluate --data data --ensemble runs/training/ensemble.pt --local-context --station-reference --output runs/evaluation
```

Training selects CUDA, Apple Metal or CPU according to availability. The fixed-round recipe reproduces the selected stopping points for both members. Pass `--ensemble runs/training/ensemble.pt` to use your newly trained weights for inference. [Full training and evaluation guide](https://github.com/itsminni/giano/wiki/Training-and-reproduction)

## Reconstruction accuracy

In the historical Meteotrentino evaluation we hid measured hours at 16 stations. The table compares mean absolute error on the same targets over 480 simulated outages lasting one hour to 90 days. Longer gaps contribute more hours.

| Hourly variable | Giano with both station options | Giano 1.0 | Interpolation |
|---|---:|---:|---:|
| Temperature (°C) | 1.894 | 4.680 | 4.780 |
| Relative humidity (percentage points) | 12.573 | 18.550 | 18.854 |
| Station pressure (hPa) | 4.358 | 5.987 | 6.034 |
| Precipitation (mm) | 0.112 | 0.190 | 0.193 |

Giano improved on the first release in 41 of 60 comparisons by variable, outage duration and available inputs. Fifteen of the 19 higher errors occurred on one- or six-hour gaps. The Trentino comparison had no paired wind observations. [Full comparison, including the WebValley BiLSTM](https://github.com/itsminni/giano/wiki/Model-and-evidence#comparison-with-the-first-release-on-the-same-missing-hours)

In Switzerland, Giano had lower average error than calibrated neighbours at each of eight stations for temperature, wind speed and wind direction during 7- and 30-day outages. For 30-day temperature gaps, the errors were **1.01 °C** for Giano and **1.85 °C** for calibrated neighbours. On six-hour gaps, linear interpolation was more accurate for temperature and wind speed. [Gap-filling evaluation](https://github.com/itsminni/giano/wiki/Model-and-evidence)

A later test used the frozen release on 432 simulated outages at 16 German stations in 2026. With neighbours reporting, Giano beat interpolation for all six variables at 7, 30 and 90 days. For 30-day gaps, calibrated neighbours were more accurate on pressure and precipitation. During 30-day network-wide blackouts Giano underestimated precipitation volume and interpolation was better on both wind measures. [Final test protocol and scores](examples/dwd/README.md#final-release-test-on-2026-observations)

In the Ridge forecast study, when both filling methods could use one extra week of history, Giano gave lower forecast error for temperature, humidity, pressure and rain. Interpolation gave lower error for wind direction. [Forecast study and paired results](https://github.com/itsminni/giano/wiki/Downstream-Utility)

## Project history and credits

Giano 1.0 used separate models for each variable. Its station reconstructions appear in the website explorer and its forecast study used those reconstructions as inputs. Release 2.0 introduced the multivariate ensemble evaluated in Trentino, Switzerland and Germany. [Station explorer](https://itsminni.github.io/giano/?tab=mappa) · [Original implementation](https://github.com/itsminni/giano/tree/dfc63ccbb47b4eb6a0845c1ece91180b0b32beac)

Anita Cappello, Gabriele Mininni, Martina Pellegrini, Daniele Corn and Michele Pio Lomartire developed Giano with Rishabh Wanjari as their main tutor. The original work drew on ImputeFormer, Meteotrentino station observations and ERA5 reanalysis. [WebValley 2025](https://webvalley.fbk.eu/challenge/) · [ImputeFormer](https://doi.org/10.1145/3637528.3671751) · [Meteotrentino](https://www.meteotrentino.it/) · [ERA5](https://doi.org/10.1002/qj.3803) · [Premio G.F. Marilli 2026](https://projects.hackersgen.com/progetti/2026)
