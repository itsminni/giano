# Meteotrentino station coverage

Each row in `station_coverage.csv` represents a station and a measured variable in the original public Giano station explorer. The span runs from the first to the last measurement of that variable at that station. Hours inside that span with no measurement count as missing. A row’s denominator changes with the lifetime of its instrument. For network-wide coverage, divide total observed hours by total span hours. The fraction of observations by itself gives no information about accuracy inside a gap.

The counts come from the `data_coverage` blocks in the station cards distributed in the v1.0.0 release. To regenerate the CSV, extract `artifacts/website/giano-history` from `giano-1.0.0-artifacts.zip` and run the command below from the repository root. The script verifies each source card against the release manifest before reading it. [Original release and data archive](https://github.com/itsminni/giano/releases/tag/v1.0.0)

```bash
python examples/trentino/extract_coverage.py /path/to/giano-history > station_coverage.csv
```

## Training preview

`training-preview.npz` contains the first eight official training windows and the first eight development windows, prepared from the public Drive files using `giano-prepare --preview`. Array names group fields by split. The target masks reflect the same quality exclusions as the full run. The training notebook opens this small archive to demonstrate a real gradient update without downloading the full network. [Download the original observations](https://drive.google.com/drive/folders/1DPsYf6dVRtCqxE3mDYAx4evTj0WASXK7?usp=sharing)

## Historical evaluation table

`historical-scores.csv` is the compact output of `giano-evaluate --data data --output runs/evaluation` with the packaged weights. Each row represents one variable, duration and information regime. The 480 episodes share 90 such combinations. Read the paired count before interpreting an error. Wind has no common targets in this historical evaluation. Notebook 04 shows how to compare the table with a fresh evaluation.

`release-comparison.csv` scores interpolation, the first release, its fair BiLSTM baseline and the current ensemble on the same target hours. The `all` rows include every paired estimate. The `native` rows cover the subset where the legacy neural windows produced predictions. [Comparison protocol and reproduction commands](https://github.com/itsminni/giano/wiki/Model-and-evidence#comparison-with-the-first-release-on-the-same-missing-hours)

`station-context-scores.csv` contains the same 2020 replay with both `--local-context` and `--station-reference`. The website reports the pooled MAE computed by weighting each row by its paired hours. Rows with zero paired hours have no MAE.
