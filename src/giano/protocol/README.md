# Trentino reproduction protocol

These files define the inputs and decisions needed to repeat the official training run. The measurements themselves come from the Drive folders. The training code reads the protocol as data. It fits the network by gradient descent on the selected observations.

| File | Contents |
|---|---|
| sources.json | Ordered station roster and digests of the 497 hourly series and original ERA5 surface-pressure files. Some files named for precipitation also contain the pressure field. |
| train.npz, val.npz | One row per retained window in the 2009–2010 and 2011 splits. The six columns of `windows` are station index, outage start, exclusive end, regime index, first core hour and pressure-calibration index. Times are hour offsets from 1 January 2009. Regimes follow `REGIMES` in `windows.py`. |
| pressure.npz | Offsets, supporting pair counts and validity masks for 383 pressure references. These were fitted on quality-filtered 2009–2010 observations, excluding the simulated training outage. A station needs 24 finite pressure pairs. |
| initial.pt | The original random state dictionaries before either seed received a gradient update. |
| learning-curves.json | Original training losses, 2011 development scores and selected rounds for both seeds. Fixed-round training reads only the selected round. The full selection procedure calculates the scores anew. |
| historical.json, test-support.npz | Stations, start dates, durations, regimes, auxiliary sources and per-station eligibility masks for scoring the fixed 2020 experiment. The training command never opens these files. |

Both window archives store the common target quality mask packed along the 72-hour axis, membership in the original subset used by the sampler and a SHA-256 digest of each source window before the pressure input and target-mask changes. The protocol retains observations for scoring when the value agrees before and after removing raw quality code 140. The model's inputs use the original processed observations. This distinction keeps the predictor inputs consistent with the published run.

After training, inference combines the average of the two learned predictors with the retained physical references. The station statistics, humidity coefficients, rain-donor correction and rainfall-volume ratios are distributed in `giano/weights/`.
