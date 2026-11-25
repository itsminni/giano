# Swiss network transfer study

This study uses the eight closest Swiss ISD stations to 46.167° N, 8.850° E whose catalogue records span 2022–2024. The distance ranking uses 111 km per degree latitude and 77 km per degree longitude in this region. The original Locarno-Monti, Magadino-Cadenazzo and Lugano targets are evaluated in both network configurations. Cimetta, Cevio, Grono, Matro and Monte Generoso extend the network across different elevations. [Station files and frozen checksums](sources.csv) · [NOAA station catalogue](https://www.ncei.noaa.gov/pub/data/noaa/isd-history.csv)

Download the 24 original gzip files from the NOAA public ISD bucket. From the repository root, run:

```sh
mkdir -p runs/swiss/raw
for year in 2022 2023 2024; do
  for site in 067600 067620 067700 067590 067520 067580 067540 067770; do
    curl -fLsS --retry 2 "https://noaa-isd-pds.s3.amazonaws.com/isd-lite/data/$year/$site-99999-$year.gz" -o "runs/swiss/raw/$site-$year.gz" || exit 1
  done
done
uv run --locked python examples/swiss/evaluate.py --raw runs/swiss/raw --local-context --station-reference --output runs/swiss/results
```

Each target loses temperature, wind speed and direction for 6 hours, 7 days or 30 days beginning on 15 January, April, July or October in 2023 and 2024. All three fields are hidden for the whole outage. Giano uses its public retrospective fill routine, with its Trentino-trained weights and station history available before the outage. The inputs carry no external weather fields. The other stations continue to report. 2022 provides local history including a full year before the first outage. ISD-Lite provides nominal hourly reports and omits source quality flags. Calm directions are excluded.

Temporal interpolation uses the nearest visible observations on either side of the gap. The spatial baseline at each hour takes the median across available neighbouring stations after correcting each donor by its median historical difference from the target. At least 24 paired pre-outage readings are required for each offset. For directions it uses circular means for offsets and combined estimates. All direction errors use the shortest angular distance.

The `3-original` and `8-original` rows score the original three targets on the exact same observed hours where all methods work in both networks. The `8-all` rows score all eight targets in the larger network. Temperature, wind speed and direction have separate observation and coverage counts. Tables give each paired hour equal weight within a duration. An episode win requires lower MAE than the named baseline on that episode's paired hours. The output includes the individual estimates and the original observations for each hour. Eight shared start dates and a single region limit the independence of the results. [Current station-context scores](station-context-scores.csv) · [Ensemble with physical references, both options off](scores.csv)
