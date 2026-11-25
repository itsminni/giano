# Swiss hourly station example

The two CSV files contain a 2024 extract from NOAA ISD-Lite for Locarno-Monti (067600-99999), Magadino-Cadenazzo (067620-99999) and Lugano (067700-99999). The identifiers combine the USAF and WBAN codes. The station catalogue uses WGS84 coordinates from the ISD history file. These are nominal hourly reports, which can differ from hourly averages. [NOAA format specification](https://www.ncei.noaa.gov/pub/data/noaa/isd-lite/isd-lite-format.txt)

The observations table carries UTC timestamps, temperature in °C, wind speed in m/s and wind direction in degrees. Source missing codes were removed before unit conversion. Calm directions are empty. ISD-Lite omits the original quality flags. The extract has no derived humidity, pressure or rainfall series. The source manifest records the four NOAA file URLs, their original SHA-256 digests and the digests of the two included CSVs. [Source manifest](source_manifest.json)

Run the [coverage notebook](../../notebooks/01_station_coverage.ipynb) to inspect reporting gaps before filling. To reproduce the temperature comparison from the repository root, run the following command.

```sh
uv run --locked python examples/evaluate_temperature.py
```

Both methods receive the same masked input. Each outage starts at 00:00 UTC on the first of March, June or September and lasts 6 or 168 hours. The target station loses all three variables for that interval. The two other stations remain visible. The baseline joins the nearest visible temperatures on either side of each gap. Scoring uses only hours with a measured target and an estimate from each method. The [paired comparison notebook](../../notebooks/04_paired_temperature.ipynb) exposes the per-hour results, including unscored hours.

The two-year follow-up tests eight stations, three outage lengths and an offset-adjusted spatial baseline on temperature and wind. [Extended transfer study](../swiss/README.md)
