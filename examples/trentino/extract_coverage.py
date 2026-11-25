"""Extract measured-hour counts from the original Giano release station cards."""

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "history", type=Path, help="Path to the released giano-history directory"
    )
    history = parser.parse_args().history
    files = json.loads((history / "manifest.json").read_text())["files"]
    writer = csv.writer(sys.stdout, lineterminator="\n")
    writer.writerow(
        ("station", "variable", "observed_hours", "span_hours", "missing_hours")
    )
    for name in sorted(files):
        if not name.startswith("stations/") or not name.endswith(".json"):
            continue
        raw = (history / name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != files[name]:
            raise ValueError(f"Station card checksum mismatch: {name}")
        card = json.loads(raw)
        for variable, fields in sorted(card["variables"].items()):
            coverage = fields.get("data_coverage")
            if coverage:
                writer.writerow(
                    (
                        card["id"],
                        variable,
                        coverage["observed_hours"],
                        coverage["hours_between_first_and_last"],
                        coverage["natural_missing_hours_inside_coverage"],
                    )
                )


if __name__ == "__main__":
    main()
