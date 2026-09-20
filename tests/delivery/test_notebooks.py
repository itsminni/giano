from __future__ import annotations

import json
from pathlib import Path


def test_maintained_notebooks_are_clean_and_syntactically_valid() -> None:
    notebook_dir = Path(__file__).resolve().parents[2] / "notebooks"
    notebook_paths = sorted(notebook_dir.glob("*.ipynb"))
    assert [path.name for path in notebook_paths] == [
        "01_data_analysis.ipynb",
        "02_reconstruction_diagnostics.ipynb",
        "03_downstream_utility.ipynb",
        "04_reconstruction_example.ipynb",
    ]

    for path in notebook_paths:
        notebook = json.loads(path.read_text(encoding="utf-8"))
        assert notebook["nbformat"] == 4
        assert notebook["metadata"]["kernelspec"] == {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        }
        assert notebook["metadata"]["language_info"] == {"name": "python"}
        assert notebook["cells"]
        for cell_index, cell in enumerate(notebook["cells"]):
            assert cell.get("id"), f"missing cell id in {path}:cell-{cell_index}"
            if cell["cell_type"] != "code":
                continue
            assert cell.get("execution_count") is None
            assert cell.get("outputs") == []
            source = "".join(cell.get("source", []))
            compile(source, f"{path}:cell-{cell_index}", "exec")
