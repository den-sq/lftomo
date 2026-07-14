"""Regression canary for the archived B10 print_times helper."""

import csv
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np


def _write_scan(path: Path, timestamp: float) -> None:
	path.parent.mkdir(parents=True)
	with h5py.File(path, "w") as handle:
		handle.create_dataset("measurement/instrument/time_stamp", data=[timestamp])
		handle.create_dataset("exchange/data_white", data=np.ones((2, 2, 2), dtype=np.uint16))
		handle.create_dataset("exchange/data_dark", data=np.zeros((2, 2, 2), dtype=np.uint16))


def test_print_times_writes_each_sorted_scan_without_stale_rows(tmp_path):
	_write_scan(tmp_path / "sample" / "20kev" / "scan_late" / "late.h5", 200.0)
	_write_scan(tmp_path / "sample" / "20kev" / "scan_early" / "early.h5", 100.0)
	script = Path(__file__).parents[1] / "B10Scripts" / "print_times.py"

	subprocess.run([sys.executable, str(script)], cwd=tmp_path, check=True, capture_output=True, text=True)

	with (tmp_path / "b10scantimes.csv").open(newline="") as csvfile:
		rows = list(csv.DictReader(csvfile))
	assert [float(row["start_time (unix)"]) for row in rows] == [100.0, 200.0]
	assert [row["scan"] for row in rows] == ["scan_early", "scan_late"]
