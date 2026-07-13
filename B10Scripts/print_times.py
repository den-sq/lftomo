"""Inventory B10 scans and generate per-scan mean flat fields."""

import csv
from datetime import datetime
from os import walk
from pathlib import Path
from zoneinfo import ZoneInfo

import h5py
import numpy as np
import tifffile as tf


scan_dict = {}
new_flats_dir = "yolo_flats"

with open("b10scantimes.csv", "w", newline="\n") as csvfile:
	fieldnames = ["sample", "energy", "scan", "start_time (unix)", "start_time"]
	writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
	writer.writeheader()
	for root, dirs, files in walk("."):
		for fname in files:
			if Path(fname).suffix == ".h5":
				full_path = Path(root, fname)
				base_time = np.array(h5py.File(full_path, "r")["measurement"]["instrument"]["time_stamp"])[0]
				scan_dict[full_path] = {
					"sample": full_path.parts[0],
					"energy": full_path.parts[1],
					"scan": full_path.parts[2],
					"start_time (unix)": base_time,
					"start_time": datetime.fromtimestamp(base_time, tz=ZoneInfo("US/Pacific")).strftime(
						"%Y-%m-%d %H:%M:%S"
					),
				}
	sorted_scan_list = sorted(scan_dict.items(), key=lambda item: item[1]["start_time (unix)"])

	flats_dict = {}
	for x in range(0, len(sorted_scan_list)):
		k, v = sorted_scan_list[x]
		writer.writerow(v)

		flats_dict["gain"] = np.mean(h5py.File(k, "r")["exchange"]["data_white"][:], axis=0)
		flats_dict["dark"] = np.mean(h5py.File(k, "r")["exchange"]["data_dark"][:], axis=0)
		if x < (len(sorted_scan_list) - 1):
			k2, v2 = sorted_scan_list[x + 1]
			flats_dict["gain_post"] = np.mean(h5py.File(k2, "r")["exchange"]["data_white"][:], axis=0)
			flats_dict["dark_post"] = np.mean(h5py.File(k2, "r")["exchange"]["data_dark"][:], axis=0)
		else:
			flats_dict["gain_post"] = flats_dict["gain"]
			flats_dict["dark_post"] = flats_dict["dark"]

		flat_target = k.parent.joinpath(new_flats_dir)
		flat_target.mkdir(parents=True, exist_ok=True)
		for flat_name, flat_data in flats_dict.items():
			tf.imwrite(flat_target.joinpath(f"{flat_name}.tif"), flat_data)


print(scan_dict)
