from collections import namedtuple
from collections.abc import Mapping, Callable
from multiprocessing import shared_memory, Pool
from os import PathLike
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike
from psutil import cpu_count
from skimage import measure
import tomopy
import tifffile as tf

from .log import log, LOG
from .mem import SharedNP, ReconOrder
from .window import DataWindow


def np_convert(dtype: np.dtype, source: ArrayLike):
	if np.issubdtype(dtype, np.integer):
		dtype_range = np.iinfo(dtype).max - np.iinfo(dtype).min
		source_range = int(np.max(source) - np.min(source))
		return (source * (dtype_range // source_range))
	elif np.issubdtype(dtype, np.floating):
		return source.astype(dtype)


def byteread_helper(target: SharedNP, image: PathLike, i_dtype: np.dtype, offsets: ArrayLike, size: int):
	""" Sinogram order-capable reader using direct buffer reading.

		TODO: Figure out if you can get rid of for loop - order matters.

		:param target: Shared memory to read into.
		:param image: Path to image to read from.
		:param i_dtype: Data type of image.  Unused here, since reading is direct buffer.
		:param offsets: Array of memory offsets to read into.  Should be sequential in file.
		:param size: Size of chunk of memory to read.
	"""
	sm = shared_memory.SharedMemory(name=target)
	with open(image, "rb") as handle:
		handle.seek(offsets[0]["source"])
		for offset in offsets:
			handle.readinto(sm.buf[offset["target"]:offset["target"] + size])
	sm.close()


def distribute_read(target_mem: SharedNP, window: DataWindow, source_files: ArrayLike, sino_order: bool = False,
					thread_max: int = cpu_count(), read_func: Callable = byteread_helper):
	""" Distributes file reading into shared memory across multiple threads.

		Currently can work in projection or sinogram order, maybe?

		:param target_mem: Memory to read files into.
		:param window: Window of data we're reading.
		:param source_files: List of files to read.
		:param read_func: Function that does the actual data reading.
		:param thread_max: Maximum # of threads to use.
	"""
	# Find the offset values for start of blocks.
	# I wrote that this needs to be changed for .. some reason?
	base_offset = target_mem.buffer_address.start

	h_step = window.itemsize * window.shape.X

	block_size = h_step * (window.shape.Theta if sino_order else window.shape.Y)
	size = h_step if sino_order else block_size

	with tf.TiffFile(source_files[0]) as tif:
		offset = tif.pages[0].dataoffsets[0]

	# log.write("Distributing Read", f"Step {h_step} ; Block {block_size} ; Size {size}", LOG.INFO)
	# log.write("Distributing Read", f"Write Offset {base_offset} ; Read Offset {offset}", LOG.INFO)

	def generate_offset_pairs_sino(i):
		return [{"source": offset + (window.start + j) * h_step,
				"target": int(base_offset + j * block_size + i * h_step)}
					for j in range(window.shape.Y)]

	def generate_offset_pairs_proj(i):
		return [{"source": offset + (window.start) * h_step, "target": int(base_offset + i * block_size)}]

	pairs_func = generate_offset_pairs_sino if sino_order else generate_offset_pairs_proj

	log.write("Mem Read", f"{base_offset}|{window.shape.Y}")

	# if sino_order:
	# 	log.write("Read Example", generate_offset_pairs_sino(0), LOG.INFO)
	# else:
	# 	log.write("Read Example", generate_offset_pairs_proj(0), LOG.INFO)

	# Distribute actual reading
	with Pool(thread_max) as pool:
		pool.starmap(read_func,
			[(target_mem.name, image, window.dtype, pairs_func(i), size) for i, image in enumerate(source_files)])


def _slice_at(shape: namedtuple, item: str, value: int):
	temp_slice = [np.s_[:]] * len(shape)
	temp_slice[shape._fields.index(item)] = value
	return tuple(temp_slice)


def write_centers(center_mem: SharedNP, path: PathLike, center_range, angles,
					proc_count: int = cpu_count(), calc_entropy: bool = True, disk_write: bool = True) -> Mapping:
	""" Creates the reconstructions for a set of centers to disk, calculating entropy
		for center finding and/or writing to disk.

		:param center_mem: Metadata for center sinograms.
		:param path: Base path for center files.
		:param center_range: Centers to iterate through.
		:param angles: Reconstruction angles.
		:param proc_count: # of simulatenous processes to use for reconstruction and writing.
		:param calc_entropy: Whether to make an entropy calculation for a center calculation.
		:param disk_write: Whether to write center slices to disk.
	"""

	if not (disk_write or calc_entropy):
		log.write("Center Finding", "Given no tasks (disk write or entropy calculation.)", LOG.ERROR)
		return

	center_entropy = {}

	recon_shape = ReconOrder(Y=len(center_range), Z=center_mem.shape.X, X=center_mem.shape.X)

	with SharedNP("cf_write_mem", dtype=np.float32, shape=recon_shape) as cf_write_mem:
		with center_mem as center_data, cf_write_mem as write_mem:
			for i, cent in enumerate(center_range):
				write_mem[i, :, :] = tomopy.recon(np.expand_dims(center_data, axis=0), angles, center=cent,
													sinogram_order=True, algorithm='gridrec')
				if calc_entropy:
					center_entropy[str(cent)] = measure.shannon_entropy(write_mem[i, :, :])

		if disk_write:
			with Pool(proc_count) as pool:
				pool.starmap(write_slice, [(cf_write_mem, Path(path, f"center_{cent:.2f}.tif"), i)
												for i, cent in enumerate(center_range)])

	if calc_entropy:
		return sorted(center_entropy.items(), key=lambda item: item[1])[0][0]
	else:
		return None


def write_slice(sino_mem: SharedNP, path: PathLike, i, slice_at: str = None, out_type: np.dtype = None):
	""" Writes a portion of sinogram to disk.

		:param sino_mem: Metadata for reconstruction shared memory.
		:param path: Path on disk to write to.
		:param i: Vertical slice(s) (y) of reconstruction to write.
		:param slice_act: Field of named tuple we are slicing the write at.
	"""
	if slice_at is None:
		write_slice = np.s_[i, :, :]
	else:
		write_slice = _slice_at(sino_mem.shape, slice_at, i)

	with sino_mem as sino:
		if out_type is None:
			tf.imwrite(path, sino[write_slice])
		else:
			tf.imwrite(path, np_convert(sino[write_slice]))
