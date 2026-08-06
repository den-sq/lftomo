from collections.abc import Mapping, Callable
from dataclasses import Field, fields
from functools import partial
from multiprocessing import shared_memory, Pool, cpu_count
from multiprocessing.pool import ThreadPool
from multiprocessing.connection import Connection
from os import PathLike
from pathlib import Path
from typing import Sequence

import h5py
import numpy as np
from natsort import natsorted
from numpy.typing import ArrayLike
from skimage import measure
import tifffile as tf

from .log import log, LOG
from .mem import SharedNPArray, SharedNPManager
from .shapes import ReconOrder, ContigMemIter, DataRange, DataShape, ProjOrder
from .window import TomoWindow


def np_convert(dtype: np.dtype, source: ArrayLike):
	if np.issubdtype(dtype, np.integer):
		dtype_range = np.iinfo(dtype).max - np.iinfo(dtype).min
		source_range = np.max(source) - np.min(source)

		# Avoid divide by 0, esp. as numpy segfaults when you do.
		if source_range == 0.0:
			source_range = 1.0

		return (source * max(int(dtype_range / source_range), 1)).astype(dtype)
	elif np.issubdtype(dtype, np.floating):
		return source.astype(dtype)


def byteread_helper(target: str, image: PathLike, source_offsets: int | ArrayLike,
					target_offsets: int | ArrayLike, size: int):
	""" Sinogram order-capable reader using direct buffer reading.

		TODO: Figure out if you can get rid of for loop - order matters.

		:param target: Shared memory to read into.
		:param image: Path to image to read from.
		:param i_dtype: Data type of image.  Unused here, since reading is direct buffer.
		:param offsets: Array of memory offsets to read into.  Should be sequential in file.
		:param size: Size of chunk of memory to read.
	"""
	if isinstance(source_offsets, Sequence) and isinstance(target_offsets, Sequence):
		raise ValueError("At least one offset should be length 1 or this gets real confusing, don't try it.")

	sm = shared_memory.SharedMemory(name=target)
	with open(image, "rb") as handle:
		if not isinstance(source_offsets, Sequence):
			handle.seek(source_offsets)
			if not isinstance(target_offsets, Sequence):
				handle.readinto(sm.buf[target_offsets:target_offsets + size])
			else:
				for i in range(len(target_offsets)):
					handle.readinto(sm.buf[target_offsets[i]:target_offsets[i] + size])
		else:
			for i in range(len(source_offsets)):
				handle.seek(source_offsets[i])
				handle.readinto(sm.buf[target_offsets + i * size:target_offsets + (i + 1) * size])
	sm.close()


def distribute_tiff_read(source_files: Sequence[PathLike], target_mem: SharedNPArray,
							y_start: int = 0, y_stop: int | None = None,
							thread_max: int = cpu_count() * 2) -> None:
	"""Read an uncompressed TIFF stack using recon_minimal's per-file dispatch.

	Unlike ``read_window``, TIFF metadata is inspected once and each worker gets
	one contiguous source/target pair. Multi-page TIFFs use one source offset per
	page. This is the canonical TIFF-stack path selected by the Tier-2 benchmark
	for issue #23.
	"""
	files = [Path(path) for path in source_files]
	if not files:
		raise ValueError("TIFF source list is empty")
	with tf.TiffFile(files[0]) as tif:
		pages = tif.pages
		page = pages[0]
		if page.compression != 1 or not page.is_contiguous:
			raise ValueError("direct TIFF reading requires uncompressed contiguous pages")
		rows, columns = page.shape
		dtype = np.dtype(page.dtype)
		stop = rows if y_stop is None else y_stop
		if not 0 <= y_start < stop <= rows:
			raise ValueError(f"invalid TIFF Y window [{y_start}, {stop}) for {rows} rows")
		page_offsets = [entry.dataoffsets[0] for entry in pages]

	count = len(files) if len(files) > 1 else len(page_offsets)
	window_rows = stop - y_start
	expected = ProjOrder(Theta=count, Y=window_rows, X=columns)
	if target_mem.shape != expected or target_mem.dtype != dtype:
		raise ValueError(
			f"target must have shape {expected} and dtype {dtype}, got {target_mem.shape} and {target_mem.dtype}"
		)
	row_bytes = columns * dtype.itemsize
	image_bytes = window_rows * row_bytes

	if len(files) == 1:
		source_offsets = [offset + y_start * row_bytes for offset in page_offsets]
		calls = [(target_mem.name, files[0], source_offsets, 0, image_bytes)]
	else:
		base_offset = page_offsets[0] + y_start * row_bytes
		calls = [
			(target_mem.name, path, base_offset, index * image_bytes, image_bytes)
			for index, path in enumerate(files)
		]
	with ThreadPool(min(thread_max, count)) as pool:
		pool.starmap(byteread_helper, calls)


def get_file_set(source_files: ArrayLike, shape: type = ProjOrder):
	# TODO: Handle 4D
	# TODO: Handle limited iteration over theta
	suffix = Path(source_files[0]).suffix
	if suffix == ".h5":
		# Single h5 file with dataset.
		h5_ds = h5py.File(source_files[0], 'r')['exchange']['data']
		itemsize = np.dtype(h5_ds.dtype).itemsize

		source_base_offset = h5_ds.id.get_offset()
		# input_shape = shape(h5_ds.shape)
		input_stride = shape(h5_ds.nbytes // len(h5_ds), h5_ds.shape[2] * itemsize, itemsize)
		source_set = [source_files[0]] * len(h5_ds)

	elif suffix in [".tif", ".tiff"]:
		with tf.TiffFile(source_files[0]) as tif:
			itemsize = np.dtype(tif.pages[0].dtype).itemsize

			source_base_offset = tif.pages[0].dataoffsets[0]
			if len(source_files) == 1: 	# Multi-Page Tiff
				index_offset = tif.pages[1].dataoffsets[0] - source_base_offset
				source_set = [source_files[0]] * len(tif.pages)
			else: 					# Single Page Tiff(s)
				index_offset = 0
				source_set = natsorted(source_files)
			# input_shape = shape(len(source_set), *tif.pages[0].shape)
			input_stride = shape(index_offset, tif.pages[0].shape[1] * itemsize, itemsize)

	return source_base_offset, input_stride, source_set 	# input_shape


def file_interval_pad(base: DataRange, shape: DataShape, file_fields: list[str | Field]):
	intervals = [f.name if isinstance(f, Field) else f for f in file_fields]
	full_range_fields = set(intervals + [f.name for f in fields(base.step)])
	if len(full_range_fields) == len(fields(base.step)):
		return base
	else:
		# File stop points from shape object
		fstop = shape.extract(DataShape.gen(intervals))

		return DataRange(base.start.merge(fstop.make_with(0)),
						base.stop.merge(fstop),
						base.step.merge(fstop.make_with(1)))


def read_window(window: TomoWindow,
					source_files: ArrayLike,
					target_mem: SharedNPArray,
					thread_max: int = cpu_count() * 2,
					mem_window: TomoWindow = None,
					file_read_func: Callable = get_file_set,
					read_func: Callable = byteread_helper,
					file_fields: list[str | Field] = ["Theta"]):
	# TODO: iteration over file fields isn't checked for in source set.
	# TODO: padding?
	# TODO: Revise mem iter setup.
	# f_range.extract(DataShape.gen(file_fields)) as a pass
	log.write("RW START")

	mem_offset = 0

	# If we don't have a separate internal window, just clone ours.
	if mem_window is None:
		mem_window = window.internal

	file_offset, input_stride, source_set = file_read_func(source_files, type(window.shape))

	log.write("RW FILE INFO")

	m_range = file_interval_pad(mem_window.frange, window.shape, file_fields)
	f_range = file_interval_pad(window.frange, window.shape, file_fields)

	# Create indicies.
	i_mem = m_range.get_iter(conv=partial(ContigMemIter, offset=mem_offset, shape=window.shape, stride=window.stride,
							jump=file_fields))
	i_file = f_range.get_iter(conv=partial(ContigMemIter, offset=file_offset, shape=window.shape, stride=input_stride,
							jump=file_fields))

	m_contig = i_mem.conv.contig_stride
	f_contig = i_file.conv.contig_stride

	if (m_contig % f_contig) and (f_contig % m_contig):
		log.write("FILE READ", "Source and target memory not fully contiguous.", log_level=LOG.WARN)

	log.write("RW INDICIES")

	def addr_batch(addr, count, comp):
		vals = np.fromiter(addr, dtype=int, count=count)
		return vals[::count // len(source_set)] if comp else vals.reshape(len(source_set), count // len(source_set))

	# log.write("FILE READ", f"read length {read_length} from {i_mem.conv.contig_stride, i_file.conv.contig_stride}")

	m_addr = addr_batch(i_mem, len(m_range), m_contig >= f_contig)
	f_addr = addr_batch(i_file, len(f_range), f_contig >= m_contig)

	log.write("RW ADDR")

	sources = zip(m_addr, f_addr, source_set)

	log.write("RW ZIP")

	# Disperse reading amongst threads.
	# We're reading in contiguous memory blocks, so go with the shortest of the two memory steps.
	with ThreadPool(thread_max) as pool:
		pool.starmap(read_func, [(target_mem.name, image_path, file_addr, mem_addr, min(m_contig, f_contig))
									for mem_addr, file_addr, image_path in sources])

	log.write("RW READ")


def read_file(filename, source_shape, source_type, source_base_offset, target_mem, source_slice):
	source = np.memmap(filename, dtype=source_type, mode='r', shape=source_shape, offset=source_base_offset)
	with target_mem as tm:
		tm[:] = source[source_slice]


def file_loader(conn: Connection, source: ArrayLike):
	win = conn.recv()

	log.write("File Read", "File Loader Started")

	if win is None:
		# Exit early message.
		return
	if not isinstance(win, TomoWindow):
		return

	try:
		with SharedNPManager() as (snm,):
			full_win = win.with_pad()
			read_mem = snm.SharedNPArray(full_win.shape, full_win.dtype)
			read_window(full_win, source, target_mem=read_mem)
			conn.send(read_mem)
			try:
				while conn.recv() is not None:
					log.write("File Read", "Reading Window")
					read_window(win.with_pad(), source, target_mem=read_mem)
					conn.send(read_mem)
			except EOFError:
				log.write("File Read", "EOF Reached, exiting.")
	except Exception as ex:
		log.write("File Read", f"Error while reading files: {ex}")

	log.write("File Read", "Exiting.")
	conn.close()




def write_centers(center_mem: SharedNPArray, path: PathLike, center_range, angles,
					proc_count: int = cpu_count(), calc_entropy: bool = True,
					disk_write: bool = True) -> Mapping:
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

	import tomopy

	center_entropy = {}

	recon_shape = ReconOrder(Y=len(center_range), Z=center_mem.shape.X, X=center_mem.shape.X)

	cf_write_mem = SharedNPArray("cf_write_mem", dtype=np.float32, shape=recon_shape, allocate=True)
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


def write_slice(sha: SharedNPArray, path: PathLike, i=None, out_type: np.dtype = None):
	""" Writes a portion of image in memory to disk.

		:param sha: Metadata for shared memory array.
		:param path: Path on disk to write to.
		:param i: Slice to write.
		:param slice_act: Field of named tuple we are slicing the write at.
	"""
	with sha[i] as data:
		if out_type is None:
			tf.imwrite(path, data)
		else:
			tf.imwrite(path, np_convert(out_type, data))
