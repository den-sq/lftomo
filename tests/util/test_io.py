from dataclasses import astuple
from multiprocessing import Pipe, Process
from multiprocessing.pool import ThreadPool
from pathlib import Path
import pytest
from time import sleep

from natsort import natsorted
import numpy as np
import tifffile as tf

from lftomo.util.log import log
from lftomo.util.io import np_convert, byteread_helper, file_interval_pad, get_file_set, write_slice
from lftomo.util.io import read_window, read_file, file_loader
from lftomo.util.shapes import X, Y, DataRange, ProjOrder, XSlice
from lftomo.util.mem import SharedNPArray, SharedNPManager
from lftomo.util.window import TomoWindow, TomoDataSet, CTFov


@pytest.fixture(autouse=True)
def _test_logdir(tmp_path):
	log.set_logdir(tmp_path / "logs")


def test_np_convert():
	int_array = np.arange(0, 256, dtype=np.uint8)
	float_array = np.array([x / 255 for x in range(0, 256)], dtype=np.float32)

	print(float_array[0:25])
	print(int_array[0:25])
	print(np_convert(np.uint8, float_array))
	print(np.arange(0, 256, dtype=np.uint8))

	assert np.all(np_convert(np.float32, int_array) == np.arange(0, 256, dtype=np.float32))
	assert np.all(np_convert(np.uint8, float_array) == np.arange(0, 256, dtype=np.uint8))


def test_byteread_helper(tmp_path):
	x = SharedNPArray("ByteReadTest", X(2048), np.uint16, allocate=True)
	image_path = tmp_path / "brhelp.tif"
	tf.imwrite(image_path, np.ones((2048), np.uint16), dtype=np.uint16)
	with tf.TiffFile(image_path) as tif:
		source_offset = tif.pages[0].dataoffsets[0]

	# One each of read/write.
	with x as x_arr:
		x_arr[:] = 52
		assert np.all(x_arr == np.full((2048), 52))
		byteread_helper(x.name, image_path, source_offset, 0, x.size())
		assert np.all(x_arr == np.ones((2048), np.uint16))

	print(type(x_arr))

	read = x.size() // 4

	# Multiple writes for one input read.
	with x as x_arr:
		x_arr[:] = 52
		assert np.all(x_arr == np.full((2048), 52))
		byteread_helper(x.name, image_path, source_offset, [i * read for i in range(0, 4)], read)
		assert np.all(x_arr == np.ones((2048), np.uint16))

	# Multiple reads, each with one write.
	# Also writing only some of memory.
	with x as x_arr:
		x_arr[:] = 52
		assert np.all(x_arr == np.full((2048), 52))
		byteread_helper(x.name, image_path, [source_offset + i * read for i in range(0, 2)], 0, read)
		target_arr = np.ones((2048), np.uint16)
		target_arr[1024:] = 52

		assert np.all(x_arr == target_arr)

	# Get the error:
	with pytest.raises(ValueError):
		byteread_helper(x.name, image_path, [0, 25], [0, 25], read)

	del x


def test_file_interval_pad():
	base = Y.drange(start=1024, stop=1048, step=1)
	shape = ProjOrder(Theta=1501, Y=2048, X=2048)
	file_fields = ["Theta"]

	# Test padding with a new file field.
	result = file_interval_pad(base, shape, file_fields)
	assert astuple(result) == astuple(DataRange(XSlice(Theta=0, Y=1024), XSlice(Theta=1501, Y=1048), XSlice(Theta=1, Y=1)))

	# Test no-op padding.
	result = file_interval_pad(base, shape, ["Y"])
	assert astuple(result) == astuple(base)


@pytest.mark.integration
def test_get_file_set():
	shape = ProjOrder

	# Tiff Stack
	file_list = list(Path("data/input/Sam19").iterdir())
	offset, stride, sources = get_file_set(file_list, shape)
	assert stride == ProjOrder(0, 2048 * 2, 2)
	assert sources == natsorted(file_list)
	assert offset == 256

	# Single Tiff
	file_list = ["data/input/R3B_L3A_FP.tif"]
	offset, stride, sources = get_file_set(file_list, shape)
	assert stride == ProjOrder(28800, 120 * 2, 2)
	assert sources == file_list * 1124
	assert offset == 256

	# Single H5
	# h5py file open is stupid
	file_list = [Path("/mnt/wsl/PHYSICALDRIVE2p1/B10/PCRBox_Octo0/20kev/voi2/20241205_114905_octo_0_centerradula_fov2_scan4.h5")] 	# noqa: E501
	offset, stride, sources = get_file_set(file_list, shape)
	assert stride == ProjOrder(11059200, 2560 * 2, 2)
	assert sources == file_list * 1969
	assert offset == 168206858


def test_write_slice(tmp_path):
	array_mem = SharedNPArray("SliceTest", ProjOrder(5, 24, 2048), np.float32, allocate=True)

	with array_mem as st:
		st[:] = 0.5

	# No data conversion during write.
	image_path = tmp_path / "slice_test_32bit.tif"
	write_slice(array_mem, image_path, np.s_[1, :, :])
	floatimg = tf.imread(image_path)
	assert np.all(floatimg == np.full((24, 2048), 0.5, dtype=np.float32))

	# Data conversion during write.
	image_path = tmp_path / "slice_test_16bit.tif"
	write_slice(array_mem, image_path, np.s_[2:4, :, :], out_type=np.uint16)
	intimg = tf.imread(image_path)
	assert np.all(intimg == np.full((2, 24, 2048), 32767, dtype=np.uint16))


def test_write_centers():
	pass


@pytest.mark.integration
def test_read_file():
	source_files = list(Path("data/input/Sam19").iterdir())
	shape = ProjOrder(Theta=1501, Y=96, X=2048)
	dtype = np.uint16
	source_files = list(Path("data/input/Sam19").iterdir())
	thread_max = 36
	tw = TomoWindow(shape, Y(0), Y(1024), Y(1), np.uint16, np.float32)

	with SharedNPManager() as (snm,):
		input_mem = snm.SharedNPArray(shape, dtype)
		log.write("RF START")

		source_files = natsorted(source_files)

		with tf.TiffFile(source_files[0]) as tif:
			source_base_offset = tif.pages[0].dataoffsets[0]
			source_shape = tif.pages[0].shape

		m_index = tw.internal.index[1:]
		f_index = tw.index[1:]

		log.write("RF SETUP")

		with ThreadPool(thread_max) as pool:
			pool.starmap(read_file, [(source_files[i], source_shape, tw.dtype, source_base_offset,
										input_mem[(i,) + m_index], f_index)
										for i in range(len(source_files))])
		log.write("RF READ")

		out_path = Path("data/output/rf_test/")
		out_path.mkdir(parents=True, exist_ok=True)

		with ThreadPool(thread_max) as pool:
			pool.starmap(write_slice, [(input_mem, out_path.joinpath(f"rf_test_{i}.tif"), np.s_[:, i - tw.pos.Y, :])
											for i in tw.int_iter()])


@pytest.mark.integration
def test_read_window():
	shape = ProjOrder(Theta=1501, Y=96, X=2048)
	dtype = np.uint16
	source_files = list(Path("data/input/Sam19").iterdir())
	thread_max = 36
	tw = TomoWindow(shape, Y(0), Y(1024), Y(1), np.uint16, np.float32)

	with SharedNPManager() as (snm,):
		input_mem = snm.SharedNPArray(shape, dtype)
		read_window(tw, source_files, input_mem, thread_max)

		out_path = Path("data/output/rw_test/")
		out_path.mkdir(parents=True, exist_ok=True)

		with ThreadPool(thread_max) as pool:
			pool.starmap(write_slice, [(input_mem, out_path.joinpath(f"rw_test_{i}.tif"), np.s_[:, i - tw.pos.Y, :])
											for i in tw.int_iter()])


@pytest.mark.integration
def test_file_loader():
	fov = CTFov(1501, 180, 1)
	over = Y(48)
	work_dtype = np.float32
	thread_max = 32
	out_path = Path("data/output/fl_test/")
	out_path.mkdir(parents=True, exist_ok=True)

	tds = TomoDataSet.from_disk(Path("data/input/Sam19"), fov, ProjOrder, over, work_dtype=work_dtype)
	tds_iter = iter(tds)

	log.write("TDS Setup")

	main, loader = Pipe()
	fl = Process(target=file_loader, args=((loader, natsorted(Path("data/input/Sam19").iterdir()))))
	fl.start()

	log.write("FL Started")
	tw = next(tds_iter)
	main.send(tw)

	iter_exhausted = False

	with SharedNPManager() as (snm,):
		proj_mem = snm.SharedNPArray(tw.shape, tw.dtype)

		while not iter_exhausted:
			input_mem = main.recv()
			log.write("Mem Fetched")

			with proj_mem as pm, input_mem as im:
				pm[:] = im[:]

			try:
				twn = next(tds_iter)
				main.send(twn)
				log.write("Window Sent")
			except StopIteration:
				iter_exhausted = True

			with ThreadPool(thread_max) as pool:
				pool.starmap(write_slice, [(proj_mem, out_path.joinpath(f"rw_test_{i}.tif"), np.s_[:, i - tw.pos.Y, :])
												for i in tw.int_iter()])
			log.write("Mem Written")
			tw = twn

	main.send(None)

	main.close()
	loader.close()

	sleep(0.1)

	fl.close()
