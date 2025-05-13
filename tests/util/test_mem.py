from dataclasses import astuple
from sys import getsizeof

import numpy as np

from lftomo.util.mem import SharedNPManager
from lftomo.util.shapes import SinoOrder, ProjOrder, ReconOrder


def test_alt_creation():
	po = ProjOrder(Y=12, Theta=1501, X=2048)
	with SharedNPManager(queue_mem=[(po.to(SinoOrder), np.uint16), ((po, np.uint16), (ReconOrder.of(po), np.uint16))]) as \
		(_snm, sino_mem, proj_mem, recon_mem):
		assert sino_mem.shape == SinoOrder(Y=12, Theta=1501, X=2048)
		assert proj_mem.shape == ProjOrder(Y=12, Theta=1501, X=2048)
		assert recon_mem.shape == ReconOrder(Y=12, X=2048, Z=2048)

		assert recon_mem.name == proj_mem.name
		assert recon_mem.name != sino_mem.name

		with sino_mem as si, proj_mem as pj, recon_mem as rc:
			si[0, 0, 0] = 6
			pj[0, 0, 0] = 5
			assert rc[0, 0, 0] == pj[0, 0, 0]
			assert pj[0, 0, 0] != si[0, 0, 0]


def test_mem():
	snm = SharedNPManager()
	snm.start()

	snp = snm.SharedNPArray(SinoOrder(Y=24, Theta=1501, X=2048), dtype=np.uint16)

	snpv = snp[1, :, :]

	snpvv = snpv[5:10, :]

	assert snp.shape == SinoOrder(Y=24, Theta=1501, X=2048)
	assert astuple(snpv.shape) == (1501, 2048)
	assert astuple(snpvv.shape) == (5, 2048)
	assert snp.size() == 147554304
	assert snpv.size() == 6148096
	assert snpvv.size() == 20480
	assert snp.dtype == snpv.dtype == snpvv.dtype == np.dtype(np.uint16)
	assert snp.name == snpv.name == snpvv.name

	print(f"Shapes: {snp.shape} | {snpv.shape} | {snpvv.shape}")

	print(f"Sizes: {getsizeof(snm)} | {getsizeof(snpv)} | {getsizeof(snpvv)}")

	with snpv as var, snp as ar, snpvv as vvar:
		print(f"MSM: {np.shares_memory(var, ar[1, :, :])}")

		assert np.all(var[:, :] == ar[1, :, :])

		print(f"{type(ar)} - {ar.shape} | {type(var)} - {var.shape} | {type(vvar)} - {var.shape}")

		ar[1, 5, 1] = 80

		print(vvar[:, 1])

		assert vvar[0, 1] == 80

		var[0:10, :] = 77

		assert all(ar[1, 0:10, 5] == 77)

		print(ar[1, 0:10, 0:5])

	snm.shutdown()
