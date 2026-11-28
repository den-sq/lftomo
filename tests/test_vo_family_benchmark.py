import pytest

import benchmarks.vo_family as vo_family


def test_b10_shaped_plans_report_rows_proportional_exact_bytes():
	window = vo_family.run("bottleneck", "chain", (1969, 12, 2560), plan_only=True)
	whole = vo_family.run("bottleneck", "chain", (1969, 2160, 2560), plan_only=True)

	assert window.median_pad_work_bytes == 1969 * 12 * (2560 + 50) * 4
	assert window.managed_scratch_bytes == (
		2 * 1969 * 12 * 2560 * 4 + 1969 * 12 * (2560 + 50) * 4
	)
	assert whole.managed_scratch_bytes == (
		2 * 1969 * 2160 * 2560 * 4 + 1969 * 2160 * (2560 + 50) * 4
	)
	assert whole.managed_scratch_bytes == 180 * window.managed_scratch_bytes


@pytest.mark.system
def test_scipy_and_bottleneck_dispatcher_arms_are_numerically_equivalent():
	scipy = vo_family.run("scipy", "chain", (16, 2, 24), workers=1)
	bottleneck = vo_family.run("bottleneck", "chain", (16, 2, 24), workers=1)

	assert scipy.managed_scratch_bytes == 2 * 16 * 2 * 24 * 4
	assert scipy.median_pad_work_bytes == 0
	assert bottleneck.managed_scratch_bytes == (
		2 * 16 * 2 * 24 * 4 + 16 * 2 * (24 + 50) * 4
	)
	assert bottleneck.median_pad_work_bytes == 16 * 2 * (24 + 50) * 4
	assert scipy.fingerprint == bottleneck.fingerprint
	assert scipy.private_bytes_peak is not None
	assert bottleneck.private_bytes_peak is not None
