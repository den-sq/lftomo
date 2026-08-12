import benchmarks.vo_family as vo_family


def test_b10_shaped_plans_report_rows_proportional_exact_bytes():
	window = vo_family.run("reusable", "chain", (1969, 12, 2560), plan_only=True)
	whole = vo_family.run("reusable", "chain", (1969, 2160, 2560), plan_only=True)

	assert window.managed_scratch_bytes == 2 * 1969 * 12 * 2560 * 4
	assert whole.managed_scratch_bytes == 2 * 1969 * 2160 * 2560 * 4
	assert whole.managed_scratch_bytes == 180 * window.managed_scratch_bytes


def test_legacy_and_reusable_benchmark_arms_are_numerically_equivalent():
	legacy = vo_family.run("legacy", "chain", (16, 2, 24), workers=1)
	reusable = vo_family.run("reusable", "chain", (16, 2, 24), workers=1)

	assert legacy.managed_scratch_bytes == 0
	assert reusable.managed_scratch_bytes == 2 * 16 * 2 * 24 * 4
	assert legacy.fingerprint == reusable.fingerprint
	assert legacy.private_bytes_peak is not None
	assert reusable.private_bytes_peak is not None
