from pathlib import Path


CATALOG = Path(__file__).parents[1] / "docs" / "recon_minimal_tomopy_fix_catalog.md"


def test_tomopy_fix_catalog_covers_all_local_algorithm_decisions():
	text = CATALOG.read_text(encoding="utf-8")
	for required in (
		"phase_retrieval",
		"remove_sort_vo",
		"remove_large_vo",
		"remove_dead_vo",
		"recon.gridrec",
		"recon.*_cuda",
		"center.vo",
		"center.pc",
		"phase.paganin",
	):
		assert required in text
	assert "todo: upstream" not in text.lower()
