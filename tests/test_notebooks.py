import ast
import json
from pathlib import Path


NOTEBOOKS = tuple(Path("notebook").glob("*.ipynb"))
LEGACY_NAMES = (
	"DataWindowSet",
	"SharedNP(",
	"distribute_read",
	"DataWindow",
	"ImgShape",
	"ThetaSlice",
)


def test_notebooks_use_current_api_and_contain_valid_python():
	assert NOTEBOOKS
	for path in NOTEBOOKS:
		notebook = json.loads(path.read_text(encoding="utf-8"))
		code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
		source = "\n".join("".join(cell["source"]) for cell in code_cells)

		assert not any(name in source for name in LEGACY_NAMES), path
		ast.parse(source, filename=str(path))
		assert all(cell["execution_count"] is not None for cell in code_cells), path
		assert not any(
			output.get("output_type") == "error"
			for cell in code_cells
			for output in cell.get("outputs", [])
		), path
