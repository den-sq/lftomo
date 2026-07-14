# Berkeley helical-remap port fidelity

`helical_correct.index_remap` is traced to the requested producer source:

- repository: [`den-sq/dan_scripts`](https://github.com/den-sq/dan_scripts)
- file: `berkeley_2021/allinwonderful_v1.5_A1_Helical.py`
- commit: `8d324b270ab6c569acde246a01ea4572924cbab0` (2022-05-24)
- source-file SHA-256:
  `4b63d6b3d57ce9065e888728aa4c127fe56fb3ad1ed59e0c8ea312a7289b19ea`
- relevant producer functions: `parallel_fill_array_helically` and
  `parallel_helical_write`

The producer builds a linear `slope_function`, rounds its row displacement for
each projection, and—for the upward scan—selects
`height - round(slope(projection)) - 1`. The Stage preserves that rounded
per-projection displacement. It operates on an already-loaded `(Y, Theta, X)`
window instead of opening and normalizing one TIFF row inside the remapper;
reading and normalization are separate pipeline Stages.

`going_up=True` selects the producer's decreasing-row convention.
`projection_start` preserves the absolute projection offset for windowed input.
The generalized Stage can either crop to rows valid for every projection or
retain the input height and fill out-of-bounds rows. A source-equation regression
test pins the Berkeley bottom-row selection directly.
