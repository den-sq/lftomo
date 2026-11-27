# API to Pipeline

Existing lftomo function APIs become registered Pipeline algorithms without
losing their mathematical identities.

## Vo stripe removal

| Function API | Registered algorithm | Contract |
| --- | --- | --- |
| `prep.stripe.remove_sort_vo` | `stripe.vo_sort` | Mutates `(Y, Theta, X)` sinograms by Y window; no metadata or artifacts; 3× memory hint for values and sort indices. |
| `prep.stripe.remove_dead_vo` | `stripe.vo_dead` | Mutates by Y window; no metadata or artifacts; 2× hint for interpolation scratch. Optional `norm` runs local large-stripe cleanup. |
| `prep.stripe.remove_large_vo` | `stripe.vo_large` | Mutates by Y window; no metadata or artifacts; 3× hint for sorted/smoothed values and indices. |

The final names deliberately keep all three Vo algorithms visible. They are not
implementations to benchmark against one another: sorting, dead-line
interpolation, and large-stripe correction solve different artifact classes.
The function and Stage paths share array kernels so fixes cannot drift between
the APIs.

## Fourier-wavelet stripe removal

`prep.stripe.remove_stripe_fw` maps to `stripe.fw`. It mutates `(Y, Theta, X)`
sinograms by Y window, consumes no metadata or artifacts, and produces no artifacts.
The Stage accepts `level`, `wname`, `sigma`, `zero_pad`, and `theta_mirror`.
This is the Münch et al.
Fourier-wavelet method and remains separate from every Vo option.

`theta_mirror` defaults to `false`, preserving the existing filter. When enabled,
each sinogram is followed by its Theta-reversed copy before filtering, then
cropped to its original extent. This removes the first/last-projection FFT seam
for column offsets that drift during a scan. Sigma doubles with the Theta
length so its damping band keeps the same physical frequency; the automatically
selected wavelet level still uses the original input dimensions. `zero_pad`
continues to select the wavelet boundary mode independently.

The mirror canvas and crop use `TomoWindow`. Managed pipeline execution requests
one row-indexed `fw_theta_mirror_work` buffer through the existing bounded Stage
scratch lifecycle, shared across FW stages in the segment and retired afterward.
Direct array-kernel calls allocate the mirror plane locally. PyWavelets
coefficients and reconstruction results and NumPy FFT outputs remain
external-library temporaries in either mode.
