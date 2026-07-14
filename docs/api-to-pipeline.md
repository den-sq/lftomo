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
sinograms by Y window, consumes no metadata or artifacts, produces no artifacts,
and declares a 2× memory hint for the wavelet/FFT coefficients. Its parameters
remain `level`, `wname`, `sigma`, and `zero_pad`. This is the Münch et al.
Fourier-wavelet method and remains separate from every Vo option.
