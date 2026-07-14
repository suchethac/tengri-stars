# Five-band forecast campaign — band-combination sweep

Mock sweep: Teff [4000, 4500, 5000, 5500] × [Fe/H] [-3.0, -2.0, -1.0] × {RGB log g 2.0, MS log g 4.6} × 3 seeds; μ free (no distance); σ = (u 0.05, CaHK 0.03, gri 0.02); P(correct) = posterior mass on the true side of log g = 3.3. Same noise realizations for every config.

| config | bands | median σ([Fe/H]) [dex] | P(correct RGB) | P(correct MS) |
|---|---|---|---|---|
| all5 | u+CaHK+g+r+i | 0.392 | 0.63 | 0.50 |
| no-u | CaHK+g+r+i | 0.494 | 0.59 | 0.46 |
| no-cahk | u+g+r+i | 0.643 | 0.63 | 0.42 |
| gri | g+r+i | 1.384 | 0.60 | 0.43 |
| magic3 | CaHK+g+i | 0.526 | 0.58 | 0.47 |
| u-magic3 | u+CaHK+g+i | 0.421 | 0.65 | 0.49 |
| ugi | u+g+i | 0.697 | 0.64 | 0.42 |

Δ vs all5 ceiling:

- **no-u**: σ([Fe/H]) +0.102 dex, P(RGB) -0.04, P(MS) -0.04
- **no-cahk**: σ([Fe/H]) +0.252 dex, P(RGB) +0.00, P(MS) -0.08
- **gri**: σ([Fe/H]) +0.992 dex, P(RGB) -0.03, P(MS) -0.07
- **magic3**: σ([Fe/H]) +0.134 dex, P(RGB) -0.05, P(MS) -0.03
- **u-magic3**: σ([Fe/H]) +0.030 dex, P(RGB) +0.02, P(MS) -0.01
- **ugi**: σ([Fe/H]) +0.305 dex, P(RGB) +0.01, P(MS) -0.08
