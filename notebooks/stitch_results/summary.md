# MAGIC stitch campaign — policy × parallax-quality mock matrix

Bands CaHK,g,r,i; σ = (np.float64(0.03), np.float64(0.02), np.float64(0.02), np.float64(0.02)); isochrone mocks at [Fe/H] {-2.5,-2,-1.5,-1}, DM = 16.5; 12 positions × 2 branches × 60 noise draws. 'coverage' = fraction of truth inside the 68% interval (mixture only; the hard policies have no interval).

| config | P(correct) | feh bias | feh 68% half-width | 68% coverage |
|---|---|---|---|---|
| magic-hard_none | 0.50 | +0.000 | 0.330 |  |
| magic-hard_p50 | 0.91 | +0.000 | 0.200 |  |
| magic-hard_p20 | 0.99 | +0.000 | 0.190 |  |
| magic-hard_knownDM | 1.00 | +0.000 | 0.040 |  |
| hard-vote_none | 0.69 | +0.000 | 0.250 |  |
| hard-vote_p50 | 0.69 | +0.000 | 0.250 |  |
| hard-vote_p20 | 0.69 | +0.000 | 0.250 |  |
| hard-vote_knownDM | 1.00 | +0.000 | 0.040 |  |
| mixture_none | 0.70 | +0.000 | 0.250 | 0.40 (0.67 interior) |
| mixture_p50 | 0.93 | +0.000 | 0.200 | 0.41 (0.65 interior) |
| mixture_p20 | 1.00 | +0.000 | 0.170 | 0.43 (0.65 interior) |
| mixture_knownDM | 1.00 | +0.000 | 0.050 | 0.67 (0.73 interior) |
