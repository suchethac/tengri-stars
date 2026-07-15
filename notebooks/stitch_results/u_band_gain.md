# What adding LSST u buys the stitch (mock measurement, 2026-07-12)

Same mocks, seeds, and mixture policy as the 4-band campaign (`summary.md`);
bands u+CaHK+g+r+i with σ_u as marked, others (0.03, 0.02, 0.02, 0.02).
Entries: P(correct branch) / realized [Fe/H] 68% half-width [dex].

| parallax mode | 4-band (no u) | +u, σ_u = 0.05 | +u, σ_u = 0.02 |
|---|---|---|---|
| none    | 0.70 / 0.250 | 0.78 / 0.230 | 0.83 / 0.210 |
| p50     | 0.93 / 0.200 | 0.95 / 0.210 | 0.96 / 0.190 |
| p20     | 1.00 / 0.170 | 1.00 / 0.170 | 1.00 / 0.170 |
| knownDM | 1.00 / 0.090 | 1.00 / 0.050 | 1.00 / 0.040 |

Reading: u helps most where distance information is weakest (+8–13 points of
classification with no parallax), is redundant once the parallax is good, and
roughly doubles the [Fe/H] precision in known-distance (dSph/cluster) mode.
Mock-only — no real u photometry exists for these fields to validate against.
