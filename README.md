# tolstack

Monte Carlo tolerance stack-up analysis for mechanical assemblies. No
dependencies beyond the Python standard library.

## The problem

You have a chain of dimensions that add up to one gap that has to stay
inside a spec. There are three ways to predict that gap:

- **Worst case.** Sum the tolerance bands. Assumes every part in the
  assembly arrives at the wrong end of its band on the same day. Safe, and
  usually so pessimistic that you buy tolerances you never needed.
- **RSS.** Root sum square of the bands. Assumes everything is normal,
  centred and independent. Closer, but it lies the moment a process runs
  off-center or a dimension is uniform.
- **Monte Carlo.** Sample each dimension from the distribution it actually
  has. Slower, and right.

This tool runs all three side by side so the gap between them is visible.

## Use

```
python3 tolstack.py examples/shaft-housing.json
python3 tolstack.py examples/lever-linkage.json -n 500000 --seed 7
python3 tolstack.py examples/shaft-housing.json --csv samples.csv
python3 tolstack.py --selftest
```

## Defining a stack

```json
{
  "name": "Bearing seat clearance",
  "units": "mm",
  "lower": 0.020,
  "upper": 0.180,
  "dimensions": [
    { "name": "housing bore", "nominal": 40.000, "tol": 0.050,
      "direction": 1, "dist": "normal", "sigma_ratio": 3.0 },
    { "name": "bearing od",   "nominal": 39.960, "tol": 0.012,
      "direction": -1, "dist": "normal", "sigma_ratio": 4.0 }
  ]
}
```

| field | meaning |
| --- | --- |
| `nominal` | centre of the dimension |
| `tol` | half band, so `40.0 +/- 0.05` is nominal 40.0 and tol 0.05 |
| `direction` | `1` if the dimension opens the gap, `-1` if it closes it |
| `dist` | `normal`, `uniform` or `triangular` |
| `sigma_ratio` | how many sigma the band covers. 3 gives Cp = 1.0, 4 gives a tighter process |
| `shift` | mean offset, for a process you know runs off-centre |

## What it reports

Mean, standard deviation, defects in ppm, yield, Cpk, an ASCII histogram
with the spec limits flagged, and a ranked variance contribution table.

That last table is the useful one. Variances add on a linear stack, so each
dimension's share of the output variance tells you exactly which single
tolerance is worth money to tighten, and which ones you are wasting effort
on.

## Sample output

Running the bearing example: worst case spans 0.0030 to 0.1970 against a
spec of 0.020 to 0.180, so worst case says the design fails. Monte Carlo
over 200,000 assemblies gives 125 ppm defective and a Cpk of 1.17. The
design is marginal rather than broken, and the housing bore owns 61 percent
of the variance, so that is the one dimension to tighten.

## Tests

`--selftest` checks the nominal sum, the worst-case band, the RSS band
against a hand calculation, the sampled mean and standard deviation against
theory, the uniform standard deviation of `tol/sqrt(3)`, that a mean shift
lowers Cpk, and that a fixed seed reproduces exactly.
