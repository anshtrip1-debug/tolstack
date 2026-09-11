#!/usr/bin/env python3
"""
tolstack - Monte Carlo tolerance stack-up analysis.

Worst-case stack-up assumes every part in the assembly arrives at the wrong
end of its tolerance band at the same time. That basically never happens, so
designers who use it end up buying tolerances they did not need. RSS assumes
everything is normal, centered, and independent, which is closer but still
wrong the moment a process runs off-center or a dimension is uniform.

This tool runs all three so you can see the gap:

    worst case   arithmetic sum of the tolerance bands
    RSS          root sum square of the bands
    Monte Carlo  sample each dimension from its real distribution

Only the standard library is used, so it runs anywhere Python 3.8 does.

Usage
-----
    python3 tolstack.py examples/shaft-housing.json
    python3 tolstack.py examples/shaft-housing.json -n 500000 --seed 7
    python3 tolstack.py --selftest
"""

import argparse
import json
import math
import random
import statistics
import sys

# --------------------------------------------------------------------------
# distributions
# --------------------------------------------------------------------------
# Each dimension owns a sampler. "tol" is the half band, so a dimension of
# 40.0 +/- 0.05 has nominal 40.0 and tol 0.05.


def _normal(rng, nominal, tol, sigma_ratio, shift):
    """Normal centred at nominal+shift, with tol at sigma_ratio sigmas.

    sigma_ratio of 3 means the tolerance band is +/- 3 sigma, which is the
    usual assumption behind a Cp of 1.0.
    """
    sigma = tol / sigma_ratio
    return rng.gauss(nominal + shift, sigma)


def _uniform(rng, nominal, tol, sigma_ratio, shift):
    """Flat across the band. This is what a sorted or trimmed part looks like."""
    return rng.uniform(nominal + shift - tol, nominal + shift + tol)


def _triangular(rng, nominal, tol, sigma_ratio, shift):
    """Peaked at nominal, zero at the limits. A decent default when you have
    no process data but expect the operator to aim for the middle."""
    return rng.triangular(nominal + shift - tol, nominal + shift + tol,
                          nominal + shift)


SAMPLERS = {
    "normal": _normal,
    "uniform": _uniform,
    "triangular": _triangular,
}

# The standard deviation of each distribution over a band of +/- 1, used for
# the RSS estimate so that RSS stays honest when a dimension is not normal.
UNIT_SIGMA = {
    "normal": lambda sigma_ratio: 1.0 / sigma_ratio,
    "uniform": lambda sigma_ratio: 1.0 / math.sqrt(3.0),
    "triangular": lambda sigma_ratio: 1.0 / math.sqrt(6.0),
}


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------

class Dimension:
    def __init__(self, name, nominal, tol, direction=1, dist="normal",
                 sigma_ratio=3.0, shift=0.0):
        if dist not in SAMPLERS:
            raise ValueError("unknown distribution %r for %r" % (dist, name))
        if tol < 0:
            raise ValueError("tolerance must be positive on %r" % name)
        if direction not in (1, -1):
            raise ValueError("direction must be +1 or -1 on %r" % name)
        self.name = name
        self.nominal = float(nominal)
        self.tol = float(tol)
        self.direction = int(direction)
        self.dist = dist
        self.sigma_ratio = float(sigma_ratio)
        self.shift = float(shift)

    def sample(self, rng):
        return SAMPLERS[self.dist](rng, self.nominal, self.tol,
                                   self.sigma_ratio, self.shift)

    @property
    def sigma(self):
        return self.tol * UNIT_SIGMA[self.dist](self.sigma_ratio)

    @classmethod
    def from_dict(cls, d):
        return cls(
            name=d["name"],
            nominal=d["nominal"],
            tol=d["tol"],
            direction=d.get("direction", 1),
            dist=d.get("dist", "normal"),
            sigma_ratio=d.get("sigma_ratio", 3.0),
            shift=d.get("shift", 0.0),
        )


class Stack:
    """A chain of dimensions that sum, with signs, into one measured gap."""

    def __init__(self, name, dims, lower=None, upper=None, units="mm"):
        if not dims:
            raise ValueError("a stack needs at least one dimension")
        if lower is None and upper is None:
            raise ValueError("give at least one spec limit")
        if lower is not None and upper is not None and lower >= upper:
            raise ValueError("lower spec limit must sit below the upper")
        self.name = name
        self.dims = dims
        self.lower = lower
        self.upper = upper
        self.units = units

    @property
    def nominal(self):
        return sum(d.direction * d.nominal for d in self.dims)

    @property
    def worst_case(self):
        band = sum(d.tol for d in self.dims)
        return self.nominal - band, self.nominal + band

    @property
    def rss(self):
        sigma = math.sqrt(sum(d.sigma ** 2 for d in self.dims))
        band = 3.0 * sigma
        return self.nominal - band, self.nominal + band

    def sample(self, rng):
        return sum(d.direction * d.sample(rng) for d in self.dims)

    @classmethod
    def from_dict(cls, d):
        return cls(
            name=d.get("name", "stack"),
            dims=[Dimension.from_dict(x) for x in d["dimensions"]],
            lower=d.get("lower"),
            upper=d.get("upper"),
            units=d.get("units", "mm"),
        )


# --------------------------------------------------------------------------
# analysis
# --------------------------------------------------------------------------

class Result:
    def __init__(self, samples, stack):
        self.stack = stack
        self.n = len(samples)
        self.mean = statistics.fmean(samples)
        self.sd = statistics.stdev(samples)
        ordered = sorted(samples)
        self.min = ordered[0]
        self.max = ordered[-1]
        self.p = {q: ordered[min(self.n - 1, int(q / 100.0 * self.n))]
                  for q in (0.135, 1, 50, 99, 99.865)}
        lo, up = stack.lower, stack.upper
        self.fail_low = sum(1 for s in samples if lo is not None and s < lo)
        self.fail_high = sum(1 for s in samples if up is not None and s > up)
        self.samples = ordered

    @property
    def ppm(self):
        return (self.fail_low + self.fail_high) / self.n * 1e6

    @property
    def yield_pct(self):
        return 100.0 * (1.0 - (self.fail_low + self.fail_high) / self.n)

    @property
    def cpk(self):
        """Process capability against the tighter of the two limits.

        Cpk above 1.33 is the usual "this will not bite you" threshold.
        """
        if self.sd == 0:
            return float("inf")
        parts = []
        if self.stack.upper is not None:
            parts.append((self.stack.upper - self.mean) / (3 * self.sd))
        if self.stack.lower is not None:
            parts.append((self.mean - self.stack.lower) / (3 * self.sd))
        return min(parts)

    def contributions(self, rng_seed):
        """Share of output variance owed to each dimension.

        For a linear stack the variances add, so each dimension's share is
        just its own variance over the total. That makes it obvious which
        single tolerance is worth spending money to tighten.
        """
        total = sum(d.sigma ** 2 for d in self.stack.dims)
        out = []
        for d in self.stack.dims:
            share = 0.0 if total == 0 else d.sigma ** 2 / total
            out.append((d, share))
        out.sort(key=lambda x: -x[1])
        return out


def run(stack, n, seed=None):
    rng = random.Random(seed)
    return Result([stack.sample(rng) for _ in range(n)], stack)


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------

def histogram(result, width=54, rows=16):
    lo, hi = result.min, result.max
    if hi == lo:
        return ["all samples identical at %.4f" % lo]
    edges = [lo + (hi - lo) * i / rows for i in range(rows + 1)]
    counts = [0] * rows
    for s in result.samples:
        i = int((s - lo) / (hi - lo) * rows)
        counts[min(i, rows - 1)] += 1
    peak = max(counts) or 1
    lines = []
    for i in range(rows):
        mid = (edges[i] + edges[i + 1]) / 2
        bar = "#" * int(round(counts[i] / peak * width))
        flag = " "
        if result.stack.lower is not None and edges[i + 1] <= result.stack.lower:
            flag = "<"
        if result.stack.upper is not None and edges[i] >= result.stack.upper:
            flag = ">"
        lines.append("%9.4f %s|%s" % (mid, flag, bar))
    return lines


def report(result, out=sys.stdout):
    s = result.stack
    u = s.units
    w = lambda t="": print(t, file=out)

    w("=" * 68)
    w("  %s" % s.name)
    w("=" * 68)
    w()
    w("  chain")
    for d in s.dims:
        sign = "+" if d.direction > 0 else "-"
        shift = "" if d.shift == 0 else "  shift %+.4f" % d.shift
        w("    %s %-22s %9.4f +/- %.4f  %-10s%s"
          % (sign, d.name, d.nominal, d.tol, d.dist, shift))
    w()

    lo_wc, hi_wc = s.worst_case
    lo_rss, hi_rss = s.rss
    w("  nominal gap      %.4f %s" % (s.nominal, u))
    w("  spec limits      %s to %s %s"
      % ("none" if s.lower is None else "%.4f" % s.lower,
         "none" if s.upper is None else "%.4f" % s.upper, u))
    w()
    w("  method        range                       band")
    w("  worst case    %8.4f to %8.4f      %.4f" % (lo_wc, hi_wc, hi_wc - lo_wc))
    w("  RSS (3 sd)    %8.4f to %8.4f      %.4f" % (lo_rss, hi_rss, hi_rss - lo_rss))
    w("  Monte Carlo   %8.4f to %8.4f      %.4f"
      % (result.p[0.135], result.p[99.865], result.p[99.865] - result.p[0.135]))
    w()
    w("  Monte Carlo, n = %d" % result.n)
    w("    mean            %.4f %s" % (result.mean, u))
    w("    sd              %.4f %s" % (result.sd, u))
    w("    observed range  %.4f to %.4f" % (result.min, result.max))
    w("    below LSL       %d" % result.fail_low)
    w("    above USL       %d" % result.fail_high)
    w("    defects         %.0f ppm" % result.ppm)
    w("    yield           %.4f %%" % result.yield_pct)
    w("    Cpk             %.3f%s"
      % (result.cpk, "" if result.cpk >= 1.33 else "   (below 1.33)"))
    w()
    w("  variance contribution")
    for d, share in result.contributions(None):
        bar = "#" * int(round(share * 34))
        w("    %-24s %5.1f%%  %s" % (d.name, share * 100, bar))
    w()
    w("  distribution")
    for line in histogram(result):
        w("  " + line)
    w()

    if result.ppm == 0:
        w("  No sampled assembly fell outside the limits. If worst case failed")
        w("  but Monte Carlo did not, the worst-case number was costing you")
        w("  tolerance you did not have to buy.")
    else:
        worst = result.contributions(None)[0][0]
        w("  Tightening %r is the cheapest way to cut the defect rate: it owns"
          % worst.name)
        w("  the largest share of the output variance.")
    w("=" * 68)


# --------------------------------------------------------------------------
# self test
# --------------------------------------------------------------------------

def selftest():
    ok = True

    def check(label, got, want, tol):
        nonlocal ok
        good = abs(got - want) <= tol
        ok = ok and good
        print("  [%s] %-46s got %.5f want %.5f"
              % ("pass" if good else "FAIL", label, got, want))

    # Two dimensions that subtract: 40 +/- 0.05 minus 39.9 +/- 0.03.
    stack = Stack("selftest", [
        Dimension("housing bore", 40.0, 0.05, direction=1),
        Dimension("shaft od", 39.9, 0.03, direction=-1),
    ], lower=0.02, upper=0.18)

    check("nominal is the signed sum", stack.nominal, 0.1, 1e-12)
    check("worst case band is the arithmetic sum",
          stack.worst_case[1] - stack.worst_case[0], 0.16, 1e-12)

    # RSS band should be 3 * sqrt(sum of sigma^2), sigma = tol/3.
    want_rss = 6.0 * math.sqrt((0.05 / 3) ** 2 + (0.03 / 3) ** 2)
    check("RSS band matches root sum square",
          stack.rss[1] - stack.rss[0], want_rss, 1e-12)

    r = run(stack, 200000, seed=1)
    check("Monte Carlo mean lands on nominal", r.mean, 0.1, 2e-4)
    check("Monte Carlo sd matches RSS sigma", r.sd, want_rss / 6.0, 5e-5)

    # A uniform dimension has sigma = tol/sqrt(3), wider than the normal case.
    uni = Stack("uniform", [Dimension("slot", 10.0, 0.1, dist="uniform")],
                lower=9.8, upper=10.2)
    ru = run(uni, 200000, seed=2)
    check("uniform sd is tol/sqrt(3)", ru.sd, 0.1 / math.sqrt(3), 1e-3)

    # A mean shift should move Cpk down even though sd is unchanged.
    centered = Stack("centered", [Dimension("a", 10.0, 0.1)],
                     lower=9.85, upper=10.15)
    shifted = Stack("shifted", [Dimension("a", 10.0, 0.1, shift=0.05)],
                    lower=9.85, upper=10.15)
    c1 = run(centered, 100000, seed=3).cpk
    c2 = run(shifted, 100000, seed=3).cpk
    print("  [%s] %-46s %.3f then %.3f"
          % ("pass" if c2 < c1 else "FAIL",
             "mean shift lowers Cpk", c1, c2))
    ok = ok and c2 < c1

    # Reproducibility.
    a = run(stack, 5000, seed=42).mean
    b = run(stack, 5000, seed=42).mean
    print("  [%s] %-46s" % ("pass" if a == b else "FAIL",
                            "same seed gives the same answer"))
    ok = ok and a == b

    print("\n  %s" % ("all checks passed" if ok else "SOME CHECKS FAILED"))
    return 0 if ok else 1


# --------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Monte Carlo tolerance stack-up analysis.")
    ap.add_argument("stackfile", nargs="?", help="JSON stack definition")
    ap.add_argument("-n", "--samples", type=int, default=100000)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--csv", help="write raw samples to this path")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)

    if args.selftest:
        return selftest()
    if not args.stackfile:
        ap.error("give a stack file, or use --selftest")

    with open(args.stackfile) as f:
        stack = Stack.from_dict(json.load(f))

    result = run(stack, args.samples, args.seed)
    report(result)

    if args.csv:
        with open(args.csv, "w") as f:
            f.write("gap\n")
            for s in result.samples:
                f.write("%.6f\n" % s)
        print("  wrote %d samples to %s" % (result.n, args.csv))
    return 0


if __name__ == "__main__":
    sys.exit(main())
