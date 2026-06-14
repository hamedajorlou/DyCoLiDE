"""
Overnight pipeline producing THREE saved figures: ER2, ER3, ER4.

- ER2: waits for the currently-running sigma=1..10 d=40 job to finish, then
  snapshots its result/figure as svar_sigma5_er2.{json,png,_table.txt}.
- ER3, ER4: runs the full experiment at 5 seeds for each density -- d-sweep at
  sigma=5 (panels a,b) + sigma-sweep 1..10 at d=40 (panel c, sigma=5 reused from
  the d-sweep) -- saving svar_sigma5_er{3,4}.{json,png,_table.txt}.

Every cell is saved incrementally and wrapped in try/except, so a slow/failed
draw (possible at ER4, large d) never kills the night; partial results are
recoverable and re-plottable.
"""

import os
import sys
import json
import time
import shutil
import importlib.util

_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, '..')))
sys.path.insert(0, os.path.join(os.path.abspath(os.path.join(_HERE, '..')), 'SVAR'))
sys.path.insert(0, _HERE)

spec = importlib.util.spec_from_file_location("s5", os.path.join(_HERE, "svar_sigma5.py"))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)                       # SIG_LIST=[1..10], SIG_D=40, SIGMA_MAIN=5

SEEDS5 = [42, 43, 44, 45, 46]
OUT = m.JSON_PATH                                # svar_sigma5.json (ER2, 10 seeds)
SHARED_TXT = os.path.join(_HERE, 'svar_sigma5_table.txt')
NAN5 = (float('nan'),) * 5


def paths(tag):
    return (os.path.join(_HERE, f"svar_sigma5_{tag}.json"),
            os.path.join(_HERE, f"svar_sigma5_{tag}.png"),
            os.path.join(_HERE, f"svar_sigma5_{tag}_table.txt"))


def cell(d, seed, sigma, dens):
    try:
        X, Wt, At = m.gen_gauss(d, m.N, seed, float(sigma), density=dens)
        return m.fit_methods(X, Wt, At, seed)
    except Exception as e:
        print(f"    cell d={d} seed={seed} sig={sigma} ER{dens} FAILED: {e}", flush=True)
        return {meth: NAN5 for meth in m.METHODS}


def run_density(dens, seeds, jp, fp, tp):
    res = {'d': {meth: {str(d): [] for d in m.D_LIST} for meth in m.METHODS},
           'sig': {meth: {str(s): [] for s in m.SIG_LIST} for meth in m.METHODS}}

    def save():
        json.dump({'D_LIST': m.D_LIST, 'SIG_LIST': m.SIG_LIST, 'SEEDS': seeds,
                   'SIGMA_MAIN': m.SIGMA_MAIN, 'SIG_D': m.SIG_D, 'DENSITY': dens,
                   'results': res}, open(jp, 'w'), indent=2)

    print(f"=== ER{dens} d-sweep (sigma={m.SIGMA_MAIN}), {len(seeds)} seeds ===", flush=True)
    for d in m.D_LIST:
        for seed in seeds:
            o = cell(d, seed, m.SIGMA_MAIN, dens)
            for meth in m.METHODS:
                res['d'][meth][str(d)].append(o[meth])
            save()
        print(f"  ER{dens} d={d} done", flush=True)

    print(f"=== ER{dens} sigma-sweep (d={m.SIG_D}, sigma=5 reused) ===", flush=True)
    for meth in m.METHODS:
        res['sig'][meth]['5'] = list(res['d'][meth][str(m.SIG_D)])
    for s in [x for x in m.SIG_LIST if x != 5]:
        for seed in seeds:
            o = cell(m.SIG_D, seed, s, dens)
            for meth in m.METHODS:
                res['sig'][meth][str(s)].append(o[meth])
            save()
        print(f"  ER{dens} sigma={s} done", flush=True)

    m.DENSITY = dens
    m.SEEDS = seeds
    m.FIG_PATH = fp
    m.plot(res)
    m.table(res)
    shutil.copy(SHARED_TXT, tp)
    print(f"ER{dens} COMPLETE -> {fp}", flush=True)


# ---- 1. wait for the running ER2 sigma=1..10 job, then snapshot it ----
print("waiting for ER2 sigma=10 to complete...", flush=True)
while True:
    try:
        rb = json.load(open(OUT))['results']
        if len(rb['sig']['DyCoLiDE'].get('10', [])) >= 10:
            break
    except Exception:
        pass
    time.sleep(30)
time.sleep(25)
e2j, e2p, e2t = paths('er2')
shutil.copy(OUT, e2j)
m.DENSITY = 2
m.SEEDS = list(range(42, 52))
m.FIG_PATH = e2p
_er2 = json.load(open(OUT))['results']
m.plot(_er2)
m.table(_er2)
shutil.copy(SHARED_TXT, e2t)
print(f"ER2 saved -> {e2p}", flush=True)

# ---- 2. ER3 (5 seeds) ----
run_density(3, SEEDS5, *paths('er3'))

# ---- 3. ER4 (5 seeds) ----
run_density(4, SEEDS5, *paths('er4'))

print("ALL DONE: ER2, ER3, ER4 figures saved.", flush=True)
