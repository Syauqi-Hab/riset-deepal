# test_pso.py
# Basic Particle Swarm Optimization (PSO) + PyQGIS data hook
# Works in:
# - QGIS Python Console (paling mudah)
# - QGIS Processing Script (dengan sedikit penyesuaian)
# - Standalone PyQGIS (butuh init QGIS env)

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable, List, Tuple, Optional, Dict, Any

# ============================================================
# 1) BASIC PSO CORE
# ============================================================

@dataclass
class PSOConfig:
    n_particles: int = 30
    n_iters: int = 100
    w: float = 0.72          # inertia
    c1: float = 1.49         # cognitive
    c2: float = 1.49         # social
    seed: int = 42
    # clamp velocity to avoid explosion (optional)
    v_clamp: Optional[float] = None


@dataclass
class PSOResult:
    best_position: List[float]
    best_fitness: float
    history_best: List[float]


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def pso_optimize(
    objective: Callable[[List[float]], float],
    bounds: List[Tuple[float, float]],
    cfg: PSOConfig = PSOConfig(),
) -> PSOResult:
    """
    Minimize objective(x).
    bounds = [(min1,max1), (min2,max2), ...]
    """
    random.seed(cfg.seed)

    dim = len(bounds)

    # init particles
    pos = []
    vel = []
    pbest_pos = []
    pbest_fit = []

    for _ in range(cfg.n_particles):
        x = [random.uniform(bounds[d][0], bounds[d][1]) for d in range(dim)]
        v = [0.0 for _ in range(dim)]
        f = objective(x)

        pos.append(x)
        vel.append(v)
        pbest_pos.append(x[:])
        pbest_fit.append(f)

    # global best
    gbest_idx = min(range(cfg.n_particles), key=lambda i: pbest_fit[i])
    gbest_pos = pbest_pos[gbest_idx][:]
    gbest_fit = pbest_fit[gbest_idx]

    history_best = [gbest_fit]

    # loop
    for _it in range(cfg.n_iters):
        for i in range(cfg.n_particles):
            for d in range(dim):
                r1 = random.random()
                r2 = random.random()

                vel[i][d] = (
                    cfg.w * vel[i][d]
                    + cfg.c1 * r1 * (pbest_pos[i][d] - pos[i][d])
                    + cfg.c2 * r2 * (gbest_pos[d] - pos[i][d])
                )

                if cfg.v_clamp is not None:
                    vel[i][d] = _clamp(vel[i][d], -cfg.v_clamp, cfg.v_clamp)

                pos[i][d] = pos[i][d] + vel[i][d]
                pos[i][d] = _clamp(pos[i][d], bounds[d][0], bounds[d][1])

            f = objective(pos[i])

            if f < pbest_fit[i]:
                pbest_fit[i] = f
                pbest_pos[i] = pos[i][:]

                if f < gbest_fit:
                    gbest_fit = f
                    gbest_pos = pos[i][:]

        history_best.append(gbest_fit)

    return PSOResult(best_position=gbest_pos, best_fitness=gbest_fit, history_best=history_best)


# ============================================================
# 2) PYQGIS DATA LOADER (QGIS Console friendly)
# ============================================================

def load_points_from_active_layer(
    x_field: Optional[str] = None,
    y_field: Optional[str] = None,
    weight_field: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Load point data from active QGIS layer.

    - If layer is Point geometry: use geometry().asPoint()
    - If layer is not point OR you want specific fields: set x_field/y_field
    - weight_field optional (e.g., demand, intensity)

    Returns list of dict: [{"x":..., "y":..., "w":...}, ...]
    """
    # Import inside function so file still importable outside QGIS
    from qgis.utils import iface

    layer = iface.activeLayer()
    if layer is None:
        raise RuntimeError("No active layer. Please select a layer in QGIS.")

    feats = layer.getFeatures()
    out = []

    for f in feats:
        if x_field and y_field:
            x = float(f[x_field])
            y = float(f[y_field])
        else:
            geom = f.geometry()
            if geom is None or geom.isEmpty():
                continue
            if geom.type() != 0:  # 0 = point
                raise RuntimeError("Active layer is not a Point layer. Provide x_field/y_field or use a point layer.")
            pt = geom.asPoint()
            x, y = float(pt.x()), float(pt.y())

        w = float(f[weight_field]) if weight_field else 1.0
        out.append({"x": x, "y": y, "w": w})

    if not out:
        raise RuntimeError("No valid points loaded from layer.")
    return out


# ============================================================
# 3) EXAMPLE OBJECTIVE FOR SPATIAL OPTIMIZATION
#    (Minimize weighted distance to points)
# ============================================================

def make_weighted_distance_objective(points: List[Dict[str, Any]]) -> Callable[[List[float]], float]:
    """
    Decision variables:
      x[0] = X candidate
      x[1] = Y candidate

    Fitness:
      sum_i w_i * EuclideanDistance(candidate, point_i)
    """
    def objective(x: List[float]) -> float:
        cx, cy = x[0], x[1]
        s = 0.0
        for p in points:
            dx = cx - p["x"]
            dy = cy - p["y"]
            dist = math.sqrt(dx*dx + dy*dy)
            s += p["w"] * dist
        return s
    return objective


def bounds_from_points(points: List[Dict[str, Any]], padding_ratio: float = 0.05) -> List[Tuple[float, float]]:
    xs = [p["x"] for p in points]
    ys = [p["y"] for p in points]
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)

    padx = (maxx - minx) * padding_ratio if maxx > minx else 1.0
    pady = (maxy - miny) * padding_ratio if maxy > miny else 1.0

    return [(minx - padx, maxx + padx), (miny - pady, maxy + pady)]


# ============================================================
# 4) RUNNER (QGIS Console)
# ============================================================

def run_pso_qgis_example(
    x_field: Optional[str] = None,
    y_field: Optional[str] = None,
    weight_field: Optional[str] = None,
    n_particles: int = 40,
    n_iters: int = 150,
):
    """
    Jalankan di QGIS Python Console:
      from test_pso import run_pso_qgis_example
      run_pso_qgis_example(weight_field="demand")

    Output: print best position and fitness.
    """
    points = load_points_from_active_layer(x_field=x_field, y_field=y_field, weight_field=weight_field)
    obj = make_weighted_distance_objective(points)
    bnds = bounds_from_points(points)

    cfg = PSOConfig(
        n_particles=n_particles,
        n_iters=n_iters,
        w=0.72,
        c1=1.49,
        c2=1.49,
        seed=42,
        v_clamp=None,
    )

    res = pso_optimize(obj, bnds, cfg)

    print("=== PSO RESULT ===")
    print("Best (X,Y):", res.best_position)
    print("Best fitness:", res.best_fitness)
    print("History last:", res.history_best[-5:])

    return res


# If you want: run automatically when executed as a script (optional)
if __name__ == "__main__":
    # This block is mainly for non-QGIS testing.
    # In QGIS, call run_pso_qgis_example() from console.
    pass