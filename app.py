from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from pyproj import Transformer


# =========================
# CONFIG
# =========================
# Metti il file Excel nella repo (es: ./data/spettri2008.xls)
XLS_PATH = "./data/spettri2008.xls"

LAT_COL = "LAT"
LON_COL = "LON"

# Distanze in metri (Italia): WGS84 -> UTM32N
PROJ_METERS = ("EPSG:4326", "EPSG:32632")

# Interpolazione spaziale
K_NEIGHBORS = 4
IDW_POWER = 2.0

# Unità ag nel file: FISSO come richiesto
AG_UNIT = "ms2"  # m/s^2 => ag[g] = ag[m/s^2] / 9.81


# =========================
# TR disponibili + colonne (0-based pandas)
# =========================
TR_AVAILABLE = [30, 50, 72, 101, 140, 201, 475, 975, 2475]

TR_TO_COLS_0B: Dict[int, Tuple[int, int, int]] = {
    30:  (5 - 1, 6 - 1, 7 - 1),
    50:  (8 - 1, 9 - 1, 10 - 1),
    72:  (11 - 1, 12 - 1, 13 - 1),
    101: (14 - 1, 15 - 1, 16 - 1),
    140: (17 - 1, 18 - 1, 19 - 1),
    201: (20 - 1, 21 - 1, 22 - 1),
    475: (23 - 1, 24 - 1, 25 - 1),
    975: (26 - 1, 27 - 1, 28 - 1),
    2475:(29 - 1, 30 - 1, 31 - 1),
}

# Normativa: CU e PVR
CU_BY_CLASS = {"I": 0.7, "II": 1.0, "III": 1.5, "IV": 2.0}
PVR_BY_STATE = {"SLO": 0.81, "SLD": 0.63, "SLV": 0.10, "SLC": 0.05}
STATE_ORDER = ["SLO", "SLD", "SLV", "SLC"]


# =========================
# Models (input)
# =========================
class SeismicRequestItem(BaseModel):
    lat: float = Field(..., description="Latitude WGS84")
    lon: float = Field(..., description="Longitude WGS84")
    vn: float = Field(..., gt=0, description="Vita nominale [anni]")
    class_: str = Field(..., alias="class", description="Classe d'uso: I, II, III, IV")

    class Config:
        populate_by_name = True


# =========================
# Data structures
# =========================
@dataclass(frozen=True)
class NeighborPoint:
    idx: int
    lat: float
    lon: float
    dist_m: float


# =========================
# Helpers
# =========================
def ag_to_g(ag_raw: float) -> float:
    if AG_UNIT.lower() == "ms2":
        return float(ag_raw) / 9.81
    raise ValueError("AG_UNIT fissato a 'ms2' in questo servizio.")


def compute_VR(vn: float, cu: float) -> float:
    return float(vn) * float(cu)


def compute_TR(vr: float, pvr: float) -> float:
    return -float(vr) / math.log(1.0 - float(pvr))


def idw(values: np.ndarray, distances: np.ndarray, p: float = 2.0) -> Tuple[float, np.ndarray]:
    eps = 1e-9
    distances = np.asarray(distances, dtype=float)
    values = np.asarray(values, dtype=float)

    w = 1.0 / np.power(distances + eps, p)
    w_norm = w / np.sum(w)
    v = float(np.sum(w_norm * values))
    return v, w_norm


def linear_interp(x: float, x0: float, x1: float, y0: float, y1: float) -> float:
    if abs(x1 - x0) < 1e-12:
        return float(y0)
    return float(y0 + (y1 - y0) * (x - x0) / (x1 - x0))


def bracket_TR(tr: float) -> Tuple[int, int]:
    tr = float(tr)
    if tr <= TR_AVAILABLE[0]:
        return TR_AVAILABLE[0], TR_AVAILABLE[0]
    if tr >= TR_AVAILABLE[-1]:
        return TR_AVAILABLE[-1], TR_AVAILABLE[-1]
    for i in range(len(TR_AVAILABLE) - 1):
        a = TR_AVAILABLE[i]
        b = TR_AVAILABLE[i + 1]
        if a <= tr <= b:
            return a, b
    return TR_AVAILABLE[-1], TR_AVAILABLE[-1]


# =========================
# Engine (loads Excel once)
# =========================
class SeismicEngine:
    def __init__(self, xls_path: str):
        self.df = pd.read_excel(xls_path)

        if LAT_COL not in self.df.columns or LON_COL not in self.df.columns:
            raise ValueError(f"Non trovo colonne {LAT_COL}/{LON_COL}. Colonne: {list(self.df.columns)}")

        self.proj = Transformer.from_crs(PROJ_METERS[0], PROJ_METERS[1], always_xy=True)

        lats = pd.to_numeric(self.df[LAT_COL], errors="coerce").to_numpy()
        lons = pd.to_numeric(self.df[LON_COL], errors="coerce").to_numpy()
        valid = np.isfinite(lats) & np.isfinite(lons)
        if not np.all(valid):
            self.df = self.df.loc[valid].reset_index(drop=True)
            lats = lats[valid]
            lons = lons[valid]

        x_nodes, y_nodes = self.proj.transform(lons, lats)
        self.x_nodes = np.asarray(x_nodes, dtype=float)
        self.y_nodes = np.asarray(y_nodes, dtype=float)

    def neighbors_4(self, lat: float, lon: float, k: int = 4) -> List[NeighborPoint]:
        x0, y0 = self.proj.transform(lon, lat)
        d = np.hypot(self.x_nodes - x0, self.y_nodes - y0)

        idx = np.argpartition(d, k)[:k]
        idx = idx[np.argsort(d[idx])]

        out: List[NeighborPoint] = []
        for i in idx:
            out.append(
                NeighborPoint(
                    idx=int(i),
                    lat=float(self.df.loc[int(i), LAT_COL]),
                    lon=float(self.df.loc[int(i), LON_COL]),
                    dist_m=float(d[int(i)]),
                )
            )
        return out

    def _spatial_at_TR_discrete(self, neighbors: List[NeighborPoint], tr: int) -> Dict[str, Any]:
        c_ag, c_f0, c_tc = TR_TO_COLS_0B[tr]
        dists = np.array([n.dist_m for n in neighbors], dtype=float)

        ag_raw = np.array([float(self.df.iat[n.idx, c_ag]) for n in neighbors], dtype=float)
        f0_vals = np.array([float(self.df.iat[n.idx, c_f0]) for n in neighbors], dtype=float)
        tc_vals = np.array([float(self.df.iat[n.idx, c_tc]) for n in neighbors], dtype=float)

        ag_raw_i, w = idw(ag_raw, dists, p=IDW_POWER)
        f0_i, _ = idw(f0_vals, dists, p=IDW_POWER)
        tc_i, _ = idw(tc_vals, dists, p=IDW_POWER)

        return {
            "TR": tr,
            "ag_g": ag_to_g(ag_raw_i),
            "F0": f0_i,
            "Tc_star": tc_i,
            "weights": w.tolist(),
        }

    def params_at_TR(self, neighbors: List[NeighborPoint], tr_target: float) -> Dict[str, Any]:
        tr_target = float(tr_target)

        # exact TR
        if any(abs(tr_target - t) < 1e-9 for t in TR_AVAILABLE):
            tr_int = int(round(tr_target))
            one = self._spatial_at_TR_discrete(neighbors, tr_int)
            return {
                "TR_target": tr_target,
                "TR_low": tr_int,
                "TR_high": tr_int,
                "alpha": 0.0,
                "ag_g": one["ag_g"],
                "F0": one["F0"],
                "Tc_star": one["Tc_star"],
            }

        # interpolate between bracketing TRs
        tr_low, tr_high = bracket_TR(tr_target)
        low = self._spatial_at_TR_discrete(neighbors, tr_low)
        high = self._spatial_at_TR_discrete(neighbors, tr_high)

        if tr_low == tr_high:
            return {
                "TR_target": tr_target,
                "TR_low": tr_low,
                "TR_high": tr_high,
                "alpha": 0.0,
                "ag_g": low["ag_g"],
                "F0": low["F0"],
                "Tc_star": low["Tc_star"],
            }

        alpha = (tr_target - tr_low) / (tr_high - tr_low)

        return {
            "TR_target": tr_target,
            "TR_low": tr_low,
            "TR_high": tr_high,
            "alpha": float(alpha),
            "ag_g": linear_interp(tr_target, tr_low, tr_high, low["ag_g"], high["ag_g"]),
            "F0": linear_interp(tr_target, tr_low, tr_high, low["F0"], high["F0"]),
            "Tc_star": linear_interp(tr_target, tr_low, tr_high, low["Tc_star"], high["Tc_star"]),
        }


# Load engine at startup (so Excel is read once)
try:
    ENGINE = SeismicEngine(XLS_PATH)
except Exception as e:
    ENGINE = None
    STARTUP_ERROR = str(e)
else:
    STARTUP_ERROR = None


# =========================
# FastAPI app
# =========================
app = FastAPI(title="Seismic Params API", version="1.0.0")


@app.get("/health")
def health():
    if ENGINE is None:
        return {"ok": False, "error": STARTUP_ERROR}
    return {"ok": True}


def _print_table(states_sorted: List[Dict[str, Any]]) -> None:
    # stampa tabella nei log di Render
    print("\nStato Limite\tTR [anni]\tag [g]\tFo\tTc* [s]")
    for s in states_sorted:
        print(
            f"{s['state']}\t"
            f"{int(round(s['TR']))}\t"
            f"{s['ag_g']:.3f}\t"
            f"{s['F0']:.3f}\t"
            f"{s['Tc_star']:.3f}"
        )


@app.post("/seismic")
def seismic(items: List[SeismicRequestItem]):
    if ENGINE is None:
        raise HTTPException(status_code=500, detail=f"Startup error: {STARTUP_ERROR}")

    out_all: List[Dict[str, Any]] = []

    for it in items:
        lat = float(it.lat)
        lon = float(it.lon)
        vn = float(it.vn)
        cls = str(it.class_).strip().upper()

        if cls not in CU_BY_CLASS:
            raise HTTPException(status_code=400, detail=f"Invalid class '{cls}'. Use I, II, III, IV")

        cu = CU_BY_CLASS[cls]
        vr = compute_VR(vn, cu)

        neighbors = ENGINE.neighbors_4(lat, lon, k=K_NEIGHBORS)

        states = []
        for state, pvr in PVR_BY_STATE.items():
            tr_calc = compute_TR(vr, pvr)
            params = ENGINE.params_at_TR(neighbors, tr_calc)
            states.append({
                "state": state,
                "PVR": pvr,
                "VN": vn,
                "CU": cu,
                "VR": vr,
                "TR": tr_calc,
                "TR_low": params["TR_low"],
                "TR_high": params["TR_high"],
                "alpha": params["alpha"],
                "ag_g": params["ag_g"],
                "F0": params["F0"],
                "Tc_star": params["Tc_star"],
            })

        states_sorted = sorted(states, key=lambda s: STATE_ORDER.index(s["state"]))
        _print_table(states_sorted)

        out_all.append({
            "lat": lat,
            "lon": lon,
            "vn": vn,
            "class": cls,
            "cu": cu,
            "vr": vr,
            "nearest_points": [asdict(n) for n in neighbors],
            "states": states_sorted,
            "meta": {
                "AG_UNIT": AG_UNIT,
                "TR_available": TR_AVAILABLE,
                "k_neighbors": K_NEIGHBORS,
                "idw_power": IDW_POWER,
                "projection_meters": PROJ_METERS[1],
            },
        })

    return out_all