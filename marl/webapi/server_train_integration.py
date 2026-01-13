import json
import os
import re
import threading
from datetime import datetime
from typing import Optional, Dict, Any

import wandb
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from marl.cooperative.algorithms.train_mappo_multiwalker import train_mappo_gold
from marl.webapi.train_metrics_recorder import TrainMetricsRecorder

# ==========================================================
# 🌐 PATHS & CONFIG
# ==========================================================
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

TRAJ_ROOT = os.path.join(BASE_DIR, "cooperative", "trajectories")
ARCHIVE_DIR = os.path.join(TRAJ_ROOT, "iter_archive")
LATEST_TRAJ = os.path.join(TRAJ_ROOT, "latest_trajectory.json")

os.makedirs(ARCHIVE_DIR, exist_ok=True)

# ==========================================================
# 🚀 FASTAPI APP
# ==========================================================
app = FastAPI(title="MARL Training Backend", version="0.5.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================================
# 🧠 TRAINING STATE (THREAD-SAFE)
# ==========================================================
TRAIN_THREAD: Optional[threading.Thread] = None
TRAIN_STOP_EVENT = threading.Event()
_STATUS_LOCK = threading.Lock()

METRICS_RECORDER = TrainMetricsRecorder()


def default_train_status() -> Dict[str, Any]:
    return {
        "state": "idle",          # idle | running | stopping | error | finished
        "algo": None,
        "env": None,

        "iter": 0,
        "max_iter": None,
        "progress": 0.0,
        "message": "Ready",

        "multiwalker": {
            "health": {
                "alive_agents": 0,
                "fallen_agents": 0,
            },
            "progress": {
                "mean_x": 0.0,
                "delta_x": 0.0,
                "is_advancing": False,
            },
            "episode": {
                "reward_mean": 0.0,
                "done": False,
            },
        },

        "summary": None,
    }


TRAIN_STATUS: Dict[str, Any] = default_train_status()

# ==========================================================
# 🔧 STATE UPDATE (DEEP & SAFE)
# ==========================================================
def deep_update(dst: Dict[str, Any], src: Dict[str, Any]):
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            deep_update(dst[k], v)
        else:
            dst[k] = v


def update_train_status(payload: Dict[str, Any]):
    with _STATUS_LOCK:
        if not payload:
            return

        # iter monotonic
        if "iter" in payload:
            TRAIN_STATUS["iter"] = max(
                TRAIN_STATUS.get("iter", 0),
                int(payload["iter"]),
            )

        # simple fields
        for k in ("state", "message", "max_iter", "progress", "algo", "env", "summary"):
            if k in payload:
                TRAIN_STATUS[k] = payload[k]

        # multiwalker deep merge
        if "multiwalker" in payload and isinstance(payload["multiwalker"], dict):
            deep_update(TRAIN_STATUS["multiwalker"], payload["multiwalker"])


# ==========================================================
# 🧠 TRAINING ENDPOINTS
# ==========================================================
@app.get("/")
def root():
    return {
        "message": "MARL backend online 🚀",
        "status": TRAIN_STATUS,
    }


@app.post("/train/mappo")
def start_mappo():
    global TRAIN_THREAD

    with _STATUS_LOCK:
        if TRAIN_STATUS["state"] == "running":
            raise HTTPException(status_code=400, detail="Training already running")

        # 🔥 reset eventi e METRICHE (SOURCE OF TRUTH)
        TRAIN_STOP_EVENT.clear()
        METRICS_RECORDER.reset()

        TRAIN_STATUS.clear()
        TRAIN_STATUS.update(default_train_status())
        TRAIN_STATUS.update({
            "state": "running",
            "algo": "mappo",
            "env": "multiwalker",
            "message": "Training started",
        })

    def run_train():
        try:
            train_mappo_gold(
                on_status_update=update_train_status,
                stop_event=TRAIN_STOP_EVENT,
                metrics_recorder=METRICS_RECORDER,  # 🔥 STESSA ISTANZA
            )
        except Exception as e:
            update_train_status({
                "state": "error",
                "message": str(e),
            })

    # 🔥 THREAD UNICO, DAEMON
    TRAIN_THREAD = threading.Thread(
        target=run_train,
        daemon=True,
        name="train-mappo-thread",
    )
    TRAIN_THREAD.start()

    return {
        "status": "started",
        "algo": "mappo",
        "env": "multiwalker",
    }


@app.get("/train/status")
def get_status():
    with _STATUS_LOCK:
        return TRAIN_STATUS


@app.get("/train/metrics")
def get_metrics():
    latest = METRICS_RECORDER.get_last()

    return {
        "count": len(METRICS_RECORDER.get_all()),
        "metrics": latest,              # ✅ OGGETTO, NON LISTA
        "summary": METRICS_RECORDER.summary(),
    }





@app.post("/train/stop")
def stop_train():
    with _STATUS_LOCK:
        if TRAIN_STATUS["state"] != "running":
            raise HTTPException(status_code=400, detail="No training running")

        TRAIN_STOP_EVENT.set()
        TRAIN_STATUS["state"] = "stopping"
        TRAIN_STATUS["message"] = "Stop requested"

    return {"status": "stopping"}


# ==========================================================
# 📈 TRAJECTORIES
# ==========================================================
@app.get("/train/trajectory")
def get_latest_trajectory():
    if not os.path.exists(LATEST_TRAJ):
        raise HTTPException(status_code=404, detail="Trajectory file not found")

    with open(LATEST_TRAJ, "r") as f:
        return {"iteration": "latest", "trajectory": json.load(f)}


@app.get("/train/trajectory/list")
def list_trajectories():
    pattern = re.compile(r"traj_mappo_iter(\d+)_([0-9]{8}_[0-9]{6})\.json")
    trajectories = []

    for fname in sorted(os.listdir(ARCHIVE_DIR)):
        match = pattern.match(fname)
        if not match:
            continue

        iter_num, ts = match.groups()
        trajectories.append({
            "iteration": int(iter_num),
            "timestamp": ts,
            "datetime": datetime.strptime(ts, "%Y%m%d_%H%M%S").isoformat(),
            "filename": fname,
        })

    latest = None
    if os.path.exists(LATEST_TRAJ):
        latest = {
            "iteration": "latest",
            "filename": os.path.basename(LATEST_TRAJ),
            "datetime": datetime.fromtimestamp(
                os.path.getmtime(LATEST_TRAJ)
            ).isoformat(),
        }

    return {
        "count": len(trajectories),
        "latest": latest,
        "trajectories": trajectories,
    }


@app.get("/train/trajectory/{iteration}")
def get_trajectory_by_iter(iteration: str):
    if iteration == "latest":
        target = LATEST_TRAJ
    else:
        target = None
        for f in os.listdir(ARCHIVE_DIR):
            if f.startswith(f"traj_mappo_iter{int(iteration):03d}_"):
                target = os.path.join(ARCHIVE_DIR, f)
                break

        if target is None:
            raise HTTPException(status_code=404, detail="Trajectory not found")

    if not os.path.exists(target):
        raise HTTPException(status_code=404, detail="Trajectory file not found")

    with open(target, "r") as f:
        data = json.load(f)

    return {
        "iteration": iteration,
        "count": len(data),
        "trajectory": data,
    }


# ==========================================================
# 🧩 WANDB API (AUXILIARY)
# ==========================================================
WANDB_PROJECT = "paolo-pangallo23-university-of-calabria/marl-from-scratch"


@app.get("/wandb/runs")
def list_wandb_runs():
    try:
        api = wandb.Api()
        runs = api.runs(WANDB_PROJECT)

        return [
            {
                "id": run.id,
                "name": run.name,
                "reward": run.summary.get("train/reward_mean"),
                "step": run.summary.get("_step"),
                "created_at": run.created_at.isoformat() if run.created_at else None,
            }
            for run in runs
        ]
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/wandb/metrics/{run_id}")
def wandb_metrics(run_id: str):
    try:
        api = wandb.Api()
        run = api.run(f"{WANDB_PROJECT}/{run_id}")
        hist = run.history(samples=500)

        return {
            "run_id": run_id,
            "metrics": hist.to_dict(orient="records"),
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# ==========================================================
# 🧭 DEBUG
# ==========================================================
@app.get("/debug/routes")
def debug_routes():
    from fastapi.routing import APIRoute
    return [
        {"path": r.path, "methods": list(r.methods)}
        for r in app.routes if isinstance(r, APIRoute)
    ]


print("[BOOT] Unified MARL backend loaded ✅")
print("[DEBUG] Loaded from:", __file__)
