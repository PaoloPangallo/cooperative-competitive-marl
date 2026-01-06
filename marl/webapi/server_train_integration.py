# marl/webapi/server_train_integration.py
import threading
import os
import json
import signal
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

from marl.cooperative.algorithms.train_mappo_multiwalker import train_mappo_gold

app = FastAPI(title="MARL Training Backend", version="0.2.0")

TRAIN_THREAD: Optional[threading.Thread] = None
TRAIN_STOP_EVENT = threading.Event()
TRAIN_STATUS = {"state": "idle", "iter": 0, "message": "Ready"}

TRAJ_PATH = os.path.join(
    os.path.dirname(__file__),
    "..", "cooperative", "trajectories", "latest_trajectory.json"
)

class TrainStatus(BaseModel):
    state: str
    iter: int
    message: str


@app.get("/")
def root():
    return {"message": "MARL backend online 🚀", "status": TRAIN_STATUS}


@app.post("/train/mappo")
def start_mappo():
    global TRAIN_THREAD, TRAIN_STATUS, TRAIN_STOP_EVENT

    if TRAIN_STATUS["state"] == "running":
        raise HTTPException(status_code=400, detail="Training already running")

    TRAIN_STOP_EVENT.clear()
    TRAIN_STATUS.update({"state": "running", "iter": 0, "message": "Training started"})

    def run_train():
        try:
            train_mappo_gold()
            TRAIN_STATUS.update({"state": "done", "message": "Training completed"})
        except Exception as e:
            TRAIN_STATUS.update({"state": "error", "message": str(e)})

    TRAIN_THREAD = threading.Thread(target=run_train, daemon=True)
    TRAIN_THREAD.start()

    return {"status": "started", "message": "Training MAPPO avviato in background."}


@app.get("/train/status", response_model=TrainStatus)
def get_status():
    return TRAIN_STATUS


@app.post("/train/stop")
def stop_train():
    global TRAIN_STATUS

    if TRAIN_STATUS["state"] != "running":
        raise HTTPException(status_code=400, detail="No training running")

    TRAIN_STOP_EVENT.set()
    TRAIN_STATUS.update({"state": "stopping", "message": "Stop signal sent"})

    # Se vuoi fermare davvero i processi lenti / GPU puoi fare:
    os.kill(os.getpid(), signal.SIGINT)

    return {"status": "stopping", "message": "Training stop requested"}


@app.get("/train/trajectory")
def get_latest_trajectory():
    if not os.path.exists(TRAJ_PATH):
        raise HTTPException(status_code=404, detail="Trajectory file not found")

    try:
        with open(TRAJ_PATH, "r") as f:
            data = json.load(f)
        return {"status": "ok", "trajectory": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading trajectory: {e}")
