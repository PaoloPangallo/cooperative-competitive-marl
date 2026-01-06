from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os, json, wandb, glob

# Se il modulo viene eseguito dal root del progetto
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
LOG_DIR = os.path.join(BASE_DIR, "cooperative", "algorithms", "logs")

app = FastAPI(title="MARL Research Backend")

# CORS per comunicazione con frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "MARL backend attivo", "logs_dir": LOG_DIR}

@app.get("/trajectory/latest")
def latest_trajectory():
    files = sorted(glob.glob(os.path.join(LOG_DIR, "traj_*.json")), key=os.path.getmtime)
    if not files:
        return JSONResponse(content={"error": "Nessuna trajectory trovata"}, status_code=404)
    latest = files[-1]
    with open(latest, "r") as f:
        data = json.load(f)
    return {"filename": os.path.basename(latest), "data": data}

@app.get("/wandb/runs")
def list_runs():
    """Restituisce i run disponibili su wandb"""
    try:
        api = wandb.Api()
        runs = api.runs("paolo-pangallo23-university-of-calabria/marl-from-scratch")
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
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/wandb/metrics/{run_id}")
def wandb_metrics(run_id: str):
    """Scarica la history completa di un run specifico"""
    try:
        api = wandb.Api()
        run = api.run(f"paolo-pangallo23-university-of-calabria/marl-from-scratch/{run_id}")
        hist = run.history(samples=500)
        return {"run_id": run_id, "metrics": hist.to_dict(orient="records")}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
