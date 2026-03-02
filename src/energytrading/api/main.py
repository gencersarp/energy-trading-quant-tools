from fastapi import FastAPI
from pydantic import BaseModel
from energytrading.models.jump_diffusion import MertonJumpDiffusion, JumpDiffusionParams
from energytrading.risk.metrics import compute_risk_metrics
import numpy as np

app = FastAPI(title="EnergyTrading Quant API", description="REST API for simulation and risk.")

class SimRequest(BaseModel):
    s0: float
    steps: int
    paths: int
    mu: float
    sigma: float
    jump_intensity: float
    jump_mean: float
    jump_std: float

@app.post("/simulate/jump_diffusion")
def sim_jd(req: SimRequest):
    params = JumpDiffusionParams(
        req.mu, req.sigma, req.jump_intensity, req.jump_mean, req.jump_std
    )
    model = MertonJumpDiffusion(params)
    paths = model.simulate(req.s0, req.steps, req.paths)
    return {"paths": paths.tolist()}

@app.post("/risk/metrics")
def risk(returns: list[float]):
    return compute_risk_metrics(np.array(returns))