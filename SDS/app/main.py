from fastapi import FastAPI
import numpy as np
import torch

from app.schemas import NetworkEvent
from app.utils import (
    supervised_model,
    unsupervised_model,
    policy_net,
    Q
)
from app.rl_utils import q_learning_predict
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi import Request

app = FastAPI(title="Cybersecurity AI API")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})



@app.post("/predict/all")
def predict_all(event: NetworkEvent):

    X = np.array([[
        event.Duration,
        event.Src_Pt,
        event.Dst_Pt,
        event.Bytes,
        event.bytes_per_packet
    ]])

    # ---------- 1️⃣ Supervised ----------
    sup_pred = int(supervised_model.predict(X)[0])
    sup_label = "NORMAL" if sup_pred == 0 else "ATTACK"

    # ---------- 2️⃣ Unsupervised ----------
    unsup_pred = int(unsupervised_model.predict(X)[0])
    unsup_label = "NORMAL" if unsup_pred == 1 else "ANOMALY"

    # ---------- 3️⃣ RL ----------
    state = tuple(int(x) for x in X[0])
    rl_action = Q.get(state, 0)
    rl_label = "BLOCK" if rl_action == 1 else "ALLOW"

    # ---------- 4️⃣ DRL ----------
    with torch.no_grad():
        state_t = torch.tensor(X, dtype=torch.float32)
        drl_action = int(torch.argmax(policy_net(state_t), dim=1).item())
    drl_label = "BLOCK" if drl_action == 1 else "ALLOW"

    return {
        "supervised": sup_label,
        "unsupervised": unsup_label,
        "rl": rl_label,
        "drl": drl_label
    }
