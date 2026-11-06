"""Fase 4 · PLANTILLA de la CNN (todavía sin entrenar). Correr en una máquina con GPU o en Colab.

Uso:   python scripts/04_entrenar_cnn.py [pre|raw] [semilla]     (requiere 01_preprocesar_todo.py y torch)
       EPOCHS=40 BATCH=32 python scripts/04_entrenar_cnn.py pre 0

- Entrada "pre": 3 canales 128x128 = [celda normalizada sin busbars, mapa de déficit, mapa de crestas].
  Entrada "raw": la celda original repetida en 3 canales (sirve para medir cuánto aporta el preprocesamiento).
- Salida: regresión a la probabilidad (0 / .33 / .67 / 1), como Deitsch et al. 2019 -> se reporta en 2 clases
  (umbral 0,5) y en 4 (redondeo).
- Partición: la congelada en data/particion.csv. Se entrena con train, se elige la época con val.
  TEST NO SE TOCA hasta la evaluación final: pasar EVALUAR_TEST=1 solo esa vez.
- Pendiente para el equipo: cambiar la red pequeña por ResNet18 preentrenada (torchvision) si hay GPU,
  y exportar a ONNX para la tarjeta Hailo (opcional, ver docs/README).
"""
import json
import os
import sys
import time

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score

MODO = sys.argv[1] if len(sys.argv) > 1 else "pre"
SEMILLA = int(sys.argv[2]) if len(sys.argv) > 2 else 0
EPOCHS = int(os.environ.get("EPOCHS", 40)); BATCH = int(os.environ.get("BATCH", 32))
EVALUAR_TEST = os.environ.get("EVALUAR_TEST") == "1"
INT = os.path.join(RAIZ, "data", "interim"); MOD = os.path.join(RAIZ, "models"); os.makedirs(MOD, exist_ok=True)
MET = os.path.join(RAIZ, "reports", "metricas"); os.makedirs(MET, exist_ok=True)
dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(SEMILLA); np.random.seed(SEMILLA)

F = pd.read_csv(os.path.join(INT, "features.csv"))
p = F.prob.values.astype(np.float32)
X = np.load(os.path.join(INT, "X_pre.npy")) if MODO == "pre" else np.repeat(np.load(os.path.join(INT, "X_raw.npy"))[:, None], 3, 1)
X = X.astype(np.float32) / 255.0
tr, va, te = (np.where(F.split.values == s)[0] for s in ["train", "val", "test"])
mu = X[tr].mean((0, 2, 3), keepdims=True); sd = X[tr].std((0, 2, 3), keepdims=True) + 1e-6   # solo con train
Xt = torch.from_numpy((X - mu) / sd); pt = torch.from_numpy(p)
cls = F.clase4.values; wcls = len(tr) / (4 * np.bincount(cls[tr], minlength=4))             # peso por clase
wt = torch.from_numpy(wcls[cls].astype(np.float32))


def bloque(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(),
                         nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2))


class Red(nn.Module):
    def __init__(s):
        super().__init__()
        s.f = nn.Sequential(bloque(3, 24), bloque(24, 48), bloque(48, 96), bloque(96, 160))
        s.h = nn.Sequential(nn.Dropout(.3), nn.Linear(160, 1))

    def forward(s, x):
        x = s.f(x); x = torch.amax(x, (2, 3)) * .5 + x.mean((2, 3)) * .5
        return torch.sigmoid(s.h(x)).squeeze(1)


def aumentar(x):
    """Volteos y traslaciones pequeñas. Sin rotar 90°: los busbars siempre son horizontales."""
    if np.random.rand() < .5: x = x.flip(3)
    if np.random.rand() < .5: x = x.flip(2)
    dx, dy = np.random.randint(-4, 5, 2)
    return torch.roll(x, (int(dy), int(dx)), (2, 3))


red = Red().to(dev)
opt = torch.optim.AdamW(red.parameters(), 2e-3, weight_decay=1e-3)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-3, total_steps=EPOCHS * int(np.ceil(len(tr) / BATCH)))


def predecir(idx):
    red.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(idx), 128):
            xb = Xt[idx[i:i + 128]].to(dev)
            out.append(((red(xb) + red(xb.flip(3)) + red(xb.flip(2))) / 3).cpu().numpy())
    return np.concatenate(out)


def metricas(idx, pr):
    yb = p[idx] >= .5; yh = pr >= .5; ty = F.tipo.values[idx]
    c4t = np.rint(p[idx] * 3).astype(int); c4p = np.clip(np.rint(pr * 3), 0, 3).astype(int)
    return {"auc": roc_auc_score(yb, pr), "exactitud": float((yh == yb).mean()),
            "exactitud_balanceada": balanced_accuracy_score(yb, yh), "f1_defectuosa": f1_score(yb, yh),
            "exactitud_mono": float((yh == yb)[ty == "mono"].mean()), "exactitud_poly": float((yh == yb)[ty == "poly"].mean()),
            "exactitud_4clases": float((c4p == c4t).mean()), "cm_2clases": confusion_matrix(yb, yh).tolist(),
            "cm_4clases": confusion_matrix(c4t, c4p, labels=[0, 1, 2, 3]).tolist()}


ruta_pesos = os.path.join(MOD, f"cnn_{MODO}_{SEMILLA}.pt")
hist, mejor, t0 = [], 1e9, time.time()
for ep in range(EPOCHS):
    red.train(); perm = np.random.permutation(tr); L = 0.0
    for i in range(0, len(perm), BATCH):
        b = perm[i:i + BATCH]; xb = aumentar(Xt[b]).to(dev)
        loss = (wt[b].to(dev) * (red(xb) - pt[b].to(dev)) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step(); L += loss.item() * len(b)
    pv = predecir(va); lv = float(np.mean((pv - p[va]) ** 2))
    hist.append({"ep": ep, "train": L / len(tr), "val": lv, "val_auc": roc_auc_score(p[va] >= .5, pv)})
    if lv < mejor: mejor = lv; torch.save(red.state_dict(), ruta_pesos)
    print(f"ep {ep:2d} train {L / len(tr):.4f} val {lv:.4f} auc_val {hist[-1]['val_auc']:.3f} {time.time() - t0:.0f}s", flush=True)

red.load_state_dict(torch.load(ruta_pesos, map_location=dev))
res = {"modo": MODO, "semilla": SEMILLA, "epocas": EPOCHS, "n": {"train": len(tr), "val": len(va), "test": len(te)},
       "val": metricas(va, predecir(va)), "historial": hist, "segundos": time.time() - t0}
if EVALUAR_TEST:
    res["test"] = metricas(te, predecir(te))
json.dump(res, open(os.path.join(MET, f"cnn_{MODO}_{SEMILLA}.json"), "w"), indent=1, default=float)
print(json.dumps({k: v for k, v in res.items() if k != "historial"}, indent=1, default=float))
