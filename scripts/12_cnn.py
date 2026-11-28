"""Semana 3 · Entrena la CNN (§4.5) con una variante del preprocesamiento y una semilla. Mide en VALIDACIÓN.

Uso:   python scripts/12_cnn.py <variante> <semilla>         (requiere scripts/11_tensores_cnn.py y torch)
       python scripts/12_cnn.py todas                        (las 6 variantes × 5 semillas = 30 corridas; se corre en la VM,
                                                              guía en docs/ENTRENAR_CNN_EN_VM.md)

- Entrada: data/interim/X_<variante>.npy (3 × 128 × 128). La red, el aumento y la pérdida son los mismos en todas las
  variantes, para que la única diferencia entre corridas sea la entrada.
- Red v2 (27-nov): más densa que la plantilla de scripts/04_entrenar_cnn.py (v1: 4 bloques 24-48-96-160, ~0,6 M de
  parámetros). v2: 5 bloques 32-64-128-192-256 y cabeza 256 → 64 → 1, ~2 M de parámetros. Las corridas de la v1 no
  se mezclan con las de la v2: cada JSON guarda la versión de la red y 13_comparar_cnn.py exige que todas sean v2.
- `todas` es reanudable: salta las corridas que ya tienen su JSON con la red v2.
- Salida de la red: probabilidad de defecto (regresión a 0 / ⅓ / ⅔ / 1, como el trabajo original); umbral 0,5 → 2 clases.
- Pesos por clase; volteos y traslaciones pequeñas; nada de rotar 90° (los busbars son siempre horizontales).
- Se guarda la época con menor error cuadrático en validación. Es la misma regla para todas las variantes.
- La prueba no se carga: 11_tensores_cnn.py no la incluye hasta la semana 4.

Salidas: reports/metricas/cnn/<variante>_s<semilla>.json (métricas + historial) y _pred.csv (predicción por celda de val)
"""
import json
import os
import subprocess
import sys
import time

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score

VARIANTES = ["original", "In", "A-global", "A-filas", "B", "B-dedos"]
SEMILLAS = [0, 1, 2, 3, 4]
RED = "v2"
INT = os.path.join(RAIZ, "data", "interim"); MET = os.path.join(RAIZ, "reports", "metricas", "cnn"); os.makedirs(MET, exist_ok=True)


def hecha(v, s):
    f = os.path.join(MET, f"{v}_s{s}.json")
    return os.path.exists(f) and json.load(open(f)).get("red") == RED


if sys.argv[1] == "todas":   # por semilla: con las primeras semillas completas ya se puede mirar la comparación
    for s in SEMILLAS:
        for v in VARIANTES:
            if not hecha(v, s):
                subprocess.run([sys.executable, __file__, v, str(s)], check=True)
    sys.exit()

VAR, SEMILLA = sys.argv[1], int(sys.argv[2])
EPOCHS = int(os.environ.get("EPOCHS", 40)); BATCH = int(os.environ.get("BATCH", 32))
dev = "cuda" if torch.cuda.is_available() else "cpu"
if dev == "cpu": torch.set_num_threads(int(os.environ.get("HILOS", os.cpu_count() or 1)))
torch.manual_seed(SEMILLA); np.random.seed(SEMILLA); torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True

F = pd.read_csv(os.path.join(INT, "cnn_indices.csv"))
assert "test" not in set(F.split), "la prueba no se usa en la semana 3"
p = F.prob.values.astype(np.float32)
X = np.load(os.path.join(INT, f"X_{VAR}.npy")).astype(np.float32) / 255.0
tr, va = (np.where(F.split.values == s)[0] for s in ["train", "val"])
mu = X[tr].mean((0, 2, 3), keepdims=True); sd = X[tr].std((0, 2, 3), keepdims=True) + 1e-6   # solo con train
Xt = torch.from_numpy((X - mu) / sd); pt = torch.from_numpy(p)
cls = F.clase4.values; wcls = len(tr) / (4 * np.bincount(cls[tr], minlength=4))
wt = torch.from_numpy(wcls[cls].astype(np.float32))


def bloque(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(),
                         nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2))


class Red(nn.Module):
    def __init__(s):
        super().__init__()
        s.f = nn.Sequential(bloque(3, 32), bloque(32, 64), bloque(64, 128), bloque(128, 192), bloque(192, 256))   # 128 → 4 px
        s.h = nn.Sequential(nn.Dropout(.4), nn.Linear(256, 64), nn.ReLU(), nn.Dropout(.2), nn.Linear(64, 1))

    def forward(s, x):
        x = s.f(x); x = torch.amax(x, (2, 3)) * .5 + x.mean((2, 3)) * .5
        return torch.sigmoid(s.h(x)).squeeze(1)


def aumentar(x):
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
    yb = p[idx] >= .5; yh = pr >= .5; ty = F.tipo.values[idx]; r = {}
    for t, s in (("todas", slice(None)), ("mono", ty == "mono"), ("poly", ty == "poly")):
        r[t] = {"n": int(len(yb[s])), "auc": float(roc_auc_score(yb[s], pr[s])),
                "exactitud_balanceada": float(balanced_accuracy_score(yb[s], yh[s])), "f1_defectuosa": float(f1_score(yb[s], yh[s])),
                "exactitud": float((yh[s] == yb[s]).mean()), "cm_2clases": confusion_matrix(yb[s], yh[s], labels=[0, 1]).tolist()}
    return r


ruta_pesos = os.path.join(INT, f"cnn_{VAR}_s{SEMILLA}.pt")
hist, mejor, t0 = [], 1e9, time.time()
for ep in range(EPOCHS):
    red.train(); perm = np.random.permutation(tr); L = 0.0
    for i in range(0, len(perm), BATCH):
        b = perm[i:i + BATCH]; xb = aumentar(Xt[b]).to(dev)
        loss = (wt[b].to(dev) * (red(xb) - pt[b].to(dev)) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step(); L += loss.item() * len(b)
    pv = predecir(va); lv = float(np.mean((pv - p[va]) ** 2))
    hist.append({"ep": ep, "train": L / len(tr), "val": lv, "val_auc": float(roc_auc_score(p[va] >= .5, pv))})
    if lv < mejor: mejor = lv; mejor_ep = ep; torch.save(red.state_dict(), ruta_pesos)
    print(f"{VAR} s{SEMILLA} ep {ep:2d} train {L / len(tr):.4f} val {lv:.4f} auc_val {hist[-1]['val_auc']:.3f} {time.time() - t0:.0f}s", flush=True)

red.load_state_dict(torch.load(ruta_pesos, map_location=dev))
t1 = time.time(); pv = predecir(va); ms = 1000 * (time.time() - t1) / len(va)
res = {"variante": VAR, "semilla": SEMILLA, "red": RED, "parametros": sum(q.numel() for q in red.parameters()),
       "dispositivo": dev, "hilos": torch.get_num_threads(), "epocas": EPOCHS, "mejor_epoca": mejor_ep,
       "n": {"train": len(tr), "val": len(va)}, "val": metricas(va, pv), "ms_por_celda_cnn": ms, "historial": hist,
       "segundos": time.time() - t0}
pd.DataFrame({"idx": F.idx.values[va], "tipo": F.tipo.values[va], "prob": p[va], "pred": pv}).to_csv(
    os.path.join(MET, f"{VAR}_s{SEMILLA}_pred.csv"), index=False)
json.dump(res, open(os.path.join(MET, f"{VAR}_s{SEMILLA}.json"), "w"), indent=1)   # al final: marca la corrida como hecha
print(json.dumps({k: res["val"][k]["auc"] for k in res["val"]}), f"mejor época {mejor_ep}")
