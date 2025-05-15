# api_nox.py

from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib

# Charger le modèle entraîné
model = joblib.load("Nox1_modèle.pkl")

# Créer l'application FastAPI
app = FastAPI(title="API Prédiction NOx")

# Définir les colonnes d'entrée
class Donnees(BaseModel):
    un: float
    deux: float
    trois: float
    quatre: float
    cinq: float
    sept: float
    huit: float
    neuf: float
    dix: float
    quinze: float
    seize: float
    dix_sept: float
    dix_huit: float
    dix_neuf: float
    vingt: float
    vingt_et_un: float
    vingt_deux: float
    vingt_trois: float
    vingt_quatre: float
    vingt_cinq: float
    vingt_huit: float
    vingt_neuf: float
    trente: float
    trente_et_un: float
    trente_trois: float

# Route de prédiction
@app.post("/predict")
def predict(data: Donnees):
    x = np.array([[data.un, data.deux, data.trois, data.quatre, data.cinq,
                   data.sept, data.huit, data.neuf, data.dix, data.quinze,
                   data.seize, data.dix_sept, data.dix_huit, data.dix_neuf,
                   data.vingt, data.vingt_et_un, data.vingt_deux, data.vingt_trois,
                   data.vingt_quatre, data.vingt_cinq, data.vingt_huit, data.vingt_neuf,
                   data.trente, data.trente_et_un, data.trente_trois]])

    pred = model.predict(x)[0]

    if pred > 500:
        alerte = " Attention : NOx dépasse la limite"
    elif pred > 400:
        alerte = " NOx élevé, surveiller le procédé"
    else:
        alerte = " NOx dans la norme"

    return {
        "NOx_baf_pred": round(float(pred), 2),
        "alerte": alerte
    }
