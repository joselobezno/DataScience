import joblib

def save_model(model):
    joblib.dump(model,"models/modelo_jose_villalba.joblib")

    return print("modelo guardado exitosamente uwu")