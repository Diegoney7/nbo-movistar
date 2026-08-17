import joblib
import pandas as pd
import shap
import os
import numpy as np


# Traducción de una variable del modelo a un argumento comercial.
#
# Cada mensaje depende del VALOR de la variable, no solo de que SHAP la haya
# elegido. SHAP dice qué variable empujó la predicción hacia arriba, pero no en
# qué dirección leerla: `n_reclamos` puede aportar positivamente porque el
# cliente NO tiene reclamos. Un texto fijo por variable terminaba diciéndole
# "le ofrecemos una mejora tras sus contactos recientes" a un cliente que nunca
# reclamó — un argumento falso, que además se propaga al speech que redacta la
# IA generativa (ver src/llm_assistant.py). Cuando el valor no sostiene ningún
# argumento honesto, la función devuelve None y la variable se descarta.
FEATURE_TRANSLATION = {
    'mt_match': lambda v: (
        "Es elegible para Movistar Total y esta oferta es justamente el paquete convergente."
        if v >= 1 else None),
    'brecha_mt': lambda v: (
        "Está a un solo servicio de completar Movistar Total." if v == 1 else
        "Ya tiene móvil y hogar: la convergencia le aplica de forma directa." if v == 0 else None),
    'ratio_consumo_gb': lambda v: (
        "Su consumo de datos está al límite de lo que incluye su plan actual." if v >= 0.85 else None),
    'consumo_datos_gb_prom': lambda v: (
        f"Tiene un consumo de datos alto: {v:.0f} GB al mes." if v >= 25 else None),
    'dias_mora_prom': lambda v: (
        "Su historial de pagos está limpio, lo que habilita mejores condiciones." if v <= 5 else None),
    'meses_moroso': lambda v: (
        "Está al día con sus pagos." if v <= 0 else None),
    'n_reclamos': lambda v: (
        "Registra reclamos recientes: esta mejora ayuda a recuperar su satisfacción." if v >= 1 else
        "No registra reclamos: la relación está sana para proponerle una mejora."),
    'ahorro_pct': lambda v: (
        f"La oferta ahorra {v:.0f}% frente a contratar los mismos servicios por separado."
        if v > 0 else None),
    'es_usuario_app': lambda v: (
        "Usa la app Movistar, así que puede conocer y cerrar la oferta por canal digital."
        if bool(v) else None),
    'antiguedad_meses': lambda v: (
        f"Lleva {v / 12:.0f} años con Movistar: la antigüedad juega a favor de la propuesta."
        if v >= 36 else None),
    'precio_mensual': lambda v: "El precio de la oferta encaja con su nivel de gasto actual.",
    'gb_incluidos': lambda v: (
        "La oferta incluye datos ilimitados." if v >= 9999 else
        f"La oferta incluye {v:.0f} GB." if v > 0 else None),
}


class Explainer:
    def __init__(self, model_path='../../models_artifacts/acceptance_model_raw.pkl'):
        # SHAP TreeExplainer necesita el booster LightGBM directo. El modelo que
        # sirve probabilidades (acceptance_model.pkl) es un CalibratedClassifierCV
        # que envuelve al booster y TreeExplainer no lo acepta, así que la
        # explicabilidad se calcula sobre el booster crudo. La calibración
        # isotónica es monótona: reescala las probabilidades sin alterar el orden
        # ni el signo de las contribuciones, de modo que los motivos siguen siendo
        # los del modelo que se muestra.
        if not os.path.exists(model_path):
            model_path = os.path.join(os.path.dirname(__file__), '..', '..', 'models_artifacts', 'acceptance_model_raw.pkl')

        self.model = joblib.load(model_path)
        self.explainer = shap.TreeExplainer(self.model)
        self.feature_translation = FEATURE_TRANSLATION

    def explain(self, row_features: pd.DataFrame) -> list:
        """
        Calcula los valores SHAP de una predicción y devuelve los dos argumentos
        comerciales con mayor contribución positiva, en lenguaje natural.
        """
        shap_values = self.explainer.shap_values(row_features)

        # En clasificación binaria LightGBM puede devolver una lista de 2 arrays
        # (uno por clase) o un único array con la contribución a la clase 1.
        if isinstance(shap_values, list):
            sv = shap_values[1][0]
        else:
            sv = shap_values[0]

        top_indices = np.argsort(sv)[::-1]

        reasons = []
        features_list = row_features.columns.tolist()
        valores = row_features.iloc[0]

        for idx in top_indices:
            feat_name = features_list[idx]

            # Solo contribuciones positivas y variables que sepamos traducir.
            if sv[idx] <= 0 or feat_name not in self.feature_translation:
                continue

            try:
                mensaje = self.feature_translation[feat_name](valores[feat_name])
            except (TypeError, ValueError):
                mensaje = None

            # El valor de la variable no sostiene un argumento honesto: se descarta.
            if mensaje and mensaje not in reasons:
                reasons.append(mensaje)

            if len(reasons) == 2:
                break

        if not reasons:
            reasons.append("El perfil de este cliente es el que mejor encaja con esta oferta "
                           "dentro del catálogo disponible.")

        return reasons
