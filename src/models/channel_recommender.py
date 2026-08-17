import joblib
import pandas as pd
import numpy as np
import os


class ChannelRecommender:
    """
    Evalúa contrafactualmente los 4 canales de venta y devuelve el que maximiza
    la probabilidad de aceptación.
    """

    def __init__(self, model_path='../../models_artifacts/acceptance_model.pkl', categories=None):
        if not os.path.exists(model_path):
            model_path = os.path.join(os.path.dirname(__file__), '..', '..', 'models_artifacts', 'acceptance_model.pkl')

        self.model = joblib.load(model_path)
        self.channels = ['Tienda', 'Call In', 'Call Out', 'Digital']

        # Categorías fijadas en el entrenamiento. Son imprescindibles: al evaluar
        # una sola fila, un .astype('category') infiere las categorías de ese
        # único valor y LightGBM recibe el código 0 para todas las columnas, que
        # no es la codificación con la que se entrenó.
        if categories is None:
            cat_path = os.path.join(os.path.dirname(model_path), 'model_categories.pkl')
            categories = joblib.load(cat_path) if os.path.exists(cat_path) else {}
        self.categories = categories

    def _prepare(self, df):
        df = df.copy()
        for col, cats in self.categories.items():
            if col in df.columns:
                df[col] = pd.Categorical(df[col].astype(str), categories=cats)
        return df

    # Diferencia mínima entre el mejor y el peor canal para afirmar que el canal
    # importa. Por debajo de este umbral la elección sería ruido: en esta base el
    # canal tiene aceptación plana (37.3% a 37.6% según el histórico) y la
    # diferencia mediana que produce el modelo entre canales es de 0.000 pp.
    UMBRAL_DIFERENCIA = 0.01

    def recommend(self, customer_offer_features: pd.DataFrame, canal_habitual=None) -> dict:
        """
        Dada una fila cliente × oferta, predice la probabilidad de aceptación en
        cada canal y devuelve el mejor.

        Si el modelo no distingue entre canales de forma material, no se inventa
        un ganador: se cae al canal habitual del cliente, que es una heurística
        de negocio defendible, y se marca `decidido_por_modelo=False` para que la
        interfaz no presente como recomendación algorítmica lo que no lo es.
        """
        # Un solo predict con los 4 canales apilados, en vez de 4 llamadas.
        grid = pd.concat([customer_offer_features] * len(self.channels), ignore_index=True)
        grid['canal'] = self.channels
        probs = self.model.predict_proba(self._prepare(grid))[:, 1]

        results = sorted(zip(self.channels, probs), key=lambda x: x[1], reverse=True)
        best_channel, max_prob = results[0]
        diferencia = float(results[0][1] - results[-1][1])

        decidido_por_modelo = diferencia >= self.UMBRAL_DIFERENCIA
        if not decidido_por_modelo and canal_habitual in self.channels:
            best_channel = canal_habitual
            max_prob = float(dict(results)[canal_habitual])

        return {
            'best_channel': best_channel,
            'max_prob': float(max_prob),
            'all_probs': {ch: float(p) for ch, p in results},
            'diferencia_canales': diferencia,
            'decidido_por_modelo': decidido_por_modelo,
            'criterio': ('El canal cambia la probabilidad de aceptación de forma material.'
                         if decidido_por_modelo else
                         'El modelo no distingue entre canales para este caso: se usa el '
                         'canal habitual del cliente.'),
        }
