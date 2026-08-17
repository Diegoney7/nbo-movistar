"""
Segmentación comportamental de clientes (eje B de la matriz de segmentación).

El eje A (macro-segmento de brecha Movistar Total) es determinístico y vive en
src/segment_policy.py: la elegibilidad MT es una regla de negocio, no un patrón
latente, y el EDA confirma que los clientes elegibles MT tienen un perfil de
comportamiento indistinguible del resto de la base móvil.

Este módulo cubre el eje complementario: un tiering de valor / riesgo /
digitalidad sobre el que se apoyan el ranking del motor NBO y la vista de
Marketing.
"""
import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_loader import DataLoader

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATA_DIR = os.path.join(BASE_DIR, 'data')
ARTIFACTS_DIR = os.path.join(BASE_DIR, 'models_artifacts')

RANDOM_STATE = 42
SILHOUETTE_SAMPLE = 15000

# Feature space curado. Se excluyen a propósito:
#   - monto_facturado_prom_6m : r = 1.000 con monto_facturado_prom
#   - meses_moroso            : r = 0.941 con dias_mora_prom
#   - consumo_sms_prom        : coef. de variación 0.14 (casi constante)
#   - consumo_voz_min_prom    : coef. de variación 0.20 (casi constante)
# Incluir los pares redundantes duplicaría el peso de esos ejes en la distancia
# euclídea; las dos últimas solo aportarían ruido esférico.
SEGMENTATION_FEATURES = [
    'log_monto',
    'log_gb',
    'ratio_consumo_gb',
    'antiguedad_meses',
    'dias_mora_prom',
    'n_reclamos',
    'n_actividad_canal',
    'score_digital',
]

# Nombre comercial de cada rasgo, según si el centroide del cluster está por
# encima o por debajo de la media de la base.
TRAIT_NAMES = {
    ('log_monto', 'alto'): 'Alto Valor',
    ('log_monto', 'bajo'): 'Bajo Ticket',
    ('log_gb', 'alto'): 'Heavy User de Datos',
    ('log_gb', 'bajo'): 'Bajo Consumo',
    ('ratio_consumo_gb', 'alto'): 'Plan al Límite',
    ('ratio_consumo_gb', 'bajo'): 'Plan Holgado',
    ('antiguedad_meses', 'alto'): 'Veterano',
    ('antiguedad_meses', 'bajo'): 'Reciente',
    ('dias_mora_prom', 'alto'): 'Riesgo por Mora',
    ('dias_mora_prom', 'bajo'): 'Pagador Puntual',
    ('n_reclamos', 'alto'): 'Alta Fricción',
    ('n_reclamos', 'bajo'): 'Sin Fricción',
    ('n_actividad_canal', 'alto'): 'Muy Interactivo',
    ('n_actividad_canal', 'bajo'): 'Poco Interactivo',
    ('score_digital', 'alto'): 'Digital',
    ('score_digital', 'bajo'): 'Presencial',
}


def build_segmentation_features(df_clientes, df_ofertas):
    """
    Construye el feature space de segmentación.

    Función pública compartida: la usan tanto el entrenamiento como el motor NBO
    en inferencia, para que ambos apliquen exactamente la misma transformación y
    no haya train/serve skew.

    Devuelve el DataFrame de clientes con las columnas de SEGMENTATION_FEATURES
    añadidas.
    """
    d = df_clientes.copy()

    # El motor NBO ya cruza el catálogo antes de llamarnos; no repetimos el merge
    # para no generar columnas duplicadas con sufijos _x / _y.
    if 'plan_actual_gb' not in d.columns or 'plan_actual_precio' not in d.columns:
        planes_info = df_ofertas[['oferta_id', 'gb_incluidos', 'precio_mensual']].rename(
            columns={
                'oferta_id': 'plan_actual_id',
                'gb_incluidos': 'plan_actual_gb',
                'precio_mensual': 'plan_actual_precio',
            }
        )
        d = d.merge(planes_info, on='plan_actual_id', how='left')

    plan_gb = d['plan_actual_gb'].fillna(0)

    # Presión sobre el plan contratado. Misma definición que en
    # feature_engineering.py:26, con clip para que los planes muy pequeños no
    # generen outliers que dominen la distancia euclídea.
    d['ratio_consumo_gb'] = np.where(
        plan_gb > 0, d['consumo_datos_gb_prom'] / plan_gb, 0
    ).clip(0, 3)

    # Valor y consumo entran en log: ambas distribuciones son asimétricas a la
    # derecha (facturación mediana 76 vs. máximo 306).
    d['log_monto'] = np.log1p(d['monto_facturado_prom'])
    d['log_gb'] = np.log1p(d['consumo_datos_gb_prom'])

    # Digitalidad combinada: adopción de app, intensidad de uso y preferencia de
    # canal en una sola variable, en vez de one-hot de canal (que distorsiona
    # distancias en KMeans).
    d['score_digital'] = (
        d['es_usuario_app'].astype(int) * d['uso_app_movistar_prom']
        + (d['canal_mas_usado'] == 'Digital').astype(int) * 2
    )

    for col in SEGMENTATION_FEATURES:
        d[col] = d[col].fillna(0)

    return d


def evaluate_k_range(X_scaled, k_min=2, k_max=8):
    """Barre K reportando codo (inercia), silhouette y Davies-Bouldin."""
    rng = np.random.RandomState(RANDOM_STATE)
    n_sample = min(SILHOUETTE_SAMPLE, len(X_scaled))
    sample_idx = rng.choice(len(X_scaled), n_sample, replace=False)

    rows = []
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10).fit(X_scaled)
        labels = km.labels_
        sizes = np.bincount(labels) / len(labels) * 100
        rows.append({
            'k': k,
            'inercia': km.inertia_,
            'silhouette': silhouette_score(X_scaled[sample_idx], labels[sample_idx]),
            'davies_bouldin': davies_bouldin_score(X_scaled[sample_idx], labels[sample_idx]),
            'cluster_min_pct': sizes.min(),
        })

    return pd.DataFrame(rows).set_index('k')


def choose_k(metrics, min_cluster_pct=5.0):
    """
    Elige K por el codo de la curva de inercia (método kneedle: punto de máxima
    distancia a la recta que une los extremos de la curva).

    No se usa Davies-Bouldin ni silhouette como criterio principal porque en esta
    base ninguno tiene óptimo interior: DB decrece de forma monótona con K y el
    silhouette se mantiene plano entre 0.17 y 0.21. Eso confirma que no hay
    grupos naturales separables — el clustering funciona como tiering, no como
    descubrimiento — y en ese escenario el codo, junto con la restricción de que
    ningún segmento sea residual, es el criterio defendible. Además mantiene el
    número de segmentos en un rango que un asesor puede manejar bajo presión.
    """
    viable = metrics[metrics['cluster_min_pct'] >= min_cluster_pct]
    if len(viable) < 3:
        viable = metrics

    ks = viable.index.to_numpy(dtype=float)
    inertias = viable['inercia'].to_numpy(dtype=float)

    # Distancia vertical de cada punto a la recta que une el primer y último K.
    slope = (inertias[-1] - inertias[0]) / (ks[-1] - ks[0])
    line = inertias[0] + slope * (ks - ks[0])
    return int(ks[np.argmax(line - inertias)])


def name_clusters(centroids_scaled, feature_names):
    """
    Deriva un nombre legible por cluster a partir de sus rasgos más distintivos.

    Se toma el rasgo con mayor |z| del centroide (en espacio estandarizado, donde
    z=0 es la media de la base). Si dos clusters comparten rasgo principal, se
    desempata añadiendo el segundo rasgo más distintivo.
    """
    order = np.argsort(-np.abs(centroids_scaled), axis=1)
    names = {}

    for i in range(len(centroids_scaled)):
        primary = feature_names[order[i, 0]]
        direction = 'alto' if centroids_scaled[i, order[i, 0]] > 0 else 'bajo'
        names[i] = TRAIT_NAMES[(primary, direction)]

    # Desempate de nombres repetidos con el rasgo secundario.
    for name in list(names.values()):
        dupes = [i for i, n in names.items() if n == name]
        if len(dupes) > 1:
            for i in dupes:
                secondary = feature_names[order[i, 1]]
                direction = 'alto' if centroids_scaled[i, order[i, 1]] > 0 else 'bajo'
                names[i] = f"{name} · {TRAIT_NAMES[(secondary, direction)]}"

    return names


TRAIT_SOURCE = {
    'es_riesgo': 'dias_mora_prom',
    'es_alto_valor': 'monto_facturado_prom',
    'es_digital': 'score_digital',
    'plan_al_limite': 'ratio_consumo_gb',
    'alta_friccion': 'n_reclamos',
}

# Un rasgo solo se activa si el cluster destaca (z > 0.5 frente a los demás
# clusters) Y la variable realmente separa clusters. El segundo guard es
# necesario: si todos los centroides son casi iguales, el z-score amplifica
# ruido y marcaría el rasgo en cualquier cluster. Con score_digital pasa
# exactamente eso en esta base (media ~3.0 en los cinco segmentos).
TRAIT_Z_THRESHOLD = 0.5
TRAIT_MIN_SPREAD_CV = 0.10


def build_cluster_traits(profile_table):
    """
    Marca rasgos booleanos por cluster para que la política de negocio se apoye
    en características y no en números de cluster (que cambian entre corridas).
    """
    traits = {int(cid): {} for cid in profile_table.index}

    for trait, col in TRAIT_SOURCE.items():
        values = profile_table[col].astype(float)
        spread_cv = values.std() / values.mean() if values.mean() else 0.0
        discrimina = spread_cv >= TRAIT_MIN_SPREAD_CV

        z = (values - values.mean()) / values.std() if values.std() else values * 0
        for cid in profile_table.index:
            traits[int(cid)][trait] = bool(discrimina and z[cid] > TRAIT_Z_THRESHOLD)

    return traits


def train_segmentation(k=None):
    print("Cargando datos para segmentación...")
    loader = DataLoader(data_dir=DATA_DIR)
    clientes = loader.load_clientes()
    ofertas = loader.load_ofertas()

    d = build_segmentation_features(clientes, ofertas)
    X = d[SEGMENTATION_FEATURES]

    print(f"Escalando {len(SEGMENTATION_FEATURES)} features sobre {len(X):,} clientes...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("\n=== Validación del número de clusters ===")
    metrics = evaluate_k_range(X_scaled)
    print(metrics.round(3).to_string())

    if k is None:
        k = choose_k(metrics)
        print(f"\nK elegido automáticamente: {k} (codo de la curva de inercia; "
              f"DB decrece monótonamente y el silhouette es plano, así que "
              f"ninguno de los dos tiene óptimo interior)")
    else:
        print(f"\nK fijado por parámetro: {k}")

    print(f"Entrenando KMeans (K={k})...")
    kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10).fit(X_scaled)
    d['cluster'] = kmeans.labels_

    names = name_clusters(kmeans.cluster_centers_, SEGMENTATION_FEATURES)

    # Perfil en unidades originales: es lo que se muestra a Marketing.
    profile_cols = [
        'monto_facturado_prom', 'consumo_datos_gb_prom', 'ratio_consumo_gb',
        'antiguedad_meses', 'dias_mora_prom', 'n_reclamos',
        'n_actividad_canal', 'score_digital',
    ]
    profile_table = d.groupby('cluster')[profile_cols].mean()
    profile_table['n_clientes'] = d.groupby('cluster').size()
    profile_table['pct_base'] = profile_table['n_clientes'] / len(d) * 100
    profile_table['pct_elegible_mt'] = d.groupby('cluster')['elegible_mt'].mean() * 100
    profile_table['pct_ya_mt'] = d.groupby('cluster')['es_movistar_total'].mean() * 100
    profile_table['nombre'] = pd.Series(names)

    traits = build_cluster_traits(profile_table)

    print("\n=== Perfil de los segmentos comportamentales ===")
    print(profile_table.round(2).to_string())
    print("\n=== Rasgos para la política de negocio ===")
    for cid, t in traits.items():
        activos = [k_ for k_, v in t.items() if v] or ['(ninguno)']
        print(f"  {cid} · {names[cid]:<38} {', '.join(activos)}")

    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    joblib.dump(kmeans, os.path.join(ARTIFACTS_DIR, 'segmentation_model.pkl'))
    joblib.dump(scaler, os.path.join(ARTIFACTS_DIR, 'segmentation_scaler.pkl'))
    joblib.dump(SEGMENTATION_FEATURES, os.path.join(ARTIFACTS_DIR, 'segmentation_features.pkl'))
    joblib.dump(
        {
            'k': k,
            'names': names,
            'traits': traits,
            'profile_table': profile_table,
            'metrics': metrics,
        },
        os.path.join(ARTIFACTS_DIR, 'cluster_profiles.pkl'),
    )
    print(f"\nArtefactos guardados en {ARTIFACTS_DIR}/")

    return kmeans, scaler, profile_table


if __name__ == "__main__":
    train_segmentation()
