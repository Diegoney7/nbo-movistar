"""
Modelo de probabilidad de aceptación de oferta.

Nota metodológica importante — leer antes de tocar los hiperparámetros:

El techo de información de este dataset es bajo y está medido. La aceptación se
explica esencialmente por tres variables (`oferta_es_mt`, `meses_moroso`,
`n_reclamos`); el resto del catálogo de features es plano frente al target. Un
estimador oráculo construido con las tasas empíricas de esas tres variables
alcanza AUC ≈ 0.584 en el split temporal de test. Ese es el límite de Bayes
alcanzable, no una deficiencia del algoritmo: subir el AUC muy por encima de ese
valor solo sería posible sobreajustando o filtrando información del futuro.

Por eso el objetivo de este modelo no es maximizar AUC, sino ser el mejor
estimador CALIBRADO posible dentro del techo. La calibración es lo que de verdad
importa aguas abajo: el motor NBO rankea por `probabilidad × valor de negocio` y
le muestra al asesor un porcentaje explícito. Una probabilidad inflada distorsiona
el ranking y le miente al asesor.

train_acceptance_model() imprime la comparación contra el oráculo para que la
métrica se lea siempre en contexto.
"""
import os
import sys
import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import roc_auc_score, precision_score, recall_score, brier_score_loss

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_loader import DataLoader
from feature_engineering import build_features

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATA_DIR = os.path.join(BASE_DIR, 'data')
ARTIFACTS_DIR = os.path.join(BASE_DIR, 'models_artifacts')

RANDOM_STATE = 42

# Split temporal en tres tramos: los meses 1-3 entrenan, el 4 calibra (nunca visto
# por el booster) y los meses 5-6 son el test intacto.
MESES_TRAIN = [1, 2, 3]
MESES_CALIB = [4]

FEATURES = [
    # Cliente
    'antiguedad_meses', 'tiene_movil', 'tiene_hogar', 'tiene_internet_hogar',
    'monto_facturado_prom', 'es_usuario_app', 'consumo_datos_gb_prom',
    'uso_app_movistar_prom', 'dias_mora_prom', 'meses_moroso', 'n_reclamos',
    'n_actividad_canal',
    # Brecha convergente
    'servicios_mt_actuales', 'brecha_mt', 'elegible_mt', 'es_movistar_total',
    # Plan actual y ratios
    'plan_actual_gb', 'plan_actual_precio', 'ratio_consumo_gb', 'ratio_facturado_precio',
    # Oferta presentada
    'precio_mensual', 'ahorro_pct', 'gb_incluidos', 'oferta_es_mt', 'mt_match',
    # Categóricas
    'canal', 'tipo_cliente', 'edad_rango', 'ubicacion_departamento',
    'canal_mas_usado', 'segmento_objetivo', 'cluster_hogar', 'descripcion_bundle',
    'tipo_oferta',
]

# Se excluyen a propósito consumo_voz_min_prom y consumo_sms_prom: coeficiente de
# variación 0.20 y 0.14 (casi constantes) y aceptación plana frente al target.
# Solo aportan varianza al ajuste. monto_facturado_prom_6m también sale, por ser
# idéntica a monto_facturado_prom (r = 1.000).

CAT_FEATURES = [
    'canal', 'tipo_cliente', 'edad_rango', 'ubicacion_departamento',
    'canal_mas_usado', 'segmento_objetivo', 'cluster_hogar', 'descripcion_bundle',
    'tipo_oferta',
]

# Variables que gobiernan la aceptación en esta base; sostienen el oráculo.
ORACLE_KEYS = ['oferta_es_mt', 'meses_moroso', 'n_reclamos']


def _cast_categories(df, categorias):
    df = df.copy()
    for col in CAT_FEATURES:
        df[col] = pd.Categorical(df[col], categories=categorias[col])
    return df


def compute_bayes_ceiling(train, test):
    """
    Estimador oráculo: tasa empírica de aceptación por celda de las variables que
    realmente gobiernan el target, ajustada en train y aplicada a test.

    Da la referencia honesta de cuánto AUC es extraíble de estos datos.
    """
    tasas = train.groupby(ORACLE_KEYS)['target'].agg(['mean', 'size'])
    tasas = tasas[tasas['size'] >= 30]['mean']
    p = pd.Series(test.set_index(ORACLE_KEYS).index.map(tasas), index=test.index)
    p = p.fillna(train['target'].mean())
    return roc_auc_score(test['target'], p), brier_score_loss(test['target'], p)


def evaluar(nombre, y_true, p, umbral=0.5):
    pred = (p > umbral).astype(int)
    metricas = {
        'auc': roc_auc_score(y_true, p),
        'brier': brier_score_loss(y_true, p),
        'precision': precision_score(y_true, pred, zero_division=0),
        'recall': recall_score(y_true, pred, zero_division=0),
    }
    print(f"  {nombre:<26} AUC {metricas['auc']:.4f} | Brier {metricas['brier']:.4f} "
          f"| P {metricas['precision']:.3f} | R {metricas['recall']:.3f}")
    return metricas


def reporte_calibracion(y_true, p, n_bins=10):
    """Error de calibración esperado (ECE) y tabla de deciles observado vs. predicho."""
    df = pd.DataFrame({'y': np.asarray(y_true), 'p': np.asarray(p)})
    df['bin'] = pd.qcut(df['p'], n_bins, labels=False, duplicates='drop')
    tabla = df.groupby('bin').agg(
        predicho=('p', 'mean'), observado=('y', 'mean'), n=('y', 'size')
    )
    tabla['gap_pp'] = ((tabla['predicho'] - tabla['observado']) * 100).round(2)
    ece = (tabla['n'] / tabla['n'].sum() * (tabla['predicho'] - tabla['observado']).abs()).sum()
    return ece, tabla


def lift_comercial(y_true, p, top_pct=0.10):
    """
    Métrica de negocio: aceptación del top-N% rankeado por el modelo frente a la
    tasa base. Es lo que traduce el modelo a valor comercial cuando el AUC está
    limitado por los datos.
    """
    df = pd.DataFrame({'y': np.asarray(y_true), 'p': np.asarray(p)}).sort_values('p', ascending=False)
    corte = max(1, int(len(df) * top_pct))
    tasa_top = df.head(corte)['y'].mean()
    base = df['y'].mean()
    return tasa_top, base, tasa_top / base


def train_acceptance_model():
    print("Cargando datos y construyendo features...")
    loader = DataLoader(data_dir=DATA_DIR)
    clientes, ofertas, campanias = loader.load_all()
    train_full, test, _ = build_features(clientes, ofertas, campanias)

    train = train_full[train_full['mes_campania'].isin(MESES_TRAIN)].copy()
    calib = train_full[train_full['mes_campania'].isin(MESES_CALIB)].copy()

    print(f"  train {len(train):,} | calibración {len(calib):,} | test {len(test):,}")

    # Categorías compartidas: si cada split infiere las suyas, LightGBM recibe
    # codificaciones distintas entre entrenamiento e inferencia.
    categorias = {
        col: pd.Categorical(
            pd.concat([train_full[col], test[col]]).astype(str)
        ).categories
        for col in CAT_FEATURES
    }
    for col in CAT_FEATURES:
        for df in (train, calib, test, train_full):
            df[col] = df[col].astype(str)

    train = _cast_categories(train, categorias)
    calib = _cast_categories(calib, categorias)
    test = _cast_categories(test, categorias)
    train_full_c = _cast_categories(train_full, categorias)

    X_tr, y_tr = train[FEATURES], train['target']
    X_ca, y_ca = calib[FEATURES], calib['target']
    X_te, y_te = test[FEATURES], test['target']

    print("\n=== Techo de información del dataset ===")
    auc_oraculo, brier_oraculo = compute_bayes_ceiling(train_full_c, test)
    print(f"  Oráculo sobre {ORACLE_KEYS}")
    print(f"  AUC {auc_oraculo:.4f} | Brier {brier_oraculo:.4f}")
    print("  Es el límite de Bayes alcanzable: acercarse a él es el objetivo, superarlo")
    print("  de forma amplia indicaría fuga de información, no mérito del modelo.")

    print("\n=== Entrenamiento ===")
    # Sin class_weight='balanced': el target está en 37/63, no hay desbalance que
    # corregir, y reponderar destruye la calibración (era la causa de que el motor
    # mostrara 84% de probabilidad donde la tasa real es 69%).
    modelo = LGBMClassifier(
        n_estimators=2000,
        learning_rate=0.03,
        num_leaves=31,
        min_child_samples=200,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
        verbose=-1,
    )
    modelo.fit(
        X_tr, y_tr,
        eval_set=[(X_ca, y_ca)],
        eval_metric='auc',
        categorical_feature=CAT_FEATURES,
        callbacks=[early_stopping(100, verbose=False), log_evaluation(0)],
    )
    print(f"  Árboles usados (early stopping sobre el mes de calibración): {modelo.best_iteration_}")

    print("\n=== Resultados en test (meses 5-6, nunca vistos) ===")
    p_raw = modelo.predict_proba(X_te)[:, 1]
    m_raw = evaluar("LightGBM sin calibrar", y_te, p_raw)

    # Calibración isotónica sobre el mes 4, que el booster no usó para ajustar.
    # FrozenEstimator congela el booster ya entrenado: sklearn >= 1.6 reemplazó
    # con esto el antiguo cv='prefit'.
    calibrado = CalibratedClassifierCV(FrozenEstimator(modelo), method='isotonic')
    calibrado.fit(X_ca, y_ca)
    p_cal = calibrado.predict_proba(X_te)[:, 1]
    m_cal = evaluar("LightGBM calibrado", y_te, p_cal)

    print(f"  {'Oráculo (techo de Bayes)':<26} AUC {auc_oraculo:.4f} | Brier {brier_oraculo:.4f}")

    ece_raw, _ = reporte_calibracion(y_te, p_raw)
    ece_cal, tabla_cal = reporte_calibracion(y_te, p_cal)
    print(f"\n  Error de calibración (ECE): sin calibrar {ece_raw:.4f} → calibrado {ece_cal:.4f}")

    print("\n=== Calibración por decil de score (modelo final) ===")
    print(tabla_cal.round(4).to_string())

    print("\n=== Valor comercial: lift sobre targeting aleatorio ===")
    for pct in (0.05, 0.10, 0.20):
        tasa, base, lift = lift_comercial(y_te, p_cal, pct)
        print(f"  Top {int(pct*100):>2}% del ranking: {tasa*100:.2f}% de aceptación "
              f"vs {base*100:.2f}% base → lift ×{lift:.2f}")

    # Verificación puntual: la probabilidad media que el modelo asigna a ofertas MT
    # debe coincidir con la tasa real observada.
    mask_mt = test['oferta_es_mt'].astype(bool).to_numpy()
    print(f"\n  Ofertas MT en test — real {y_te[mask_mt].mean()*100:.2f}% | "
          f"predicho sin calibrar {p_raw[mask_mt].mean()*100:.2f}% | "
          f"calibrado {p_cal[mask_mt].mean()*100:.2f}%")

    print("\n=== Importancia de features (top 12) ===")
    imp = pd.Series(modelo.feature_importances_, index=FEATURES).sort_values(ascending=False)
    print(imp.head(12).to_string())

    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    joblib.dump(calibrado, os.path.join(ARTIFACTS_DIR, 'acceptance_model.pkl'))
    joblib.dump(modelo, os.path.join(ARTIFACTS_DIR, 'acceptance_model_raw.pkl'))
    joblib.dump(FEATURES, os.path.join(ARTIFACTS_DIR, 'model_features.pkl'))
    joblib.dump(CAT_FEATURES, os.path.join(ARTIFACTS_DIR, 'model_cat_features.pkl'))
    joblib.dump(categorias, os.path.join(ARTIFACTS_DIR, 'model_categories.pkl'))
    joblib.dump(
        {
            'auc': m_cal['auc'], 'brier': m_cal['brier'],
            'auc_sin_calibrar': m_raw['auc'], 'brier_sin_calibrar': m_raw['brier'],
            'precision': m_cal['precision'], 'recall': m_cal['recall'],
            'auc_oraculo': auc_oraculo, 'brier_oraculo': brier_oraculo,
            'ece': ece_cal, 'ece_sin_calibrar': ece_raw,
            'tabla_calibracion': tabla_cal,
        },
        os.path.join(ARTIFACTS_DIR, 'acceptance_metrics.pkl'),
    )
    print(f"\nArtefactos guardados en {ARTIFACTS_DIR}/")

    return calibrado


if __name__ == "__main__":
    train_acceptance_model()
