import os
import sys
import joblib
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_loader import DataLoader
from feature_engineering import build_features

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATA_DIR = os.path.join(BASE_DIR, 'data')
ARTIFACTS_DIR = os.path.join(BASE_DIR, 'models_artifacts')


def train_rebate_model():
    print("Loading data for Rebate Model...")
    loader = DataLoader(data_dir=DATA_DIR)
    clientes, ofertas, campanias = loader.load_all()
    train, test, _ = build_features(clientes, ofertas, campanias)
    
    # Reload campanias to get the original motivo_rechazo which was dropped in build_features
    # We only train on rejected offers
    df_camp_valid = campanias[campanias['resultado'] == 'rechazada'][['ofrecimiento_id', 'motivo_rechazo']].copy()
    
    train = train.merge(df_camp_valid, on='ofrecimiento_id', how='inner')
    test = test.merge(df_camp_valid, on='ofrecimiento_id', how='inner')
    
    # Drop rows with null motivo_rechazo (if any)
    train = train.dropna(subset=['motivo_rechazo'])
    test = test.dropna(subset=['motivo_rechazo'])
    
    features = joblib.load(os.path.join(ARTIFACTS_DIR, 'model_features.pkl'))
    cat_features = joblib.load(os.path.join(ARTIFACTS_DIR, 'model_cat_features.pkl'))
    categorias = joblib.load(os.path.join(ARTIFACTS_DIR, 'model_categories.pkl'))

    # Mismas categorías que el modelo de aceptación, para que la app pueda
    # preparar una sola vez la fila y consultar ambos modelos.
    for col in cat_features:
        train[col] = pd.Categorical(train[col].astype(str), categories=categorias[col])
        test[col] = pd.Categorical(test[col].astype(str), categories=categorias[col])


    X_train = train[features]
    y_train = train['motivo_rechazo'].astype('category')
    X_test = test[features]
    y_test = test['motivo_rechazo'].astype('category')
    
    print("Training Rebate Model (Multiclass)...")
    # Sin class_weight='balanced': el motivo de rechazo sigue una multinomial fija
    # e independiente de las features (verificado: la distribución es idéntica por
    # tipo de oferta, canal, mora y facturación). Reponderar las clases empujaba al
    # modelo a repartir predicciones entre motivos improbables y lo dejaba POR
    # DEBAJO del azar. Sin reponderar, converge al prior, que es lo correcto
    # cuando no hay señal.
    model = LGBMClassifier(
        n_estimators=50,
        learning_rate=0.05,
        max_depth=5,
        random_state=42,
        verbose=-1,
    )

    model.fit(X_train, y_train, categorical_feature=cat_features)

    print("Evaluating Rebate Model...")
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    # Baselines honestos: el modelo debe compararse contra "siempre la clase
    # mayoritaria", no contra el azar uniforme.
    prior = y_train.value_counts(normalize=True)
    acc_mayoritaria = (y_test == prior.idxmax()).mean()
    print(f"  Accuracy modelo            : {acc:.4f}")
    print(f"  Accuracy clase mayoritaria : {acc_mayoritaria:.4f}  ('{prior.idxmax()}')")
    print(f"  Accuracy azar uniforme     : {1/len(prior):.4f}")
    print("\n  Prior de motivos de rechazo (%):")
    print((prior * 100).round(2).to_string())
    print("\n  Lectura: el motivo de rechazo NO es predecible en estos datos — la")
    print("  distribución es la misma para cualquier oferta, canal o perfil. El")
    print("  modelo se conserva como estimador del prior; la contraoferta se decide")
    print("  con el motivo que el asesor escucha realmente (ver src/rebate_policy.py).")

    joblib.dump(prior, os.path.join(ARTIFACTS_DIR, 'rebate_prior.pkl'))

    joblib.dump(model, os.path.join(ARTIFACTS_DIR, 'rebate_model.pkl'))
    # Save the classes explicitly for prediction
    joblib.dump(model.classes_, os.path.join(ARTIFACTS_DIR, 'rebate_classes.pkl'))
    print("Model saved to models_artifacts/rebate_model.pkl")
    return model

if __name__ == "__main__":
    train_rebate_model()
