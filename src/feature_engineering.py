import pandas as pd
import numpy as np

def build_features(df_clientes, df_ofertas, df_campanias):
    """
    Construye el dataset de entrenamiento y test con features derivadas.
    Evita leakage omitiendo variables post-evento.
    """
    # 1. Preparar features de cliente
    df_feat_clientes = df_clientes.copy()
    
    # Brecha a MT: Qué le falta al cliente para ser MT (si es postpago, necesita movil + internet hogar)
    # Una métrica simple es la suma de los servicios clave que tiene. MT requiere 2 (movil + int. hogar)
    df_feat_clientes['servicios_mt_actuales'] = df_feat_clientes['tiene_movil'].astype(int) + df_feat_clientes['tiene_internet_hogar'].astype(int)
    df_feat_clientes['brecha_mt'] = 2 - df_feat_clientes['servicios_mt_actuales']
    
    # Merge con los detalles del plan actual para ratios
    # df_ofertas tiene la info de gb_incluidos y precio de cada plan
    planes_info = df_ofertas[['oferta_id', 'gb_incluidos', 'precio_mensual']].rename(
        columns={'oferta_id': 'plan_actual_id', 'gb_incluidos': 'plan_actual_gb', 'precio_mensual': 'plan_actual_precio'}
    )
    df_feat_clientes = df_feat_clientes.merge(planes_info, on='plan_actual_id', how='left')
    
    # Ratios de consumo vs plan actual
    # Si gb_incluidos es 9999 (ilimitado), el ratio será pequeño. 
    df_feat_clientes['ratio_consumo_gb'] = np.where(
        df_feat_clientes['plan_actual_gb'] > 0, 
        df_feat_clientes['consumo_datos_gb_prom'] / df_feat_clientes['plan_actual_gb'], 
        0
    )
    df_feat_clientes['ratio_facturado_precio'] = np.where(
        df_feat_clientes['plan_actual_precio'] > 0,
        df_feat_clientes['monto_facturado_prom'] / df_feat_clientes['plan_actual_precio'],
        0
    )
    
    # 2. Filtrar campañas para el modelo (Target = aceptada vs rechazada)
    # Excluimos "pendiente" (no contactados) para el modelo de probabilidad de aceptación dado el contacto
    df_campanias_valid = df_campanias[df_campanias['resultado'] != 'pendiente'].copy()
    
    # Crear target binario
    df_campanias_valid['target'] = (df_campanias_valid['resultado'] == 'aceptada').astype(int)
    
    # 3. Cruzar Campañas con Clientes
    # OJO: Excluimos motivo_rechazo, es_rebate, contactabilidad, medio_probatorio (Leakage)
    drop_leakage_cols = ['motivo_rechazo', 'es_rebate', 'contactabilidad', 'medio_probatorio', 'resultado']
    df_dataset = df_campanias_valid.drop(columns=drop_leakage_cols)
    
    # Unir con clientes
    # Evitamos duplicar columnas que ya están en campañas (tipo_cliente, antiguedad_meses, etc)
    cols_to_drop_from_client = ['tipo_cliente', 'antiguedad_meses', 'elegible_mt', 'es_movistar_total']
    df_feat_clientes_to_merge = df_feat_clientes.drop(columns=cols_to_drop_from_client, errors='ignore')
    
    df_dataset = df_dataset.merge(df_feat_clientes_to_merge, on='cliente_id', how='left')
    
    # Unir con características de la oferta propuesta
    # (Para que el modelo sepa qué se está ofreciendo)
    ofertas_info = df_ofertas.rename(columns={'oferta_id': 'oferta_propuesta_id'})
    # Renombramos oferta_id a oferta_propuesta_id en el dataset para evitar confusiones
    df_dataset = df_dataset.rename(columns={'oferta_id': 'oferta_propuesta_id'})
    
    # quitamos columnas de oferta ya presentes
    cols_to_drop_from_offer = ['nombre_oferta', 'tipo_oferta', 'es_movistar_total']
    ofertas_info = ofertas_info.drop(columns=cols_to_drop_from_offer, errors='ignore')
    
    df_dataset = df_dataset.merge(ofertas_info, on='oferta_propuesta_id', how='left')
    
    # 3b. Interacción explícita cliente × oferta.
    # oferta_es_mt es la variable con mayor poder predictivo del dataset (34% vs
    # 70% de aceptación) y solo llegaba al modelo indirectamente, embebida en la
    # categórica tipo_oferta. Se expone como booleano, junto con el cruce contra
    # la elegibilidad del cliente.
    df_dataset['mt_match'] = (
        df_dataset['elegible_mt'].astype(bool) & df_dataset['oferta_es_mt'].astype(bool)
    ).astype(int)

    # 4. Split Temporal
    # Train: Meses 1, 2, 3, 4 (Enero a Abril)
    # Test: Meses 5, 6 (Mayo a Junio)
    df_dataset['mes_campania'] = df_dataset['fecha'].dt.month
    
    df_train = df_dataset[df_dataset['mes_campania'] <= 4].copy()
    df_test = df_dataset[df_dataset['mes_campania'] >= 5].copy()
    
    # Drop IDs no predictivos (ofrecimiento_id, fecha, etc.)
    # Opcionalmente guardamos cliente_id para evaluación
    
    return df_train, df_test, df_dataset

if __name__ == "__main__":
    from data_loader import DataLoader
    loader = DataLoader()
    clientes, ofertas, campanias = loader.load_all()
    
    train, test, full = build_features(clientes, ofertas, campanias)
    print(f"Train shape: {train.shape}, Test shape: {test.shape}")
    print(f"Features: {list(train.columns)}")
