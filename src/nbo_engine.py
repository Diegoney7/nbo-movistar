import os
import sys
import pandas as pd
import numpy as np
import joblib

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from data_loader import DataLoader
from models.channel_recommender import ChannelRecommender
from explainability import Explainer
from segment_policy import (
    SegmentAssigner, apply_segment_policy, get_playbook, is_offer_eligible,
    get_momento_sugerido,
)
from rebate_policy import elegir_rebate
from value_model import resumen_beneficio

class NBOEngine:
    def __init__(self, data_dir='../data', artifacts_dir='../models_artifacts'):
        self.loader = DataLoader(data_dir=data_dir)
        # El histórico de campañas (300k filas) no se necesita para recomendar:
        # se carga bajo demanda en el funnel y en la vista de segmentos, y solo
        # con las columnas de cada caso.
        self.clientes = self.loader.load_clientes()
        self.ofertas = self.loader.load_ofertas()
        
        # We need the planes info to compute ratios just like in training
        self.planes_info = self.ofertas[['oferta_id', 'gb_incluidos', 'precio_mensual']].rename(
            columns={'oferta_id': 'plan_actual_id', 'gb_incluidos': 'plan_actual_gb', 'precio_mensual': 'plan_actual_precio'}
        )
        self.clientes_feat = self.clientes.merge(self.planes_info, on='plan_actual_id', how='left')
        self.clientes_feat['servicios_mt_actuales'] = self.clientes_feat['tiene_movil'].astype(int) + self.clientes_feat['tiene_internet_hogar'].astype(int)
        self.clientes_feat['brecha_mt'] = 2 - self.clientes_feat['servicios_mt_actuales']
        
        self.clientes_feat['ratio_consumo_gb'] = np.where(
            self.clientes_feat['plan_actual_gb'] > 0, 
            self.clientes_feat['consumo_datos_gb_prom'] / self.clientes_feat['plan_actual_gb'], 0
        )
        self.clientes_feat['ratio_facturado_precio'] = np.where(
            self.clientes_feat['plan_actual_precio'] > 0,
            self.clientes_feat['monto_facturado_prom'] / self.clientes_feat['plan_actual_precio'], 0
        )
        
        # Segmentación de dos ejes: macro-segmento MT (regla) × comportamental (KMeans).
        # Alimenta tanto la ficha del asesor como la política de ranking.
        self.segmenter = SegmentAssigner(artifacts_dir)
        self.clientes_feat = self.segmenter.assign(self.clientes_feat, self.ofertas)

        # Contrato de features y categorías, tal como quedaron en el entrenamiento.
        self.model_features = joblib.load(os.path.join(artifacts_dir, 'model_features.pkl'))
        self.cat_features = joblib.load(os.path.join(artifacts_dir, 'model_cat_features.pkl'))
        self.categories = joblib.load(os.path.join(artifacts_dir, 'model_categories.pkl'))

        self.channel_rec = ChannelRecommender(
            model_path=os.path.join(artifacts_dir, 'acceptance_model.pkl'),
            categories=self.categories,
        )
        # SHAP corre sobre el booster crudo (ver src/explainability.py).
        self.explainer = Explainer(model_path=os.path.join(artifacts_dir, 'acceptance_model_raw.pkl'))

    def get_client_segment(self, cliente_id: str) -> dict:
        """Segmento del cliente en los dos ejes, con su playbook para el asesor."""
        cliente_df = self.clientes_feat[self.clientes_feat['cliente_id'] == cliente_id]
        if cliente_df.empty:
            return {}

        cliente_row = cliente_df.iloc[0]
        traits = self.segmenter.get_traits(cliente_row['cluster'])
        playbook = get_playbook(
            cliente_row['macro_segmento'],
            cliente_row['segmento_comportamental'],
            traits,
        )
        playbook['label'] = cliente_row['segmento_label']
        playbook['traits'] = traits
        return playbook

    def get_rebate(self, recomendaciones: list, motivo: str, idx_rechazada: int = 0) -> dict:
        """
        Contraoferta ante un rechazo, dado el motivo que el asesor escuchó.

        No se predice el motivo: en estos datos es una multinomial fija e
        impredecible, y en la interacción real el asesor ya lo tiene. Ver
        src/rebate_policy.py.
        """
        if not recomendaciones:
            return {}
        rechazada = recomendaciones[idx_rechazada]
        alternativas = [r for i, r in enumerate(recomendaciones) if i != idx_rechazada]
        return elegir_rebate(motivo, rechazada, alternativas)

    def get_funnel_e2e(self) -> dict:
        """
        Funnel end-to-end del ofrecimiento, desde la clasificación del cliente
        hasta el resultado de venta, con los medios probatorios que respaldan
        cada contacto.

        Es el reporte de trazabilidad que pide el desafío: permite responder si un
        producto fue efectivamente ofrecido y con qué evidencia.
        """
        camp = self.loader.load_campanias(usecols=[
            'ofrecimiento_id', 'cliente_id', 'canal', 'resultado', 'es_rebate',
            'contactabilidad', 'medio_probatorio', 'oferta_es_mt',
        ])
        camp = camp.merge(
            self.clientes_feat[['cliente_id', 'macro_segmento', 'segmento_comportamental']],
            on='cliente_id', how='left',
        )

        contactados = camp[camp['contactabilidad'] == 'contactado']
        con_resultado = camp[camp['resultado'] != 'pendiente']
        aceptados = camp[camp['resultado'] == 'aceptada']
        rechazados = camp[camp['resultado'] == 'rechazada']
        rebates = camp[camp['es_rebate']]
        rebates_aceptados = rebates[rebates['resultado'] == 'aceptada']

        # El tramo de contactabilidad y el de "resultado conocido" son el mismo
        # conjunto: sin contacto efectivo no hay resultado. Se muestra una sola
        # etapa en vez de un escalón artificial al 100%.
        etapas = pd.DataFrame([
            ('1. Ofrecimientos realizados', len(camp)),
            ('2. Contacto efectivo (con medio probatorio)', len(contactados)),
            ('3. Ventas cerradas', len(aceptados)),
        ], columns=['etapa', 'volumen'])
        etapas['conversion_vs_anterior'] = (
            etapas['volumen'] / etapas['volumen'].shift(1) * 100
        ).round(2)

        # Medios probatorios: la evidencia de que el ofrecimiento ocurrió.
        probatorios = camp.groupby('medio_probatorio').agg(
            ofrecimientos=('ofrecimiento_id', 'size'),
            aceptados=('resultado', lambda s: (s == 'aceptada').sum()),
        )
        probatorios['tasa_cierre'] = (
            probatorios['aceptados'] / probatorios['ofrecimientos'] * 100
        ).round(2)
        probatorios['cobertura_pct'] = (
            probatorios['ofrecimientos'] / len(camp) * 100
        ).round(2)

        probatorio_x_canal = pd.crosstab(camp['canal'], camp['medio_probatorio'])

        # Funnel desglosado por macro-segmento: dónde se pierde y dónde se cierra.
        por_segmento = camp.groupby('macro_segmento').agg(
            ofrecimientos=('ofrecimiento_id', 'size'),
            contactados=('contactabilidad', lambda s: (s == 'contactado').sum()),
            aceptados=('resultado', lambda s: (s == 'aceptada').sum()),
        )
        por_segmento['tasa_contacto'] = (
            por_segmento['contactados'] / por_segmento['ofrecimientos'] * 100
        ).round(2)
        por_segmento['tasa_cierre'] = (
            por_segmento['aceptados'] / por_segmento['contactados'] * 100
        ).round(2)
        por_segmento = por_segmento.sort_values('tasa_cierre', ascending=False)

        # Ventas MT: la meta de negocio del desafío.
        mt_ofrecidas = camp[camp['oferta_es_mt']]
        mt_vendidas = mt_ofrecidas[mt_ofrecidas['resultado'] == 'aceptada']

        return {
            'etapas': etapas,
            'base': {
                'clientes_clasificados': len(self.clientes_feat),
                'ofrecimientos_por_cliente': len(camp) / len(self.clientes_feat),
            },
            'medios_probatorios': probatorios,
            'probatorio_x_canal': probatorio_x_canal,
            'por_segmento': por_segmento,
            'rebates': {
                'total': len(rebates),
                'aceptados': len(rebates_aceptados),
                'tasa': (len(rebates_aceptados) / len(rebates) * 100) if len(rebates) else 0.0,
                'tasa_sin_rebate': (
                    camp[~camp['es_rebate'] & (camp['resultado'] != 'pendiente')]['resultado']
                    .eq('aceptada').mean() * 100
                ),
            },
            'mt': {
                'ofrecidas': len(mt_ofrecidas),
                'vendidas': len(mt_vendidas),
                'tasa': (len(mt_vendidas) / len(mt_ofrecidas) * 100) if len(mt_ofrecidas) else 0.0,
                'pct_de_ventas_totales': (len(mt_vendidas) / len(aceptados) * 100) if len(aceptados) else 0.0,
            },
            'rechazados': len(rechazados),
        }

    def get_segment_overview(self) -> dict:
        """
        Vista agregada de la segmentación para Producto y Marketing.

        Cruza los dos ejes con la aceptación histórica real de
        historial_campanias.csv, para que los segmentos no se lean solo como
        descripción sino como potencial comercial medido.
        """
        campanias = self.loader.load_campanias(
            usecols=['cliente_id', 'resultado', 'oferta_es_mt'])
        contactadas = campanias[campanias['resultado'] != 'pendiente'].copy()
        contactadas['acepta'] = (contactadas['resultado'] == 'aceptada').astype(int)

        cruce = contactadas.merge(
            self.clientes_feat[['cliente_id', 'segmento_comportamental', 'macro_segmento']],
            on='cliente_id', how='left',
        )

        def _tasa(df, col):
            t = df.groupby(col)['acepta'].agg(['mean', 'count'])
            t.columns = ['tasa_aceptacion', 'ofrecimientos']
            t['tasa_aceptacion'] = (t['tasa_aceptacion'] * 100).round(2)
            return t.sort_values('tasa_aceptacion', ascending=False)

        # Aceptación de ofertas MT: es la métrica que importa para la meta de negocio.
        solo_mt = cruce[cruce['oferta_es_mt']]
        mt_por_segmento = solo_mt.groupby(['macro_segmento', 'segmento_comportamental'])['acepta'].agg(['mean', 'count'])
        mt_por_segmento.columns = ['tasa_aceptacion_mt', 'ofrecimientos_mt']
        mt_por_segmento['tasa_aceptacion_mt'] = (mt_por_segmento['tasa_aceptacion_mt'] * 100).round(2)

        base = self.clientes_feat.groupby('macro_segmento').size().rename('n_clientes')

        return {
            'perfil_clusters': self.segmenter.profile_table,
            'metricas_k': self.segmenter.metrics,
            'k': self.segmenter.k,
            'tamano_macro': base,
            'aceptacion_comportamental': _tasa(cruce, 'segmento_comportamental'),
            'aceptacion_macro': _tasa(cruce, 'macro_segmento'),
            'aceptacion_mt_cruce': mt_por_segmento[mt_por_segmento['ofrecimientos_mt'] > 50]
                                    .sort_values('tasa_aceptacion_mt', ascending=False),
        }

    def get_recommendations(self, cliente_id: str, top_n: int = 3) -> list:
        # Find client
        cliente_df = self.clientes_feat[self.clientes_feat['cliente_id'] == cliente_id]
        if cliente_df.empty:
            return []
            
        cliente_row = cliente_df.iloc[0].to_dict()
        macro = cliente_row['macro_segmento']
        traits = self.segmenter.get_traits(cliente_row['cluster'])

        # Generate candidates (cross join with all offers)
        # Avoid offering the same plan they already have
        candidates = self.ofertas[self.ofertas['oferta_id'] != cliente_row['plan_actual_id']].copy()

        # Descartar lo que el cliente no puede contratar según su macro-segmento.
        candidates = candidates[
            candidates.apply(lambda o: is_offer_eligible(macro, o), axis=1)
        ]


        # Prepare feature grid
        grid = []
        for _, offer in candidates.iterrows():
            row = cliente_row.copy()
            # Add offer features (as expected by model)
            row['precio_mensual'] = offer['precio_mensual']
            row['ahorro_pct'] = offer['ahorro_pct']
            row['gb_incluidos'] = offer['gb_incluidos']
            row['segmento_objetivo'] = offer['segmento_objetivo']
            row['cluster_hogar'] = offer['cluster_hogar']
            row['descripcion_bundle'] = offer['descripcion_bundle']
            row['tipo_oferta'] = offer['tipo_oferta']
            row['oferta_id'] = offer['oferta_id']
            row['nombre_oferta'] = offer['nombre_oferta']
            # Variable de mayor poder predictivo del modelo, más su cruce con la
            # elegibilidad del cliente (ver src/models/acceptance_model.py).
            row['oferta_es_mt'] = bool(offer['es_movistar_total'])
            row['mt_match'] = int(bool(cliente_row['elegible_mt']) and bool(offer['es_movistar_total']))


            grid.append(row)
            
        grid_df = pd.DataFrame(grid)
        
        # We need a dummy 'canal' just to evaluate base probability (ChannelRecommender will optimize this later)
        # But wait, ChannelRecommender requires the dataframe. Let's just pass the row to ChannelRecommender.
        
        recommendations = []
        for idx, row_series in grid_df.iterrows():
            # Create a 1-row DataFrame for the model
            eval_df = pd.DataFrame([row_series])
            eval_df['canal'] = 'Digital' # Dummy channel for indexing, channel_rec will overwrite it
            
            # Use ChannelRecommender to find best channel and its probability
            channel_info = self.channel_rec.recommend(
                eval_df[self.model_features],
                canal_habitual=cliente_row.get('canal_mas_usado'),
            )
            max_prob = channel_info['max_prob']
            best_channel = channel_info['best_channel']
            
            # Score base: probabilidad de aceptación × valor de negocio.
            business_value = row_series['precio_mensual']
            is_mt = (row_series['tipo_oferta'] == 'movistar_total')
            score = max_prob * business_value

            # La política de segmento modula el ranking (empuje MT diferenciado,
            # camino a MT y freno ético de upsell). Ver src/segment_policy.py.
            score, reglas = apply_segment_policy(
                score, cliente_row, row_series, macro, traits
            )

            recommendations.append({
                'oferta_id': row_series['oferta_id'],
                'nombre_oferta': row_series['nombre_oferta'],
                'probabilidad': float(max_prob),
                'score_negocio': float(score),
                'canal_sugerido': best_channel,
                'precio_mensual': float(row_series['precio_mensual']),
                'es_mt': is_mt,
                'tipo_oferta': row_series['tipo_oferta'],
                # Necesarias para calcular el beneficio al cliente (src/value_model.py).
                'ahorro_pct': float(row_series['ahorro_pct'] or 0),
                'gb_incluidos': float(row_series['gb_incluidos'] or 0),
                'descripcion_bundle': row_series['descripcion_bundle'],
                'canal_decidido_por_modelo': channel_info['decidido_por_modelo'],
                'canal_criterio': channel_info['criterio'],
                'reglas_segmento': reglas,
                # Store the DF row for explainability
                '_eval_df': eval_df
            })
            
        # Sort by business score
        recommendations.sort(key=lambda x: x['score_negocio'], reverse=True)
        top_recs = recommendations[:top_n]
        
        # Beneficio esperado y momento sugerido: entregable mínimo 5 y 6.
        for rec in top_recs:
            rec['beneficio'] = resumen_beneficio(rec, cliente_row, rec['probabilidad'], macro)
            rec['momento'] = get_momento_sugerido(cliente_row, rec['canal_sugerido'], traits)

        # Enrich with explainability
        for rec in top_recs:
            eval_df = rec['_eval_df']
            eval_df['canal'] = rec['canal_sugerido'] # Fix channel to the suggested one
            # Mismas categorías del entrenamiento: inferirlas de una sola fila
            # produciría códigos que no corresponden a los del modelo.
            for col in self.cat_features:
                eval_df[col] = pd.Categorical(
                    eval_df[col].astype(str), categories=self.categories[col]
                )

            # Filter exactly the features needed by SHAP
            X_explain = eval_df[self.model_features]
            reasons = self.explainer.explain(X_explain)
            rec['motivos'] = reasons
            
        return top_recs

if __name__ == "__main__":
    engine = NBOEngine()
    # Test with a random client
    client_id = engine.clientes.iloc[0]['cliente_id']
    print(f"Testing NBO for client {client_id}:")
    recs = engine.get_recommendations(client_id)
    for r in recs:
        print(f"- {r['nombre_oferta']} | Prob: {r['probabilidad']:.2%} | Canal: {r['canal_sugerido']}")
        print(f"  Motivos: {r['motivos']}")
