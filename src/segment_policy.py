"""
Política de negocio de la segmentación (eje A + reglas de ranking).

Este módulo concentra todas las decisiones comerciales para que sean auditables
en un solo lugar, en vez de quedar dispersas como números mágicos dentro del
motor. Las reglas se apoyan en RASGOS del segmento (es_riesgo, plan_al_limite,
...) y no en números de cluster, porque los ids de KMeans cambian entre corridas.

Los multiplicadores están calibrados sobre tasas de aceptación medidas en
historial_campanias.csv (300k ofrecimientos) — ver los comentarios de MT_BOOST.
"""
import os
import joblib
import pandas as pd

# --- Eje A: macro-segmento de brecha convergente (regla de negocio) ---------
# La elegibilidad MT es determinística y NO es descubrible por clustering: los
# clientes elegibles tienen un perfil de comportamiento idéntico al resto de la
# base móvil (facturación ~70, ~31 GB). Por eso este eje va por reglas.

MACRO_YA_MT = 'YA_MT'
MACRO_ELEGIBLE_MT = 'ELEGIBLE_MT'
MACRO_BRECHA_HOGAR = 'BRECHA_HOGAR'
MACRO_BRECHA_MOVIL = 'BRECHA_MOVIL'
MACRO_NO_CONVERGENTE = 'NO_CONVERGENTE'

MACRO_INFO = {
    MACRO_YA_MT: {
        'etiqueta': 'Ya es Movistar Total',
        'accion': 'Retener y hacer upsell dentro del portafolio MT',
        'prioridad': 2,
    },
    MACRO_ELEGIBLE_MT: {
        'etiqueta': 'Elegible Movistar Total',
        'accion': 'Prioridad máxima: empujar MT (58.9% de aceptación medida)',
        'prioridad': 1,
    },
    MACRO_BRECHA_HOGAR: {
        'etiqueta': 'Le falta hogar para MT',
        'accion': 'Vender internet hogar — lo convierte en elegible MT',
        'prioridad': 3,
    },
    MACRO_BRECHA_MOVIL: {
        'etiqueta': 'Le falta móvil postpago para MT',
        'accion': 'Vender móvil o migrar a postpago — lo convierte en elegible MT',
        'prioridad': 3,
    },
    MACRO_NO_CONVERGENTE: {
        'etiqueta': 'Sin ruta directa a MT',
        'accion': 'Migrar a postpago antes de plantear convergencia',
        'prioridad': 4,
    },
}


def get_macro_segment(cliente) -> str:
    """Clasifica al cliente en el eje de brecha convergente."""
    if cliente.get('es_movistar_total'):
        return MACRO_YA_MT
    if cliente.get('elegible_mt'):
        return MACRO_ELEGIBLE_MT

    tiene_movil = bool(cliente.get('tiene_movil'))
    es_postpago = cliente.get('tipo_cliente') == 'postpago'
    tiene_internet = bool(cliente.get('tiene_internet_hogar'))

    if tiene_movil and es_postpago and not tiene_internet:
        return MACRO_BRECHA_HOGAR
    if tiene_internet:
        return MACRO_BRECHA_MOVIL
    return MACRO_NO_CONVERGENTE


# --- Filtro de elegibilidad -------------------------------------------------

# Los macro-segmentos que pueden contratar Movistar Total. MT exige móvil
# postpago + internet hogar: a quien no cumple no se le puede ofrecer, y a
# ELEGIBLE_MT/YA_MT sí (upsell dentro del portafolio en el segundo caso).
MACROS_PUEDEN_MT = {MACRO_ELEGIBLE_MT, MACRO_YA_MT}


def is_offer_eligible(macro, oferta) -> bool:
    """
    Filtra ofertas que el cliente no puede contratar.

    Sin este filtro las MT copan el top-N de cualquier cliente: son las ofertas
    más caras del catálogo y el score de negocio es proporcional al precio, así
    que ganaban el ranking incluso en segmentos sin ruta a MT. El asesor
    terminaba ofreciendo un producto no contratable.
    """
    if oferta.get('tipo_oferta') == 'movistar_total':
        return macro in MACROS_PUEDEN_MT
    return True


# --- Multiplicadores de ranking --------------------------------------------
# Calibrados sobre la aceptación real de ofertas MT dentro de ELEGIBLE_MT:
#   Plan al Límite 73.0% | Plan Holgado 71.0% | Alta Fricción 64.8% | Reciente 64.6%
# El rasgo de riesgo recibe el boost más bajo por doble motivo: menor aceptación
# medida y exposición financiera (vender convergencia a quien ya no paga).
MT_BOOST_BASE = 1.6
MT_BOOST_PLAN_AL_LIMITE = 1.8
MT_BOOST_RIESGO = 1.2
MT_BOOST_FRICCION = 1.3

# Camino hacia MT: al cliente al que solo le falta una pata, se le prioriza
# justamente la que le falta.
BOOST_CAMINO_MT = 1.3

# Freno de upsell: no ofrecer a un cliente en mora un producto muy por encima de
# lo que hoy paga. Cubre el requisito de uso ético del desafío ("mejorar la
# conversión sin afectar negativamente la experiencia del cliente").
UMBRAL_UPSELL_RIESGO = 1.3
PENALIZACION_UPSELL_RIESGO = 0.6


def apply_segment_policy(base_score, cliente, oferta, macro, traits):
    """
    Modula el score de ranking según el segmento.

    Devuelve (score_ajustado, reglas_aplicadas) — las reglas se devuelven para
    poder mostrarlas en la UI y auditar por qué una oferta subió o bajó.
    """
    score = base_score
    reglas = []

    es_oferta_mt = oferta.get('tipo_oferta') == 'movistar_total'
    tipo_oferta = oferta.get('tipo_oferta')
    precio = float(oferta.get('precio_mensual') or 0)
    facturacion = float(cliente.get('monto_facturado_prom') or 0)

    # 1. Empuje diferenciado de MT sobre el segmento prioritario.
    if es_oferta_mt and macro == MACRO_ELEGIBLE_MT:
        if traits.get('es_riesgo'):
            factor, motivo = MT_BOOST_RIESGO, 'MT con boost reducido: segmento con mora'
        elif traits.get('alta_friccion'):
            factor, motivo = MT_BOOST_FRICCION, 'MT con boost reducido: historial de reclamos'
        elif traits.get('plan_al_limite'):
            factor, motivo = MT_BOOST_PLAN_AL_LIMITE, 'MT con boost máximo: plan al límite (73% de aceptación medida)'
        else:
            factor, motivo = MT_BOOST_BASE, 'MT priorizado: cliente elegible'
        score *= factor
        reglas.append(f"{motivo} (×{factor})")

    # 2. Camino a MT: priorizar la pata que le falta al cliente.
    elif macro == MACRO_BRECHA_HOGAR and tipo_oferta == 'plan_hogar':
        score *= BOOST_CAMINO_MT
        reglas.append(f"Hogar priorizado: lo vuelve elegible MT (×{BOOST_CAMINO_MT})")
    elif macro == MACRO_BRECHA_MOVIL and tipo_oferta == 'plan_movil':
        score *= BOOST_CAMINO_MT
        reglas.append(f"Móvil priorizado: lo vuelve elegible MT (×{BOOST_CAMINO_MT})")

    # 3. Freno ético de upsell sobre clientes en mora.
    if traits.get('es_riesgo') and facturacion > 0 and precio > facturacion * UMBRAL_UPSELL_RIESGO:
        score *= PENALIZACION_UPSELL_RIESGO
        reglas.append(
            f"Upsell frenado: cliente con mora y oferta {precio / facturacion:.1f}× "
            f"su facturación actual (×{PENALIZACION_UPSELL_RIESGO})"
        )

    return score, reglas


# --- Playbook para el asesor ------------------------------------------------

PLAYBOOK_RASGOS = {
    'plan_al_limite': 'Está al límite de los GB de su plan: abrir con el argumento de más datos.',
    'es_riesgo': 'Cliente con historial de mora: liderar con ahorro, evitar tickets altos.',
    'alta_friccion': 'Tiene reclamos registrados: reconocer el problema antes de ofrecer.',
    'es_alto_valor': 'Cliente de alto valor: enfatizar beneficios premium y prioridad de atención.',
    'es_digital': 'Perfil digital: puede cerrar por app o web sin pasar por tienda.',
}


def get_playbook(macro, cluster_name, traits):
    """Guion accionable para el asesor: qué hacer y con qué tono."""
    info = MACRO_INFO[macro]
    argumentos = [texto for rasgo, texto in PLAYBOOK_RASGOS.items() if traits.get(rasgo)]
    if not argumentos:
        argumentos = ['Perfil sin rasgos extremos: usar el argumento estándar de la oferta.']

    return {
        'macro': macro,
        'macro_etiqueta': info['etiqueta'],
        'accion': info['accion'],
        'prioridad': info['prioridad'],
        'segmento_comportamental': cluster_name,
        'argumentos': argumentos,
    }


def get_momento_sugerido(cliente, canal_sugerido, traits):
    """
    Cuándo presentar la oferta.

    Es una REGLA DE NEGOCIO, no una predicción, y así se declara en la interfaz.
    En estos datos el momento no tiene señal alguna: la aceptación es 37.15%–37.61%
    según el día de la semana, plana por mes, y por día del mes es literalmente
    constante (37.47%). Entrenar un modelo sobre eso sería presentar ruido como
    recomendación. La heurística se apoya en el canal y en el ciclo de cobranza,
    que es donde el negocio sí tiene criterio.
    """
    if canal_sugerido == 'Tienda':
        return {
            'momento': 'Ahora, durante esta visita',
            'detalle': 'El cliente ya está presente: es el contacto de mayor costo y el de '
                       'mayor cierre. No diferir.',
        }

    if traits.get('es_riesgo'):
        return {
            'momento': 'Después de su fecha de pago',
            'detalle': 'Segmento con mora: contactar recién pasada la fecha de corte evita '
                       'proponer un aumento de ticket cuando la deuda está abierta.',
        }

    if canal_sugerido == 'Digital' and cliente.get('es_usuario_app'):
        return {
            'momento': 'Push in-app, fuera del horario laboral',
            'detalle': 'Usuario activo de la app: el mensaje se consume cuando el cliente '
                       'decide, sin interrumpir.',
        }

    if canal_sugerido == 'Call Out':
        return {
            'momento': 'Media semana, en horario de tarde',
            'detalle': 'Contacto saliente: se evita el lunes y el viernes, y la franja de '
                       'tarde concentra mejor disponibilidad.',
        }

    return {
        'momento': 'En el próximo contacto entrante',
        'detalle': 'Aprovechar una interacción que inicie el propio cliente, en lugar de '
                   'generar un contacto adicional.',
    }


def get_segment_label(macro, cluster_name):
    """Etiqueta combinada de los dos ejes, para la ficha del asesor."""
    return f"{MACRO_INFO[macro]['etiqueta']} · {cluster_name}"


# --- Asignación de segmento -------------------------------------------------

class SegmentAssigner:
    """
    Aplica el modelo de segmentación entrenado a clientes en inferencia.

    Reutiliza build_segmentation_features() del módulo de entrenamiento para que
    entrenamiento e inferencia compartan exactamente la misma transformación.
    """

    def __init__(self, artifacts_dir):
        self.model = joblib.load(os.path.join(artifacts_dir, 'segmentation_model.pkl'))
        self.scaler = joblib.load(os.path.join(artifacts_dir, 'segmentation_scaler.pkl'))
        self.features = joblib.load(os.path.join(artifacts_dir, 'segmentation_features.pkl'))
        profiles = joblib.load(os.path.join(artifacts_dir, 'cluster_profiles.pkl'))
        self.names = profiles['names']
        self.traits = profiles['traits']
        self.profile_table = profiles['profile_table']
        self.metrics = profiles['metrics']
        self.k = profiles['k']

    def assign(self, df_clientes, df_ofertas):
        """Devuelve el DataFrame con cluster, nombre de segmento y macro-segmento."""
        from models.segmentation import build_segmentation_features

        d = build_segmentation_features(df_clientes, df_ofertas)
        d['cluster'] = self.model.predict(self.scaler.transform(d[self.features]))
        d['segmento_comportamental'] = d['cluster'].map(self.names)
        d['macro_segmento'] = d.apply(get_macro_segment, axis=1)
        d['segmento_label'] = [
            get_segment_label(m, c)
            for m, c in zip(d['macro_segmento'], d['segmento_comportamental'])
        ]
        return d

    def get_traits(self, cluster_id):
        return self.traits.get(int(cluster_id), {})
