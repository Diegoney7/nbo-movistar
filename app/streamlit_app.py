"""
Asistente NBO — interfaz del asesor comercial.

La interfaz está organizada alrededor del entregable mínimo del desafío: perfil
del cliente, oferta recomendada, motivo, probabilidad, canal y momento, y
beneficio esperado para ambas partes. Sobre eso se agregan las dos piezas que el
asesor necesita para ejecutar la venta —argumentario y rebate generados con IA
generativa— y las dos vistas de gestión que pide el reto: funnel E2E y
segmentación.

Todo lo que no aporta a esa conversación se dejó fuera de la pantalla principal:
las notas metodológicas viven en desplegables, no compiten con la venta.
"""

import html
import os
import sys

import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from nbo_engine import NBOEngine
from rebate_policy import MOTIVOS, MOTIVO_LABEL
from segment_policy import MACRO_INFO
from llm_assistant import (AsistenteComercial, MODELOS, construir_ficha,
                           elegir_modelo, resolver_api_key, resolver_modelo)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

st.set_page_config(
    page_title='Asistente NBO · Movistar',
    page_icon='🔵',
    layout='wide',
    initial_sidebar_state='expanded',
)


# ---------------------------------------------------------------- estilos ---
# Paleta de marca Movistar: azul #019DF4 sobre azul profundo #0B2739.

ESTILOS = """
<style>
:root {
  --mv-blue: #019DF4;
  --mv-deep: #0B2739;
  --mv-ink: #12303F;
  --mv-grey: #5C6C7A;
  --mv-soft: #8A99A6;
  --mv-line: #E1E8ED;
  --mv-bg: #F5F8FA;
  --mv-green: #3E9E1E;
  --mv-amber: #E08700;
  --mv-red: #D6336C;
}

.stApp { background: var(--mv-bg); }
.block-container { padding-top: 2.2rem; padding-bottom: 4rem; max-width: 1320px; }

h1, h2, h3, h4 { color: var(--mv-deep); letter-spacing: -0.015em; }

/* Barra de marca */
.mv-header {
  background: linear-gradient(100deg, #0B2739 0%, #0E3B57 60%, #10527A 100%);
  border-radius: 16px; padding: 20px 28px; margin-bottom: 26px;
  display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 12px;
}
.mv-logo { color: #fff; font-size: 27px; font-weight: 700; letter-spacing: -0.04em; line-height: 1; }
.mv-logo em { color: var(--mv-blue); font-style: normal; }
.mv-header-sub { color: #B8CBD8; font-size: 12.5px; margin-top: 6px; letter-spacing: .01em; }
.mv-header-tag {
  color: #fff; font-size: 12px; font-weight: 600; text-align: right;
  border-left: 3px solid var(--mv-blue); padding-left: 14px; line-height: 1.5;
}
.mv-header-tag span { display: block; color: #9FB9C9; font-weight: 400; }

/* Encabezado de sección numerado */
.mv-section { display: flex; align-items: center; gap: 12px; margin: 30px 0 14px; }
.mv-num {
  background: var(--mv-blue); color: #fff; width: 26px; height: 26px; border-radius: 8px;
  display: flex; align-items: center; justify-content: center;
  font-size: 13px; font-weight: 700; flex: 0 0 26px;
}
.mv-section-title { font-size: 17px; font-weight: 700; color: var(--mv-deep); line-height: 1.2; }
.mv-section-sub { font-size: 12.5px; color: var(--mv-soft); margin-top: 2px; }

/* Tarjetas */
.mv-card {
  background: #fff; border: 1px solid var(--mv-line); border-radius: 14px;
  padding: 18px 20px; height: 100%;
}
.mv-card h4 { font-size: 13px; text-transform: uppercase; letter-spacing: .06em;
              color: var(--mv-soft); margin: 0 0 12px; font-weight: 700; }

/* Fila de indicadores */
.mv-stats { display: flex; gap: 12px; flex-wrap: wrap; }
.mv-stat {
  background: #fff; border: 1px solid var(--mv-line); border-radius: 12px;
  padding: 13px 16px; flex: 1 1 150px; min-width: 140px;
}
.mv-stat .lbl { font-size: 11px; text-transform: uppercase; letter-spacing: .05em;
                color: var(--mv-soft); font-weight: 600; }
.mv-stat .val { font-size: 21px; font-weight: 700; color: var(--mv-deep); margin-top: 3px; line-height: 1.15; }
.mv-stat .sub { font-size: 11.5px; color: var(--mv-grey); margin-top: 3px; }
.mv-stat.acc { border-left: 3px solid var(--mv-blue); }

/* Etiquetas */
.mv-chip {
  display: inline-block; padding: 4px 11px; border-radius: 999px;
  font-size: 11.5px; font-weight: 600; margin-right: 6px; margin-top: 6px;
}
.chip-blue  { background: #E4F4FE; color: #06699F; }
.chip-green { background: #E6F4E1; color: #2E7714; }
.chip-amber { background: #FDF0DC; color: #9A5B00; }
.chip-grey  { background: #EDF1F4; color: #4E606E; }
.chip-red   { background: #FCE7EF; color: #A81F51; }

/* Tarjeta de oferta */
.mv-offer {
  background: linear-gradient(100deg, #0B2739 0%, #124C6E 100%);
  border-radius: 16px; padding: 22px 26px; color: #fff;
  display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 18px;
}
.mv-offer .eyebrow { font-size: 11px; text-transform: uppercase; letter-spacing: .09em; color: #8FC9E8; font-weight: 700; }
.mv-offer .name { font-size: 26px; font-weight: 700; margin-top: 6px; letter-spacing: -0.02em; }
.mv-offer .meta { font-size: 13px; color: #C3D6E1; margin-top: 8px; }
.mv-offer .price { text-align: right; }
.mv-offer .price .amount { font-size: 32px; font-weight: 700; color: #fff; line-height: 1; }
.mv-offer .price .unit { font-size: 12.5px; color: #9FC5DA; margin-top: 4px; }

/* Barra de probabilidad */
.mv-bar { background: #E4EBF0; border-radius: 999px; height: 9px; overflow: hidden; margin-top: 9px; }
.mv-bar > div { height: 100%; border-radius: 999px; }

/* Listas de argumentos */
.mv-list { margin: 0; padding: 0; list-style: none; }
.mv-list li { position: relative; padding-left: 20px; margin-bottom: 9px;
              font-size: 14px; color: var(--mv-ink); line-height: 1.45; }
.mv-list li:before { content: ''; position: absolute; left: 0; top: 7px;
                     width: 8px; height: 8px; border-radius: 3px; background: var(--mv-blue); }
.mv-list.rule li:before { background: #B9C6CF; }

/* Bloques del argumentario */
.mv-speech { background: #fff; border: 1px solid var(--mv-line); border-left: 3px solid var(--mv-blue);
             border-radius: 10px; padding: 14px 17px; margin-bottom: 11px; }
.mv-speech .step { font-size: 10.5px; text-transform: uppercase; letter-spacing: .08em;
                   color: var(--mv-blue); font-weight: 700; }
.mv-speech .txt { font-size: 15px; color: var(--mv-ink); line-height: 1.55; margin-top: 5px; }
.mv-speech.objection { border-left-color: var(--mv-amber); background: #FFFDF8; }
.mv-speech.objection .step { color: var(--mv-amber); }
.mv-read { font-size: 13.5px; color: var(--mv-grey); font-style: italic;
           border-left: 2px solid var(--mv-line); padding-left: 12px; margin-bottom: 14px; }

/* Sidebar */
[data-testid="stSidebar"] { background: #fff; border-right: 1px solid var(--mv-line); }
[data-testid="stSidebar"] .block-container { padding-top: 1.4rem; }
.mv-side-title { font-size: 11px; text-transform: uppercase; letter-spacing: .07em;
                 color: var(--mv-soft); font-weight: 700; margin: 18px 0 6px; }
.mv-status { border-radius: 9px; padding: 9px 12px; font-size: 12px; line-height: 1.4; }
.mv-status.on  { background: #E6F4E1; color: #2E7714; border: 1px solid #C6E5B8; }
.mv-status.off { background: #FDF0DC; color: #9A5B00; border: 1px solid #F1D9B2; }

/* Ajustes de widgets */
div[data-testid="stMetricValue"] { font-size: 22px; color: var(--mv-deep); }
.stButton > button { border-radius: 9px; font-weight: 600; }
div[data-testid="stExpander"] { border: 1px solid var(--mv-line); border-radius: 12px; background: #fff; }
hr { border-color: var(--mv-line); }
</style>
"""
st.markdown(ESTILOS, unsafe_allow_html=True)


# --------------------------------------------------------------- recursos ---

@st.cache_resource(show_spinner='Cargando motor NBO...')
def cargar_motor():
    return NBOEngine(
        data_dir=os.path.join(BASE_DIR, 'data'),
        artifacts_dir=os.path.join(BASE_DIR, 'models_artifacts'),
    )


@st.cache_resource(show_spinner=False)
def cargar_prior_rebate():
    return joblib.load(os.path.join(BASE_DIR, 'models_artifacts', 'rebate_prior.pkl'))


@st.cache_resource(show_spinner=False)
def cargar_asistente(api_key: str, modelo: str):
    return AsistenteComercial(api_key=api_key or None, modelo=modelo)


@st.cache_data(show_spinner=False, ttl=3600)
def modelos_disponibles(api_key: str) -> list:
    """
    Modelos que habilita la credencial en uso. Se consulta a la API una vez por
    hora y por key: así el selector nunca ofrece un modelo que después falle con
    'no disponible para esta API key'.
    """
    try:
        return AsistenteComercial(api_key=api_key or None).listar_modelos()
    except Exception:
        return []


@st.cache_data(show_spinner=False)
def obtener_recomendaciones(_motor, cliente_id: str, top_n: int = 8):
    # Se piden más de las que se muestran: el motor de rebate necesita
    # alternativas de otras categorías y rangos de precio para maniobrar.
    return _motor.get_recommendations(cliente_id, top_n=top_n)


@st.cache_data(show_spinner=False)
def obtener_funnel(_motor):
    return _motor.get_funnel_e2e()


@st.cache_data(show_spinner=False)
def obtener_segmentos(_motor):
    return _motor.get_segment_overview()


EJES_TIPICO = ['monto_facturado_prom', 'consumo_datos_gb_prom', 'antiguedad_meses']


@st.cache_data(show_spinner=False)
def clientes_de_ejemplo(_motor):
    """
    Un cliente típico por macro-segmento, en orden de prioridad comercial.

    Dos decisiones que antes quedaban al azar:
    - El cliente se elige por cercanía al centro del segmento (factura, consumo y
      antigüedad estandarizados), no por posición en el archivo. Con la primera
      fila el ejemplo dependía del orden del CSV y podía ser un caso atípico.
    - El orden es el de MACRO_INFO['prioridad'], la misma política que usa el
      motor para rankear. Agrupar por la columna ordenaba por la clave técnica
      (BRECHA_HOGAR, BRECHA_MOVIL, ...), que el asesor nunca ve.
    """
    df = _motor.clientes_feat
    ejemplos = {}
    for macro in sorted(MACRO_INFO, key=lambda m: MACRO_INFO[m]['prioridad']):
        grupo = df[df['macro_segmento'] == macro]
        if grupo.empty:
            continue
        desv = grupo[EJES_TIPICO].std(ddof=0).replace(0, 1)
        z = ((grupo[EJES_TIPICO] - grupo[EJES_TIPICO].mean()) / desv).abs().sum(axis=1)
        tipico = grupo.loc[z.idxmin()]
        etiqueta = f"{MACRO_INFO[macro]['etiqueta']} — {tipico['cliente_id']}"
        ejemplos[etiqueta] = tipico['cliente_id']
    return ejemplos


# ---------------------------------------------------------------- helpers ---

def esc(texto) -> str:
    return html.escape(str(texto if texto is not None else ''))


def plural(n, singular, plural_) -> str:
    """Concordancia de número. La pantalla la lee un cliente por encima del
    hombro del asesor: '1 meses con mora' descuida toda la ficha."""
    n = int(n or 0)
    return f'{n:,} {singular if n == 1 else plural_}'


# Los nombres técnicos de las columnas no se muestran nunca: las vistas de
# gestión las lee gerencia y marketing, no quien escribió el pipeline.
ETIQUETAS = {
    'etapa': 'Etapa', 'volumen': 'Volumen', 'conversion_vs_anterior': 'Conversión vs. etapa anterior (%)',
    'ofrecimientos': 'Ofrecimientos', 'aceptados': 'Ventas', 'tasa_cierre': 'Cierre (%)',
    'cobertura_pct': 'Cobertura (%)', 'contactados': 'Contactados', 'tasa_contacto': 'Contactabilidad (%)',
    'medio_probatorio': 'Medio probatorio', 'canal': 'Canal', 'macro_segmento': 'Situación comercial',
    'segmento_comportamental': 'Segmento comportamental', 'tasa_aceptacion': 'Aceptación (%)',
    'tasa_aceptacion_mt': 'Aceptación de MT (%)', 'ofrecimientos_mt': 'Ofrecimientos MT',
    'n_clientes': 'Clientes', 'cluster': 'Grupo', 'nombre': 'Segmento', 'pct_base': '% de la base',
    'pct_elegible_mt': '% elegible MT', 'pct_ya_mt': '% ya es MT', 'monto_facturado_prom': 'Factura (S/)',
    'consumo_datos_gb_prom': 'Consumo (GB)', 'ratio_consumo_gb': 'Uso del plan',
    'antiguedad_meses': 'Antigüedad (meses)', 'dias_mora_prom': 'Días de mora',
    'n_reclamos': 'Reclamos', 'n_actividad_canal': 'Interacciones', 'score_digital': 'Score digital',
    'k': 'K', 'inercia': 'Inercia', 'silhouette': 'Silhouette', 'davies_bouldin': 'Davies-Bouldin',
    'cluster_min_pct': 'Grupo más pequeño (%)',
}
# Las etiquetas de macro-segmento salen de la política de negocio, no de una
# copia local: mantener dos diccionarios hacía que la ficha del asesor dijera
# "Le falta hogar para MT" y las tablas de gestión "Le falta hogar".
MACRO_ETIQUETA = {macro: info['etiqueta'] for macro, info in MACRO_INFO.items()}
MACRO_ORDEN = sorted(MACRO_INFO, key=lambda m: MACRO_INFO[m]['prioridad'])

# La regla de arriba vale también para los VALORES, no solo para los nombres de
# columna: el medio probatorio venía del CSV en snake_case y llegaba tal cual a
# la tabla que lee gestión ("audio_llamada", "registro_plataforma").
MEDIO_ETIQUETA = {
    'audio_llamada': 'Audio de la llamada',
    'chat_log': 'Registro del chat',
    'registro_plataforma': 'Registro en plataforma',
}
VALOR_ETIQUETA = {**MACRO_ETIQUETA, **MEDIO_ETIQUETA}

# A quién sirve cada vista. Los tres usuarios salen del desafío: asesores y
# canales de atención, y Producto y Marketing.
AUDIENCIA = {
    'Asesor comercial': 'Para el asesor, durante la llamada o la visita.',
    'Funnel E2E del ofrecimiento': 'Para gestión comercial: trazabilidad del ofrecimiento.',
    'Segmentación de clientes': 'Para Producto y Marketing: dónde armar campaña.',
}

# Qué le falta al cliente para tener Movistar Total. Acompaña a la tarjeta de
# servicios contratados: el dato de al lado tiene que hablar de lo mismo que el
# título de la tarjeta.
BRECHA_SERVICIO = {
    'YA_MT': 'Convergencia completa',
    'ELEGIBLE_MT': 'Ya cumple los requisitos de MT',
    'BRECHA_HOGAR': 'Le falta internet hogar',
    'BRECHA_MOVIL': 'Le falta móvil postpago',
    'NO_CONVERGENTE': 'Requiere migrar a postpago primero',
}


def brecha_de_servicio(cliente) -> str:
    """
    Qué falta para MT, dicho como lo diría el asesor.

    BRECHA_MOVIL junta dos casos distintos —quien no tiene móvil y quien lo
    tiene prepago— y al segundo decirle "le falta móvil" contradice la propia
    tarjeta, que muestra móvil contratado.
    """
    macro = cliente['macro_segmento']
    if macro == 'BRECHA_MOVIL' and cliente['tiene_movil']:
        return 'Su móvil es prepago: falta postpago'
    return BRECHA_SERVICIO.get(macro, '')


def tabla(df, **kwargs):
    """Muestra un DataFrame con nombres legibles y decimales acotados."""
    d = df.copy()
    if isinstance(d, pd.Series):
        d = d.to_frame()
    d = d.rename(columns=ETIQUETAS).rename(columns=VALOR_ETIQUETA)
    d.index = d.index.set_names([ETIQUETAS.get(n, n) for n in d.index.names])
    if d.index.nlevels == 1:
        d.index = d.index.map(lambda v: VALOR_ETIQUETA.get(v, v))
    else:
        d.index = pd.MultiIndex.from_tuples(
            [tuple(VALOR_ETIQUETA.get(v, v) for v in fila) for fila in d.index],
            names=d.index.names)
    for col in d.select_dtypes('float').columns:
        d[col] = d[col].round(2)
    st.dataframe(d, width='stretch', **kwargs)


def seccion(numero, titulo, subtitulo=''):
    st.markdown(
        f'<div class="mv-section"><span class="mv-num">{numero}</span>'
        f'<div><div class="mv-section-title">{esc(titulo)}</div>'
        f'<div class="mv-section-sub">{esc(subtitulo)}</div></div></div>',
        unsafe_allow_html=True,
    )


def tiles(items):
    """Fila de indicadores. items: lista de (label, valor, detalle, destacar)."""
    piezas = []
    for label, valor, detalle, destacar in items:
        clase = 'mv-stat acc' if destacar else 'mv-stat'
        piezas.append(
            f'<div class="{clase}"><div class="lbl">{esc(label)}</div>'
            f'<div class="val">{esc(valor)}</div>'
            f'<div class="sub">{esc(detalle)}</div></div>'
        )
    st.markdown(f'<div class="mv-stats">{"".join(piezas)}</div>', unsafe_allow_html=True)


# --------------------------------------------------------------- gráficos ---
# Paleta de marca aplicada a Plotly, y una sola familia de gráfico: barras
# horizontales planas. El trapecio del go.Funnel de Plotly codifica el volumen
# en un área inclinada, que se lee peor que una longitud y sugiere una
# profundidad que no existe en el dato. Con barras, dos etapas parecidas se
# distinguen por lo único que hay que comparar: el largo.

COLOR_DEEP = '#0B2739'
COLOR_BLUE = '#019DF4'
COLOR_GREEN = '#5CB615'
COLOR_SOFT = '#8A99A6'
COLOR_INK = '#12303F'
COLOR_LINE = '#E1E8ED'

# El color no decora: dice qué hacer con el segmento. Azul de marca = donde está
# el objetivo del desafío, verde = ya convertido, azul profundo = trabajable,
# gris = sin ruta directa.
COLOR_MACRO = {
    'ELEGIBLE_MT': COLOR_BLUE,
    'YA_MT': COLOR_GREEN,
    'BRECHA_HOGAR': COLOR_DEEP,
    'BRECHA_MOVIL': COLOR_DEEP,
    'NO_CONVERGENTE': COLOR_SOFT,
}


def barras(etiquetas, valores, colores, texto=None, notas=None, holgura=1.42, alto_fila=44):
    """
    Barras horizontales de arriba hacia abajo.

    El valor va dentro de la barra ('auto' lo saca afuera si no entra, en vez de
    esconderlo) y la lectura de negocio —conversión, aceptación— se alinea a la
    derecha del área de dibujo. Alineadas en columna se comparan entre sí, y no
    chocan con el valor por más corta que quede la barra.
    """
    etiquetas, valores = list(etiquetas), list(valores)
    fig = go.Figure(go.Bar(
        x=valores, y=etiquetas, orientation='h',
        marker=dict(color=colores, line=dict(width=0)),
        text=texto or [f'{v:,.0f}' for v in valores],
        textposition='auto', insidetextanchor='start',
        insidetextfont=dict(color='#fff', size=13.5),
        outsidetextfont=dict(color=COLOR_INK, size=13.5),
        constraintext='none', cliponaxis=False,
        hovertemplate='%{y}<br>%{x:,.0f}<extra></extra>',
    ))
    for etiqueta, nota in zip(etiquetas, notas or [None] * len(valores)):
        if nota:
            fig.add_annotation(x=1, y=etiqueta, xref='paper', yref='y', text=nota,
                               showarrow=False, xanchor='right',
                               font=dict(size=12, color=COLOR_SOFT))
    fig.update_yaxes(autorange='reversed', showgrid=False, zeroline=False,
                     ticksuffix='   ', tickfont=dict(size=12.5, color=COLOR_INK),
                     automargin=True)
    fig.update_xaxes(visible=False, range=[0, max(valores) * holgura])
    fig.update_layout(
        height=52 + alto_fila * len(etiquetas), bargap=0.36,
        margin=dict(l=0, r=6, t=6, b=6),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        showlegend=False, font=dict(family='sans-serif', color=COLOR_INK),
        hoverlabel=dict(bgcolor='#fff', bordercolor=COLOR_LINE,
                        font=dict(color=COLOR_INK, size=12)),
    )
    return fig


def grafico(fig):
    st.plotly_chart(fig, width='stretch', config={'displayModeBar': False})


def nivel_probabilidad(p):
    if p >= 0.60:
        return 'Alta', 'var(--mv-green)', 'chip-green'
    if p >= 0.40:
        return 'Media', 'var(--mv-amber)', 'chip-amber'
    return 'Baja', 'var(--mv-soft)', 'chip-grey'


TIPO_ETIQUETA = {
    'movistar_total': 'Movistar Total',
    'plan_movil': 'Plan móvil',
    'plan_hogar': 'Plan hogar',
    'upgrade': 'Upgrade',
    'paquete_adicional': 'Paquete adicional',
    'equipo': 'Equipo',
}


def meta_oferta(rec) -> str:
    """Qué incluye la oferta, en una línea. Los planes hogar no traen GB: en ese
    caso manda el bundle, no una frase en negativo sobre datos."""
    partes = []
    gb = float(rec.get('gb_incluidos') or 0)
    if gb >= 9999:
        partes.append('Datos ilimitados')
    elif gb > 0:
        partes.append(f'{gb:.0f} GB incluidos')

    bundle = rec.get('descripcion_bundle')
    if bundle and bundle not in ('no_aplica', None):
        partes.append(str(bundle))

    return ' · '.join(partes)


# ---------------------------------------------------------------- cabecera ---

st.markdown(
    '<div class="mv-header">'
    '<div><div class="mv-logo">m<em>o</em>vistar</div>'
    '<div class="mv-header-sub">Asistente de Next Best Offer · Personalización comercial inteligente</div></div>'
    '<div class="mv-header-tag">AI Telecom Challenge<span>Movistar · Universidad de Lima</span></div>'
    '</div>',
    unsafe_allow_html=True,
)

motor = cargar_motor()
prior_rebate = cargar_prior_rebate()
catalogo_nombres = dict(zip(motor.ofertas['oferta_id'], motor.ofertas['nombre_oferta']))


# ----------------------------------------------------------------- sidebar ---

with st.sidebar:
    st.markdown('<div class="mv-side-title">Vista</div>', unsafe_allow_html=True)
    vista = st.radio(
        'Vista', list(AUDIENCIA), label_visibility='collapsed',
    )
    # Cada vista tiene un usuario distinto de los que nombra el desafío. Decirlo
    # en la barra evita que el asesor entre a pantallas de gestión buscando algo
    # que le sirva para la llamada.
    st.caption(AUDIENCIA[vista])

    cliente_id = None
    if vista == 'Asesor comercial':
        st.markdown('<div class="mv-side-title">Cliente</div>', unsafe_allow_html=True)

        # Buscar por código es un modo de trabajo, no un cliente: mezclarlo en la
        # misma lista hacía que al desplegar apareciera como si fuera uno más.
        modo = st.segmented_control(
            'Modo de búsqueda', ['Casos de ejemplo', 'Por código'],
            default='Casos de ejemplo', label_visibility='collapsed',
        ) or 'Casos de ejemplo'

        if modo == 'Por código':
            cliente_id = st.text_input('Código de cliente', value='CLI000001',
                                       placeholder='CLI000001',
                                       label_visibility='collapsed').strip().upper()
        else:
            ejemplos = clientes_de_ejemplo(motor)
            eleccion = st.selectbox(
                'Cliente', list(ejemplos), label_visibility='collapsed',
                help='Un cliente típico de cada situación comercial, ordenados por la '
                     'prioridad de ofrecimiento que aplica el motor.')
            cliente_id = ejemplos[eleccion]
            st.caption(f'Cliente **{cliente_id}**')

    # --- Credencial y modelo ------------------------------------------------
    # Si la API key viene del servidor (secrets de Streamlit o variable de
    # entorno), la pantalla es solo del asesor: no se muestra nada de
    # configuración. El panel técnico queda accesible agregando ?config=1 a la
    # URL, para que el equipo pueda cambiar de modelo o probar la conexión sin
    # tocar el código ni exponerle nada al asesor.
    key_del_servidor = resolver_api_key()
    modo_config = not key_del_servidor or st.query_params.get('config') == '1'

    probar = False
    key_de_sesion = st.session_state.get('gemini_key', '')

    if modo_config:
        st.markdown('<div class="mv-side-title">Asistente de IA generativa</div>',
                    unsafe_allow_html=True)
        with st.expander('Configurar credencial', expanded=not key_del_servidor):
            api_key_input = st.text_input(
                'API key de Google Gemini', type='password', value=key_de_sesion,
                help='Se guarda solo en esta sesión del navegador. Para dejarla fija, usar '
                     'los secrets de Streamlit o la variable de entorno GEMINI_API_KEY.',
            )
            if api_key_input != key_de_sesion:
                st.session_state['gemini_key'] = api_key_input
                key_de_sesion = api_key_input
                st.session_state.pop('ia_cache', None)
                modelos_disponibles.clear()
            probar = st.button('Probar conexión', width='stretch')

    # El selector ofrece lo que la API key habilita de verdad, consultado a la
    # API. Si no hay credencial o la consulta falla, se cae a la lista sugerida.
    opciones_modelo = modelos_disponibles(key_de_sesion) or list(MODELOS)
    modelo = elegir_modelo(resolver_modelo(), opciones_modelo)

    if modo_config:
        modelo = st.selectbox('Modelo', opciones_modelo,
                              index=opciones_modelo.index(modelo),
                              label_visibility='collapsed')
        if MODELOS.get(modelo):
            st.caption(MODELOS[modelo])

    asistente = cargar_asistente(key_de_sesion, modelo)

    # Al asesor solo se le avisa cuando la IA NO está disponible, porque eso sí
    # cambia lo que va a leer en pantalla. Que funcione es la normalidad y no
    # merece ocupar espacio: el estado técnico se muestra en modo configuración.
    if not asistente.disponible:
        st.markdown('<div class="mv-status off">Sin IA: los argumentos se generan '
                    'con plantillas del motor.</div>', unsafe_allow_html=True)
    elif modo_config:
        st.markdown(f'<div class="mv-status on">IA conectada · {esc(modelo)}</div>',
                    unsafe_allow_html=True)

    if probar:
        with st.spinner('Verificando...'):
            ok, mensaje = asistente.verificar()
        (st.success if ok else st.error)(mensaje)


# ------------------------------------------------------- vista del asesor ---

def bloque_argumentario(resultado, clave):
    """Renderiza el speech de venta generado por el asistente."""
    data = resultado['data']

    if resultado['origen'] == 'gemini':
        st.caption(f"Generado con {resultado['modelo']} en {resultado['latencia_ms'] / 1000:.1f} s "
                   "sobre los datos verificados del cliente y del catálogo.")
    else:
        st.warning(f"Argumentario de plantilla (sin IA). {resultado['error']}")

    if data.get('lectura_cliente'):
        st.markdown(f'<div class="mv-read">{esc(data["lectura_cliente"])}</div>',
                    unsafe_allow_html=True)

    izq, der = st.columns([3, 2], gap='medium')

    with izq:
        pasos = [
            ('1 · Apertura', data.get('apertura')),
            ('2 · Argumento central', data.get('argumento_central')),
        ]
        for etiqueta, texto in pasos:
            if texto:
                st.markdown(f'<div class="mv-speech"><div class="step">{esc(etiqueta)}</div>'
                            f'<div class="txt">{esc(texto)}</div></div>', unsafe_allow_html=True)

        beneficios = data.get('beneficios_clave') or []
        if beneficios:
            items = ''.join(f'<li>{esc(b)}</li>' for b in beneficios)
            st.markdown(f'<div class="mv-speech"><div class="step">3 · Beneficios para decir de corrido</div>'
                        f'<ul class="mv-list" style="margin-top:9px">{items}</ul></div>',
                        unsafe_allow_html=True)

        if data.get('cierre'):
            st.markdown(f'<div class="mv-speech"><div class="step">4 · Cierre</div>'
                        f'<div class="txt">{esc(data["cierre"])}</div></div>', unsafe_allow_html=True)

    with der:
        if data.get('objecion_anticipada'):
            st.markdown(
                f'<div class="mv-speech objection"><div class="step">Objeción probable</div>'
                f'<div class="txt">{esc(data["objecion_anticipada"])}</div></div>',
                unsafe_allow_html=True)
        if data.get('respuesta_objecion'):
            st.markdown(
                f'<div class="mv-speech objection"><div class="step">Cómo responderla</div>'
                f'<div class="txt">{esc(data["respuesta_objecion"])}</div></div>',
                unsafe_allow_html=True)
        if data.get('mensaje_digital'):
            st.markdown('**Mensaje para app o WhatsApp**')
            st.code(data['mensaje_digital'], language=None, wrap_lines=True)


def vista_asesor(cliente_id):
    if not cliente_id:
        st.info('Ingrese un código de cliente para comenzar.')
        return

    ficha_cliente = motor.clientes_feat[motor.clientes_feat['cliente_id'] == cliente_id]
    if ficha_cliente.empty:
        st.error(f'No se encontró el cliente {cliente_id}. Verifique el código (formato CLI000001).')
        return

    cliente = ficha_cliente.iloc[0]
    segmento = motor.get_client_segment(cliente_id)

    # --- 1. Perfil resumido del cliente ------------------------------------
    seccion(1, 'Perfil del cliente', 'Lo que el asesor necesita saber antes de hablar.')

    chips = []
    estado_mt = None
    if cliente['es_movistar_total']:
        estado_mt = ('chip-green', 'Ya es Movistar Total')
    elif cliente['elegible_mt']:
        estado_mt = ('chip-blue', 'Elegible Movistar Total')
    if estado_mt:
        chips.append(estado_mt)
    if segmento:
        # En YA_MT y ELEGIBLE_MT la etiqueta del macro-segmento es exactamente el
        # estado que ya se muestra en color: se omite para no repetir el chip.
        if not estado_mt or segmento['macro_etiqueta'] != estado_mt[1]:
            chips.append(('chip-grey', segmento['macro_etiqueta']))
        # El nombre del cluster describe al GRUPO, no al cliente: el segmento
        # "Reciente" promedia 38 meses pero llega hasta 180, así que sin el
        # prefijo el chip contradecía la antigüedad que se muestra debajo.
        chips.append(('chip-grey', f"Segmento {segmento['segmento_comportamental']}"))
    chips.append(('chip-grey', f"{cliente['edad_rango']} años · {cliente['ubicacion_departamento']}"))
    if int(cliente['meses_moroso'] or 0) > 0:
        chips.append(('chip-amber', plural(cliente['meses_moroso'], 'mes con mora', 'meses con mora')))
    if int(cliente['n_reclamos'] or 0) > 0:
        chips.append(('chip-red', plural(cliente['n_reclamos'], 'reclamo', 'reclamos')))
    if cliente['es_usuario_app']:
        chips.append(('chip-blue', 'Usuario de la app'))

    st.markdown(
        '<div style="margin-bottom:12px">'
        + ''.join(f'<span class="mv-chip {c}">{esc(t)}</span>' for c, t in chips)
        + '</div>', unsafe_allow_html=True)

    tiles([
        ('Plan actual', catalogo_nombres.get(cliente['plan_actual_id'], cliente['plan_actual_id']),
         f"Cliente {cliente['tipo_cliente']}", False),
        ('Factura mensual', f"S/ {cliente['monto_facturado_prom']:.2f}",
         plural(cliente['antiguedad_meses'], 'mes de antigüedad', 'meses de antigüedad'), False),
        ('Consumo de datos', f"{cliente['consumo_datos_gb_prom']:.1f} GB",
         f"{cliente['consumo_voz_min_prom']:.0f} min de voz al mes", False),
        ('Servicios contratados',
         'Móvil + Hogar' if cliente['tiene_movil'] and cliente['tiene_internet_hogar']
         else ('Solo móvil' if cliente['tiene_movil'] else 'Solo hogar'),
         brecha_de_servicio(cliente), False),
        ('Canal habitual', cliente['canal_mas_usado'],
         plural(cliente['n_actividad_canal'], 'interacción registrada',
                'interacciones registradas'), False),
    ])

    if segmento:
        with st.expander(f"Cómo tratar a este perfil · {segmento['segmento_comportamental']}"):
            st.markdown(f"**Acción comercial:** {segmento['accion']}")
            st.markdown('\n'.join(f'- {a}' for a in segmento['argumentos']))

    # --- 2. Oferta recomendada ---------------------------------------------
    with st.spinner('Evaluando el catálogo completo para este cliente...'):
        recs = obtener_recomendaciones(motor, cliente_id)

    if not recs:
        st.warning('No hay ofertas contratables para este cliente con las reglas actuales.')
        return

    mejor = recs[0]
    prob = mejor['probabilidad']
    etiqueta_prob, color_prob, chip_prob = nivel_probabilidad(prob)

    seccion(2, 'Oferta recomendada', 'La mejor jugada para este cliente en este momento.')

    st.markdown(
        f'<div class="mv-offer"><div>'
        f'<div class="eyebrow">{esc(TIPO_ETIQUETA.get(mejor["tipo_oferta"], mejor["tipo_oferta"]))}'
        f'{" · Venta convergente" if mejor["es_mt"] else ""}</div>'
        f'<div class="name">{esc(mejor["nombre_oferta"])}</div>'
        f'<div class="meta">{esc(meta_oferta(mejor))}'
        + (f' · {mejor["ahorro_pct"]:.0f}% de ahorro frente a contratar por separado'
           if mejor['ahorro_pct'] else '')
        + '</div></div>'
        f'<div class="price"><div class="amount">S/ {mejor["precio_mensual"]:.2f}</div>'
        f'<div class="unit">por mes</div></div></div>',
        unsafe_allow_html=True,
    )

    momento = mejor['momento']
    st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns([1.1, 1, 1.2], gap='small')
    with c1:
        st.markdown(
            f'<div class="mv-stat acc"><div class="lbl">4 · Probabilidad de aceptación</div>'
            f'<div class="val">{prob:.0%} <span class="mv-chip {chip_prob}" '
            f'style="vertical-align:middle">{etiqueta_prob}</span></div>'
            f'<div class="mv-bar"><div style="width:{min(prob, 1) * 100:.0f}%;'
            f'background:{color_prob}"></div></div>'
            f'<div class="sub">Probabilidad calibrada: de cada 100 clientes iguales, '
            f'{prob * 100:.0f} aceptan.</div></div>',
            unsafe_allow_html=True)
    with c2:
        st.markdown(
            f'<div class="mv-stat acc"><div class="lbl">5 · Canal sugerido</div>'
            f'<div class="val">{esc(mejor["canal_sugerido"])}</div>'
            f'<div class="sub">{esc(mejor["canal_criterio"])}</div></div>',
            unsafe_allow_html=True)
    with c3:
        st.markdown(
            f'<div class="mv-stat acc"><div class="lbl">5 · Momento sugerido</div>'
            f'<div class="val" style="font-size:17px">{esc(momento["momento"])}</div>'
            f'<div class="sub">{esc(momento["detalle"])}</div></div>',
            unsafe_allow_html=True)

    # --- 3 y 6. Motivo y beneficio -----------------------------------------
    izq, der = st.columns(2, gap='medium')

    with izq:
        seccion(3, 'Por qué esta oferta', 'Explicabilidad del modelo y reglas de negocio.')
        motivos = ''.join(f'<li>{esc(m)}</li>' for m in mejor['motivos'])
        bloque = f'<div class="mv-card"><h4>Señales del cliente</h4><ul class="mv-list">{motivos}</ul>'
        if mejor.get('reglas_segmento'):
            reglas = ''.join(f'<li>{esc(r)}</li>' for r in mejor['reglas_segmento'])
            bloque += ('<h4 style="margin-top:16px">Reglas aplicadas al ranking</h4>'
                       f'<ul class="mv-list rule">{reglas}</ul>')
        st.markdown(bloque + '</div>', unsafe_allow_html=True)

    with der:
        seccion(6, 'Beneficio esperado', 'Para el cliente y para el negocio.')
        beneficio = mejor['beneficio']
        neg = beneficio['negocio']
        items_cliente = ''.join(f'<li>{esc(b)}</li>' for b in beneficio['cliente']['beneficios'])
        st.markdown(
            f'<div class="mv-card"><h4>Para el cliente</h4>'
            f'<ul class="mv-list">{items_cliente}</ul>'
            f'<h4 style="margin-top:16px">Para el negocio</h4>'
            f'<div class="mv-stats">'
            f'<div class="mv-stat"><div class="lbl">ARPU incremental</div>'
            f'<div class="val">S/ {neg["arpu_incremental"]:.2f}</div>'
            f'<div class="sub">{esc(neg["nota"])}</div></div>'
            f'<div class="mv-stat"><div class="lbl">Valor esperado</div>'
            f'<div class="val">S/ {neg["valor_esperado_mensual"]:.2f}</div>'
            f'<div class="sub">S/ {neg["valor_esperado_anual"]:,.0f} al año por este contacto</div></div>'
            f'</div>'
            + ('<div style="margin-top:10px"><span class="mv-chip chip-green">Venta convergente: '
               'suma a la meta de Movistar Total y blinda la cuenta</span></div>' if neg['es_convergente'] else '')
            + ('<div style="font-size:11.5px;color:var(--mv-soft);margin-top:12px">'
               'El porcentaje de ahorro proviene del catálogo del desafío y es ilustrativo: '
               'no representa una tarifa oficial.</div>'
               if beneficio['cliente']['ahorro_mensual'] > 0 else '')
            + '</div>',
            unsafe_allow_html=True)

    # --- Argumentario generado con IA --------------------------------------
    seccion('IA', 'Argumentario para la conversación',
            'Generado con IA sobre los datos verificados del cliente: nada de lo que dice está inventado.')

    ficha_ia = construir_ficha(
        cliente.to_dict(), mejor, segmento,
        plan_actual_nombre=catalogo_nombres.get(cliente['plan_actual_id']),
    )
    clave = f"{cliente_id}|{mejor['oferta_id']}|{modelo}"
    cache_ia = st.session_state.setdefault('ia_cache', {})

    col_btn, col_info = st.columns([1, 3])
    with col_btn:
        generar = st.button('Generar argumentario', type='primary', width='stretch')
    with col_info:
        if not asistente.disponible:
            st.caption('Sin credencial de Gemini: se mostrará el argumentario de plantilla.')

    if generar or clave in cache_ia:
        if generar or clave not in cache_ia:
            with st.spinner('Redactando el argumentario...'):
                cache_ia[clave] = asistente.argumentario(ficha_ia)
        bloque_argumentario(cache_ia[clave], clave)

    # --- Rebate -------------------------------------------------------------
    seccion('↺', 'Si el cliente dice que no',
            'Marque la objeción que escuchó y el motor elige la contraoferta.')

    motivo = st.radio(
        'Objeción escuchada', options=MOTIVOS,
        format_func=lambda m: f'{MOTIVO_LABEL[m]}   ·   {prior_rebate.get(m, 0) * 100:.0f}% de los rechazos',
        index=None,
    )

    if motivo:
        reb = motor.get_rebate(recs, motivo)
        tipo = reb['tipo']

        if tipo == 'reintento':
            st.markdown(f'<div class="mv-card" style="border-left:3px solid var(--mv-amber)">'
                        f'<h4>Estrategia · reintento</h4>'
                        f'<div style="font-size:15px;color:var(--mv-ink)">{esc(reb["detalle"])}</div></div>',
                        unsafe_allow_html=True)
        elif tipo == 'sin_alternativa':
            st.markdown(f'<div class="mv-card" style="border-left:3px solid var(--mv-red)">'
                        f'<h4>Sin alternativa contratable</h4>'
                        f'<div style="font-size:15px;color:var(--mv-ink)">{esc(reb["detalle"])}</div></div>',
                        unsafe_allow_html=True)
        else:
            alt = reb['oferta']
            st.markdown(
                f'<div class="mv-card" style="border-left:3px solid var(--mv-blue)">'
                f'<h4>Contraoferta</h4>'
                f'<div style="font-size:19px;font-weight:700;color:var(--mv-deep)">'
                f'{esc(alt["nombre_oferta"])} · S/ {alt["precio_mensual"]:.2f} al mes</div>'
                f'<div style="font-size:14px;color:var(--mv-grey);margin-top:7px">{esc(reb["detalle"])}</div>'
                f'<div style="margin-top:9px"><span class="mv-chip chip-blue">'
                f'Probabilidad {alt["probabilidad"]:.0%}</span>'
                f'<span class="mv-chip chip-grey">Canal {esc(reb["canal"])}</span></div></div>',
                unsafe_allow_html=True)

        clave_reb = f"{clave}|{motivo}"
        if st.button('Generar speech de rebate', type='primary'):
            with st.spinner('Redactando la respuesta...'):
                cache_ia[clave_reb] = asistente.rebate(ficha_ia, MOTIVO_LABEL[motivo], reb)

        if clave_reb in cache_ia:
            res = cache_ia[clave_reb]
            data = res['data']
            if res['origen'] == 'gemini':
                st.caption(f"Generado con {res['modelo']} en {res['latencia_ms'] / 1000:.1f} s.")
            else:
                st.warning(f"Speech de plantilla (sin IA). {res['error']}")

            a, b = st.columns([3, 2], gap='medium')
            with a:
                for etiqueta, texto in [('1 · Reconocer', data.get('reconocimiento')),
                                        ('2 · Reencuadrar', data.get('argumento')),
                                        ('3 · Proponer', data.get('propuesta')),
                                        ('4 · Cerrar', data.get('cierre'))]:
                    if texto:
                        st.markdown(f'<div class="mv-speech"><div class="step">{esc(etiqueta)}</div>'
                                    f'<div class="txt">{esc(texto)}</div></div>', unsafe_allow_html=True)
            with b:
                if data.get('mensaje_seguimiento'):
                    st.markdown('**Mensaje de seguimiento**')
                    st.code(data['mensaje_seguimiento'], language=None, wrap_lines=True)
                st.caption('Estrategia del motor: ' + reb['speech'])

        with st.expander('Por qué el motivo de rechazo no se predice'):
            st.markdown(
                'En los datos históricos el motivo de rechazo sigue una distribución fija '
                '(precio 35%, no necesita 20%, ya tiene similar 15%, mal momento 15%, no confía 10%, '
                'otro 5%), idéntica para cualquier oferta, canal o perfil: un clasificador rinde por '
                'debajo del azar. En la conversación real el asesor **ya escuchó** la objeción, así que '
                'se captura y se usa para elegir la contraoferta.'
            )

    # --- Alternativas -------------------------------------------------------
    st.markdown('<div class="mv-section" style="margin-top:34px">'
                '<div><div class="mv-section-title">Otras alternativas</div>'
                '<div class="mv-section-sub">Siguientes opciones del ranking, por si la conversación '
                'toma otro rumbo.</div></div></div>', unsafe_allow_html=True)

    cols = st.columns(min(3, len(recs[1:4])) or 1)
    for col, alt in zip(cols, recs[1:4]):
        with col:
            st.markdown(
                f'<div class="mv-card"><h4>{esc(TIPO_ETIQUETA.get(alt["tipo_oferta"], alt["tipo_oferta"]))}</h4>'
                f'<div style="font-size:16px;font-weight:700;color:var(--mv-deep);line-height:1.3">'
                f'{esc(alt["nombre_oferta"])}</div>'
                f'<div style="font-size:14px;color:var(--mv-grey);margin-top:6px">'
                f'S/ {alt["precio_mensual"]:.2f} al mes'
                + (f' · {esc(meta_oferta(alt))}' if meta_oferta(alt) else '') + '</div>'
                f'<div style="margin-top:9px"><span class="mv-chip chip-blue">'
                f'Probabilidad {alt["probabilidad"]:.0%}</span>'
                f'<span class="mv-chip chip-grey">{esc(alt["canal_sugerido"])}</span></div>'
                f'<div style="font-size:13px;color:var(--mv-grey);margin-top:10px">'
                f'{esc(alt["motivos"][0]) if alt.get("motivos") else ""}</div></div>',
                unsafe_allow_html=True)


# ------------------------------------------------------------ vista funnel ---

def vista_funnel():
    st.markdown('## Funnel E2E del ofrecimiento')
    st.caption('De la clasificación del cliente al resultado de venta, con el medio probatorio '
               'que respalda cada contacto: permite auditar si un producto fue efectivamente ofrecido.')

    with st.spinner('Construyendo el funnel...'):
        f = obtener_funnel(motor)

    tiles([
        ('Clientes clasificados', f"{f['base']['clientes_clasificados']:,}",
         f"{f['base']['ofrecimientos_por_cliente']:.1f} ofrecimientos por cliente", False),
        ('Ventas Movistar Total', f"{f['mt']['vendidas']:,}",
         f"{f['mt']['pct_de_ventas_totales']:.1f}% de la venta total", True),
        ('Cierre de MT ofrecida', f"{f['mt']['tasa']:.1f}%",
         f"sobre {f['mt']['ofrecidas']:,} ofrecimientos MT", True),
        ('Rebates realizados', f"{f['rebates']['total']:,}",
         f"tasa de cierre {f['rebates']['tasa']:.2f}%", False),
    ])

    st.markdown('<div style="height:22px"></div>', unsafe_allow_html=True)
    izq, der = st.columns([3, 2], gap='medium')

    with izq:
        st.markdown('##### Del ofrecimiento a la venta')
        etapas = f['etapas']
        volumenes = etapas['volumen'].tolist()
        inicial = volumenes[0]
        # El número de etapa ya lo da la posición en el gráfico, y el detalle del
        # medio probatorio está en la tabla de al lado: en el eje solo va el nombre.
        nombres = [e.split('. ', 1)[-1].split(' (')[0] for e in etapas['etapa']]

        notas = [None if pd.isna(conv) else f'{conv:.1f}% del paso anterior'
                 for conv in etapas['conversion_vs_anterior']]

        grafico(barras(nombres, volumenes,
                       [COLOR_DEEP, COLOR_BLUE, COLOR_GREEN],
                       texto=[f'  {v:,}' for v in volumenes],
                       notas=notas, holgura=1.55, alto_fila=62))
        st.caption(f'De los {inicial:,} ofrecimientos, {volumenes[-1] / inicial * 100:.1f}% termina '
                   'en venta. La pérdida no está en el contacto —se alcanza al 85%— sino en el '
                   'cierre: de cada 100 clientes contactados, 37 compran.')

    with der:
        st.markdown('##### Medios probatorios')
        st.caption('Evidencia registrada de cada ofrecimiento.')
        tabla(f['medios_probatorios'])

    st.markdown('##### Dónde se cierra y dónde se pierde')
    st.caption('Funnel por situación comercial del cliente. La contactabilidad es homogénea '
               '(~85% en todos los segmentos): lo que separa es la tasa de cierre.')
    tabla(f['por_segmento'])

    st.markdown('##### Hallazgo: el rebate no está recuperando ventas')
    st.error(
        f"Ninguno de los {f['rebates']['total']:,} rebates registrados en el histórico terminó en "
        f"venta (0.00%), frente al {f['rebates']['tasa_sin_rebate']:.1f}% de los primeros "
        "ofrecimientos. Es la mayor oportunidad de mejora del proceso, y la razón por la que el "
        "motor elige la contraoferta según la objeción real en vez de repetir la siguiente del ranking."
    )

    with st.expander('Detalle de etapas y medios probatorios por canal'):
        tabla(f['etapas'], hide_index=True)
        tabla(f['probatorio_x_canal'])


# ------------------------------------------------------- vista segmentos ---

def vista_segmentos():
    st.markdown('## Segmentación inteligente de la base')
    st.caption('Vista de Producto y Marketing: cómo se reparte la base y a qué grupo conviene '
               'armarle campaña. El asesor no entra acá — él recibe el segmento ya aplicado en la '
               'ficha del cliente y en el argumentario.')

    with st.spinner('Calculando segmentos sobre la base...'):
        ov = obtener_segmentos(motor)

    tam, acc = ov['tamano_macro'], ov['aceptacion_macro']
    base = int(tam.sum())
    elegibles = int(tam.get('ELEGIBLE_MT', 0))
    tasa_elegible = float(acc.loc['ELEGIBLE_MT', 'tasa_aceptacion'])
    resto = acc.drop(index='ELEGIBLE_MT', errors='ignore')
    tasa_resto = ((resto['tasa_aceptacion'] * resto['ofrecimientos']).sum()
                  / resto['ofrecimientos'].sum())

    tiles([
        ('Base clasificada', f'{base:,}', 'clientes en los dos ejes', False),
        ('Elegibles MT hoy', f'{elegibles:,}', f'{elegibles / base * 100:.1f}% de la base', True),
        ('Aceptación del elegible', f'{tasa_elegible:.1f}%',
         f'vs. {tasa_resto:.1f}% del resto de la base', True),
        ('Perfiles de comportamiento', f'{ov["k"]}', 'consumo, riesgo y antigüedad', False),
    ])

    st.markdown('<div style="height:22px"></div>', unsafe_allow_html=True)

    # --- Eje A: dónde está el potencial -------------------------------------
    st.markdown('##### Dónde está el potencial · brecha hacia Movistar Total')
    st.caption('Cuántos clientes hay en cada situación comercial y cuánto aceptan históricamente. '
               'El color indica la acción: azul, el objetivo del desafío; verde, ya convertido; '
               'azul profundo, trabajable con una venta previa; gris, sin ruta directa.')
    macros = [m for m in MACRO_ORDEN if m in tam.index]
    grafico(barras(
        [MACRO_ETIQUETA[m] for m in macros],
        [int(tam[m]) for m in macros],
        [COLOR_MACRO[m] for m in macros],
        texto=[f'  {int(tam[m]):,} clientes' for m in macros],
        notas=[f"{acc.loc[m, 'tasa_aceptacion']:.1f}% de aceptación" if m in acc.index else None
               for m in macros],
    ))
    st.info(f'La elegibilidad para Movistar Total es el eje que más separa la base: el elegible '
            f'acepta {tasa_elegible:.1f}% frente a {tasa_resto:.1f}% del resto, con la misma '
            f'contactabilidad. Son {elegibles:,} clientes a los que hoy se les puede vender MT '
            f'sin ninguna venta previa.')

    # --- Eje B: cómo se comporta la base ------------------------------------
    st.markdown('##### Cómo se comporta la base · perfiles de consumo y riesgo')
    st.caption('El clustering no reemplaza al eje anterior: lo modula. Dentro de una misma '
               'situación comercial, el comportamiento decide el argumento y el orden de llamada.')
    perfil = ov['perfil_clusters'].set_index('nombre')
    accb = ov['aceptacion_comportamental']
    orden = [n for n in accb.index if n in perfil.index]
    grafico(barras(
        orden,
        [int(perfil.loc[n, 'n_clientes']) for n in orden],
        [COLOR_DEEP] * len(orden),
        texto=[f"  {int(perfil.loc[n, 'n_clientes']):,} clientes" for n in orden],
        notas=[f"{accb.loc[n, 'tasa_aceptacion']:.1f}% acepta  ·  "
               f"{perfil.loc[n, 'pct_elegible_mt']:.0f}% elegible MT" for n in orden],
        holgura=1.5,
    ))

    with st.expander('Perfil numérico de cada segmento'):
        columnas = ['n_clientes', 'pct_base', 'monto_facturado_prom', 'consumo_datos_gb_prom',
                    'ratio_consumo_gb', 'antiguedad_meses', 'dias_mora_prom', 'n_reclamos',
                    'pct_elegible_mt']
        tabla(perfil.loc[orden, columnas])
        st.caption('Se omiten interacciones por canal y score digital: valen 5.4 y 3.0 en los '
                   'cinco segmentos, así que ocupan ancho sin separar nada.')

    # --- Cruce de ejes ------------------------------------------------------
    cruce = ov['aceptacion_mt_cruce'].reset_index()
    macro_unico = cruce['macro_segmento'].nunique() == 1
    encabezado = (f'##### A quién llamar primero dentro de «{MACRO_ETIQUETA[cruce["macro_segmento"].iloc[0]]}»'
                  if macro_unico else '##### Cruce de ejes · aceptación de Movistar Total')
    st.markdown(encabezado)
    st.caption('Aquí la segmentación se vuelve accionable: a igual elegibilidad, el comportamiento '
               'separa quién acepta. Solo se muestran cruces con más de 50 ofrecimientos MT.')
    etiquetas = [c if macro_unico else f'{MACRO_ETIQUETA[m]} · {c}'
                 for m, c in zip(cruce['macro_segmento'], cruce['segmento_comportamental'])]
    grafico(barras(
        etiquetas, cruce['tasa_aceptacion_mt'].tolist(), [COLOR_BLUE] * len(cruce),
        texto=[f'  {v:.1f}% acepta MT' for v in cruce['tasa_aceptacion_mt']],
        notas=[f'{int(n):,} ofrecimientos' for n in cruce['ofrecimientos_mt']],
        holgura=1.55,
    ))

    with st.expander('Validación del número de clusters'):
        tabla(ov['metricas_k'])
        st.caption(
            'K se eligió por el codo de la curva de inercia. Davies-Bouldin decrece de forma '
            'monótona y el silhouette se mantiene plano (0.17–0.21), así que ninguno tiene óptimo '
            'interior: la base no presenta grupos naturales separables y el clustering opera como '
            'tiering de valor, riesgo y consumo, no como descubrimiento de grupos.'
        )


# ------------------------------------------------------------------ router ---

if vista == 'Asesor comercial':
    vista_asesor(cliente_id)
elif vista == 'Funnel E2E del ofrecimiento':
    vista_funnel()
else:
    vista_segmentos()
