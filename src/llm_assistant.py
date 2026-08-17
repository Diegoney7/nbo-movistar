"""
Capa de IA generativa del motor NBO — Google Gemini.

Qué resuelve
------------
El motor decide QUÉ ofrecer, a QUIÉN, por qué CANAL, en qué MOMENTO y con qué
probabilidad. Lo que un modelo tabular no resuelve es CÓMO decirlo: el asesor
necesita, en segundos, la frase con la que abre, el argumento que conecta la
oferta con el consumo real del cliente, la respuesta a la objeción que va a
escuchar y el mensaje para el canal digital.

Ese es el punto del desafío que cubre este módulo: "IA generativa para construir
argumentos comerciales, mensajes y speech de rebate personalizados".

Cómo se controla el riesgo del LLM
----------------------------------
1. El modelo NO toma decisiones comerciales. Recibe la decisión ya tomada por el
   motor (oferta, canal, momento, probabilidad, beneficios calculados) y solo la
   redacta. Nunca elige la oferta ni calcula un precio.
2. Se le entrega una FICHA cerrada de datos ya verificados y se le prohíbe usar
   cualquier dato fuera de ella. Las cifras del speech vienen del catálogo y de
   value_model.py, no del modelo de lenguaje.
3. Salida estructurada (JSON con esquema declarado): la interfaz nunca parsea
   texto libre ni depende del formato que el modelo decida usar.
4. Degradación controlada: si no hay API key o la llamada falla, se construye el
   mismo argumentario con una plantilla determinista sobre los mismos datos, y la
   interfaz declara que está en modo plantilla. La aplicación nunca se queda sin
   argumentario ni muestra un error al asesor en plena venta.
"""

import json
import os
import time

# Modelos sugeridos, en el orden en que se ofrecen. Es solo el respaldo y el
# criterio de orden: la interfaz consulta a la API qué modelos habilita
# realmente la API key en uso (ver AsistenteComercial.listar_modelos), porque el
# catálogo de Google cambia y no todas las keys tienen acceso a lo mismo.
# Se prioriza Flash porque el asesor está frente al cliente: la latencia importa
# más que la elaboración.
MODELOS = {
    'gemini-3.5-flash': 'Rápido y sólido — recomendado para atención en vivo',
    'gemini-3.5-flash-lite': 'El más rápido y económico de la familia 3.5',
    'gemini-3.7-flash': 'El más capaz de los Flash — para preparar campañas',
    'gemini-3.6-flash': 'Generación anterior de Flash',
    'gemini-2.5-flash': 'Familia 2.5, por compatibilidad',
    'gemini-2.5-flash-lite': 'Familia 2.5 en su variante más rápida',
}
MODELO_POR_DEFECTO = 'gemini-3.5-flash'

# Modelos que no sirven para esta tarea aunque la key los habilite.
_EXCLUIR_DEL_LISTADO = ('image', 'embedding', 'tts', 'aqa', 'vision', 'live', 'native-audio')

GB_ILIMITADO = 9999


# --- Instrucción de sistema -------------------------------------------------
# Concentra las reglas de marca, de tono y de uso ético en un solo lugar
# auditable, igual que segment_policy.py hace con las reglas de negocio.

SYSTEM_INSTRUCTION = """\
Eres el asistente comercial de Movistar Perú. Apoyas a asesores de venta de \
tienda, call center y canales digitales, que trabajan con poco tiempo y frente \
al cliente.

Tu tarea es convertir la recomendación de un motor de Next Best Offer en un \
argumentario listo para decir en voz alta. No decides la oferta: ya está \
decidida y te llega en la ficha.

Reglas estrictas:
1. Usa EXCLUSIVAMENTE los datos de la ficha. Nunca inventes precios, gigas, \
plazos, promociones, permanencias, cobertura ni condiciones. Si un dato no está \
en la ficha, no existe.
2. Cita las cifras exactamente como aparecen en la ficha (soles con dos \
decimales, gigas tal cual). No redondees ni estimes.
3. Español peruano neutro y profesional. Trato de "usted". Frases cortas, \
directas y fáciles de leer en voz alta. Sin superlativos vacíos ni signos de \
exclamación múltiples.
4. Prohibido mencionar cualquier tecnicismo interno: probabilidad, score, \
modelo, algoritmo, segmento, cluster, SHAP, ranking, ARPU o "el sistema me \
dice". El cliente escucha a una persona, no a un motor.
5. Uso ético de los datos: nada de urgencia falsa, miedo, culpa ni presión. No \
menciones la mora, los reclamos ni el detalle de su consumo como si lo \
estuvieras vigilando; úsalos solo para elegir el tono y el beneficio a destacar.
6. Si la ficha indica mora, lidera con ahorro y evita empujar un ticket mayor.
7. Habla del beneficio para el cliente, nunca del beneficio para la empresa.
8. Devuelve siempre JSON válido conforme al esquema pedido, sin texto adicional.\
"""


# --- Esquemas de salida estructurada ----------------------------------------

SCHEMA_ARGUMENTARIO = {
    'type': 'OBJECT',
    'properties': {
        'lectura_cliente': {
            'type': 'STRING',
            'description': 'Una sola frase, para el asesor (no para el cliente), '
                           'que resume quién es este cliente y qué necesita.',
        },
        'apertura': {
            'type': 'STRING',
            'description': 'Una o dos frases para abrir la conversación por el canal '
                           'indicado, conectando con la situación del cliente. Sin mencionar la oferta todavía.',
        },
        'argumento_central': {
            'type': 'STRING',
            'description': 'Dos o tres frases que presentan la oferta y la conectan con '
                           'la necesidad concreta del cliente, incluyendo el precio y el beneficio principal.',
        },
        'beneficios_clave': {
            'type': 'ARRAY',
            'items': {'type': 'STRING'},
            'description': 'Exactamente tres bullets muy cortos (máximo 12 palabras cada uno) '
                           'que el asesor puede decir de corrido.',
        },
        'objecion_anticipada': {
            'type': 'STRING',
            'description': 'La objeción más probable de este cliente concreto, en sus palabras, entre comillas.',
        },
        'respuesta_objecion': {
            'type': 'STRING',
            'description': 'Cómo responder esa objeción en una o dos frases, con datos de la ficha.',
        },
        'cierre': {
            'type': 'STRING',
            'description': 'Pregunta de cierre concreta que empuja al siguiente paso.',
        },
        'mensaje_digital': {
            'type': 'STRING',
            'description': 'Mensaje listo para enviar por WhatsApp o app Movistar, máximo 300 '
                           'caracteres, con el beneficio y una llamada a la acción.',
        },
    },
    'required': ['lectura_cliente', 'apertura', 'argumento_central', 'beneficios_clave',
                 'objecion_anticipada', 'respuesta_objecion', 'cierre', 'mensaje_digital'],
}

SCHEMA_REBATE = {
    'type': 'OBJECT',
    'properties': {
        'reconocimiento': {
            'type': 'STRING',
            'description': 'Una frase que valida la objeción del cliente sin discutirla ni repetir el pitch.',
        },
        'argumento': {
            'type': 'STRING',
            'description': 'Dos frases que reencuadran la conversación según la estrategia indicada, con datos de la ficha.',
        },
        'propuesta': {
            'type': 'STRING',
            'description': 'La contraoferta dicha en voz alta, con su nombre y precio exactos. '
                           'Si la estrategia es reintentar la misma oferta por otro canal, propone ese segundo contacto.',
        },
        'cierre': {
            'type': 'STRING',
            'description': 'Pregunta de cierre o compromiso concreto de siguiente paso.',
        },
        'mensaje_seguimiento': {
            'type': 'STRING',
            'description': 'Mensaje de seguimiento para enviar después del contacto, máximo 300 caracteres.',
        },
    },
    'required': ['reconocimiento', 'argumento', 'propuesta', 'cierre', 'mensaje_seguimiento'],
}


# --- Resolución de credenciales ---------------------------------------------

def resolver_api_key(explicita: str = None) -> str:
    """
    Busca la API key en el orden: parámetro explícito → variable de entorno →
    secrets de Streamlit. Nunca se escribe la clave en el código ni en el repo.
    """
    if explicita and explicita.strip():
        return explicita.strip()

    for var in ('GEMINI_API_KEY', 'GOOGLE_API_KEY'):
        valor = os.environ.get(var)
        if valor and valor.strip():
            return valor.strip()

    try:  # st.secrets levanta excepción si no existe el archivo de secretos.
        import streamlit as st
        for var in ('GEMINI_API_KEY', 'GOOGLE_API_KEY'):
            if var in st.secrets:
                return str(st.secrets[var]).strip()
    except Exception:
        pass

    return None


def resolver_modelo(por_defecto: str = MODELO_POR_DEFECTO) -> str:
    """
    Modelo fijado por quien despliega, vía `GEMINI_MODEL` (variable de entorno o
    secrets). Permite dejar la app con un modelo elegido sin tocar el código ni
    mostrarle un selector al asesor.
    """
    valor = os.environ.get('GEMINI_MODEL')
    if valor and valor.strip():
        return valor.strip()

    try:
        import streamlit as st
        if 'GEMINI_MODEL' in st.secrets:
            return str(st.secrets['GEMINI_MODEL']).strip()
    except Exception:
        pass

    return por_defecto


def elegir_modelo(preferido: str, disponibles: list) -> str:
    """
    Resuelve qué modelo usar realmente.

    Si la API key no habilita el preferido, se cae al mejor sustituto entre los
    sugeridos y, en última instancia, al primero disponible. Sin esto, el asesor
    se encuentra con un error de "modelo no disponible" que no puede resolver.
    """
    if not disponibles or preferido in disponibles:
        return preferido
    for sugerido in MODELOS:
        if sugerido in disponibles:
            return sugerido
    return disponibles[0]


# --- Ficha de datos verificados ---------------------------------------------

def _gb_texto(gb) -> str:
    """Los planes hogar no traen GB: en ese caso no se dice nada de datos, en vez
    de afirmar '0 GB', que suena a carencia."""
    gb = float(gb or 0)
    if gb >= GB_ILIMITADO:
        return 'datos ilimitados'
    return f"{gb:.0f} GB" if gb > 0 else ''


def _mensaje_error(exc: Exception) -> str:
    """Traduce el error del SDK a algo accionable para quien opera la demo."""
    texto = str(exc)
    if 'API_KEY_INVALID' in texto or 'API key not valid' in texto:
        return 'La API key de Gemini no es válida. Revísela en la barra lateral.'
    if 'RESOURCE_EXHAUSTED' in texto or '429' in texto:
        return 'Se agotó la cuota de la API key de Gemini. Reintente en unos minutos.'
    if 'NOT_FOUND' in texto or '404' in texto:
        return 'El modelo seleccionado no está disponible para esta API key. Elija otro modelo.'
    if 'PERMISSION_DENIED' in texto or '403' in texto:
        return 'La API key no tiene permiso sobre la API de Gemini.'
    if isinstance(exc, json.JSONDecodeError):
        return 'El modelo devolvió una respuesta que no se pudo interpretar.'
    return f'{type(exc).__name__}: {texto}'


def construir_ficha(cliente: dict, rec: dict, segmento: dict = None,
                    plan_actual_nombre: str = None) -> dict:
    """
    Arma el único conjunto de datos que el LLM puede usar.

    Es el contrato de grounding del módulo: todo lo que el modelo puede afirmar
    tiene que estar acá, y todo lo que está acá viene del dataset, del catálogo o
    de un cálculo explícito del motor.

    El beneficio para el negocio (ARPU, valor esperado) se excluye a propósito:
    es información interna que no debe filtrarse al discurso que escucha el
    cliente.
    """
    beneficio = rec.get('beneficio') or {}
    beneficio_cliente = beneficio.get('cliente') or {}
    momento = rec.get('momento') or {}
    segmento = segmento or {}

    ahorro_pct = float(rec.get('ahorro_pct') or 0)
    bundle = rec.get('descripcion_bundle')

    oferta = {
        'nombre': rec.get('nombre_oferta'),
        'tipo': rec.get('tipo_oferta'),
        'precio_mensual_soles': round(float(rec.get('precio_mensual') or 0), 2),
        'es_movistar_total': bool(rec.get('es_mt')),
    }
    if _gb_texto(rec.get('gb_incluidos')):
        oferta['datos_incluidos'] = _gb_texto(rec.get('gb_incluidos'))
    if ahorro_pct > 0:
        oferta['ahorro_pct_vs_contratar_por_separado'] = ahorro_pct
    if bundle and bundle not in ('no_aplica', None):
        oferta['incluye'] = bundle

    ficha = {
        'cliente': {
            'tipo_de_linea': cliente.get('tipo_cliente'),
            'antiguedad_meses': int(cliente.get('antiguedad_meses') or 0),
            'rango_de_edad': cliente.get('edad_rango'),
            'departamento': cliente.get('ubicacion_departamento'),
            'plan_actual': plan_actual_nombre or cliente.get('plan_actual_id'),
            'factura_mensual_actual_soles': round(float(cliente.get('monto_facturado_prom') or 0), 2),
            'consumo_datos_gb_mes': round(float(cliente.get('consumo_datos_gb_prom') or 0), 1),
            'consumo_voz_min_mes': round(float(cliente.get('consumo_voz_min_prom') or 0)),
            'usa_la_app_movistar': bool(cliente.get('es_usuario_app')),
            'canal_habitual': cliente.get('canal_mas_usado'),
            'tiene_movil': bool(cliente.get('tiene_movil')),
            'tiene_internet_hogar': bool(cliente.get('tiene_internet_hogar')),
            'meses_con_mora': int(cliente.get('meses_moroso') or 0),
            'reclamos_registrados': int(cliente.get('n_reclamos') or 0),
        },
        'oferta_recomendada': oferta,
        'beneficios_ya_calculados': beneficio_cliente.get('beneficios', []),
        'razones_de_la_recomendacion': rec.get('motivos', []),
        'canal_de_contacto': rec.get('canal_sugerido'),
        'momento_sugerido': momento.get('momento'),
        'contexto_interno_no_mencionar_al_cliente': {
            'probabilidad_de_aceptacion': f"{float(rec.get('probabilidad') or 0):.0%}",
            'situacion_del_cliente': segmento.get('macro_etiqueta'),
            'accion_comercial': segmento.get('accion'),
            'pistas_de_tono': segmento.get('argumentos', []),
        },
    }
    return ficha


# --- Prompts ----------------------------------------------------------------

_GUIA_CANAL = {
    'Tienda': 'Conversación presencial: el cliente está delante, puede ver el equipo y firmar hoy.',
    'Call Out': 'Llamada saliente: el asesor interrumpe al cliente, hay que ganar los primeros 10 segundos.',
    'Call In': 'Llamada entrante: el cliente llamó por otro motivo, primero se resuelve lo suyo y luego se ofrece.',
    'Digital': 'Canal digital (app o WhatsApp): mensaje breve, autoexplicativo y con un solo paso a seguir.',
}


def _prompt_argumentario(ficha: dict) -> str:
    canal = ficha.get('canal_de_contacto') or 'Tienda'
    return (
        "FICHA VERIFICADA DEL CASO — única fuente de datos permitida:\n"
        f"{json.dumps(ficha, ensure_ascii=False, indent=2)}\n\n"
        f"CANAL DE CONTACTO: {canal}. {_GUIA_CANAL.get(canal, '')}\n\n"
        "Redacta el argumentario con el que el asesor va a presentar esta oferta. "
        "El argumento central debe conectar el consumo o la situación real del cliente "
        "con lo que la oferta le resuelve, y mencionar el precio mensual. "
        "La objeción anticipada debe ser la más probable para ESTE cliente en particular, "
        "no una genérica."
    )


def _prompt_rebate(ficha: dict, motivo_label: str, rebate: dict) -> str:
    contraoferta = rebate.get('oferta') or {}
    detalle_contra = {
        'tipo_de_respuesta': rebate.get('tipo'),
        'estrategia_definida_por_el_motor': rebate.get('speech'),
        'accion_concreta': rebate.get('detalle'),
        'canal_para_el_siguiente_contacto': rebate.get('canal'),
    }
    if rebate.get('tipo') == 'contraoferta' and contraoferta:
        detalle_contra['contraoferta'] = {
            'nombre': contraoferta.get('nombre_oferta'),
            'precio_mensual_soles': round(float(contraoferta.get('precio_mensual') or 0), 2),
            'tipo': contraoferta.get('tipo_oferta'),
        }
        if _gb_texto(contraoferta.get('gb_incluidos')):
            detalle_contra['contraoferta']['datos_incluidos'] = _gb_texto(contraoferta.get('gb_incluidos'))
        bundle_contra = contraoferta.get('descripcion_bundle')
        if bundle_contra and bundle_contra not in ('no_aplica', None):
            detalle_contra['contraoferta']['incluye'] = bundle_contra

    return (
        "FICHA VERIFICADA DEL CASO — única fuente de datos permitida:\n"
        f"{json.dumps(ficha, ensure_ascii=False, indent=2)}\n\n"
        f"EL CLIENTE RECHAZÓ LA OFERTA. Objeción que el asesor escuchó: {motivo_label}\n\n"
        "RESPUESTA QUE DEFINIÓ EL MOTOR (respétala, no propongas otra oferta):\n"
        f"{json.dumps(detalle_contra, ensure_ascii=False, indent=2)}\n\n"
        "Redacta el speech de rebate. Primero reconoce la objeción sin discutirla, "
        "luego reencuadra según la estrategia definida y recién ahí plantea la acción "
        "concreta con su precio exacto. Si la estrategia es reintentar por otro canal, "
        "no bajes el precio ni inventes una alternativa: acuerda el segundo contacto."
    )


# --- Plantillas de respaldo (sin IA) ----------------------------------------
# Mismos datos, redacción determinista. Es lo que se muestra cuando no hay API
# key o la llamada falla, siempre declarado como tal en la interfaz.

def _plantilla_argumentario(ficha: dict) -> dict:
    cli = ficha['cliente']
    of = ficha['oferta_recomendada']
    beneficios = ficha.get('beneficios_ya_calculados') or []
    razones = ficha.get('razones_de_la_recomendacion') or []
    canal = ficha.get('canal_de_contacto') or 'Tienda'

    apertura = {
        'Tienda': "Aprovechando que está aquí, revisé su cuenta y hay una opción que se "
                  "ajusta mejor a lo que usa hoy.",
        'Call Out': f"Le llamo de Movistar. Vi que tiene {cli['antiguedad_meses']} meses con "
                    f"nosotros y quería comentarle un beneficio disponible para su línea.",
        'Call In': "Antes de terminar, permítame comentarle algo que le puede convenir "
                   "según lo que consume cada mes.",
        'Digital': "Tenemos una opción pensada para su consumo actual.",
    }.get(canal, "Revisé su cuenta y hay una opción que se ajusta mejor a lo que usa hoy.")

    incluye = of.get('datos_incluidos') or of.get('incluye') or ''
    argumento = (
        f"Hoy paga S/ {cli['factura_mensual_actual_soles']:.2f} al mes y consume "
        f"{cli['consumo_datos_gb_mes']} GB. Con {of['nombre']} pasa a "
        f"S/ {of['precio_mensual_soles']:.2f} al mes"
        + (f" con {incluye}." if incluye else ".")
    )
    if razones:
        argumento += f" {razones[0]}"

    return {
        'lectura_cliente': (
            f"Cliente {cli['tipo_de_linea']} con {cli['antiguedad_meses']} meses de antigüedad, "
            f"factura S/ {cli['factura_mensual_actual_soles']:.2f} y consume "
            f"{cli['consumo_datos_gb_mes']} GB al mes."
        ),
        'apertura': apertura,
        'argumento_central': argumento,
        'beneficios_clave': beneficios[:3] or [of['nombre']],
        'objecion_anticipada': '"Está muy caro"',
        'respuesta_objecion': (
            "Comparemos contra lo que ya paga hoy: la diferencia mensual es menor de lo que "
            "parece y el beneficio queda fijo en su plan."
        ),
        'cierre': "¿Le parece si lo dejamos activo desde su próximo ciclo de facturación?",
        'mensaje_digital': (
            f"Hola, tenemos {of['nombre']} por S/ {of['precio_mensual_soles']:.2f} al mes"
            + (f" con {incluye}." if incluye else ".")
            + " Responda SÍ y lo activamos."
        )[:300],
    }


def _plantilla_rebate(ficha: dict, motivo_label: str, rebate: dict) -> dict:
    contra = rebate.get('oferta') or {}
    nombre = contra.get('nombre_oferta', 'la alternativa')
    precio = float(contra.get('precio_mensual') or 0)

    if rebate.get('tipo') == 'reintento':
        propuesta = (f"Dejo registrada la propuesta y lo retomamos por "
                     f"{rebate.get('canal')} cuando le quede cómodo.")
    elif rebate.get('tipo') == 'sin_alternativa':
        propuesta = "No tengo otra alternativa contratable ahora; revisemos condiciones sobre la misma oferta."
    else:
        propuesta = f"Le propongo {nombre}, a S/ {precio:.2f} al mes."

    return {
        'reconocimiento': "Entiendo su posición y es una objeción válida; no le voy a "
                          "insistir con lo mismo.",
        'argumento': rebate.get('speech', ''),
        'propuesta': propuesta,
        'cierre': "¿Le parece que avancemos con esta opción?",
        'mensaje_seguimiento': (f"Gracias por su tiempo. Le dejo la propuesta: {nombre} "
                                f"por S/ {precio:.2f} al mes. Quedo atento.")[:300],
    }


# --- Cliente de Gemini ------------------------------------------------------

class AsistenteComercial:
    """
    Envoltorio del modelo Gemini para uso comercial.

    Se instancia una vez por sesión. Si no hay credencial o el SDK falla, el
    objeto queda en modo plantilla: los métodos siguen devolviendo un
    argumentario utilizable, marcado con origen='plantilla'.
    """

    def __init__(self, api_key: str = None, modelo: str = MODELO_POR_DEFECTO):
        self.modelo = modelo
        self.api_key = resolver_api_key(api_key)
        self.error = None
        self._client = None

        if not self.api_key:
            self.error = 'No hay API key de Gemini configurada.'
            return

        try:
            from google import genai
            self._client = genai.Client(api_key=self.api_key)
        except ImportError:
            self.error = 'Falta la librería google-genai. Instalar con: pip install google-genai'
        except Exception as exc:
            self.error = f'No se pudo inicializar el cliente de Gemini: {exc}'

    @property
    def disponible(self) -> bool:
        return self._client is not None

    def verificar(self) -> tuple:
        """Prueba mínima de credencial y modelo. Devuelve (ok, mensaje)."""
        if not self.disponible:
            return False, self.error
        try:
            from google.genai import types
            config_kwargs = dict(max_output_tokens=64, temperature=0)
            pensamiento = self._config_pensamiento(types)
            if pensamiento is not None:
                config_kwargs['thinking_config'] = pensamiento

            self._client.models.generate_content(
                model=self.modelo,
                contents='Responde solo con la palabra: listo',
                config=types.GenerateContentConfig(**config_kwargs),
            )
            return True, f'Conexión correcta con {self.modelo}.'
        except Exception as exc:
            return False, _mensaje_error(exc)

    def listar_modelos(self) -> list:
        """
        Modelos de texto que ESTA API key puede usar con generateContent.

        Se consulta a la API en vez de mantener una lista fija en el código: el
        catálogo de Google cambia y dos keys distintas no habilitan lo mismo.
        Los modelos sugeridos en MODELOS se muestran primero, y el resto va
        ordenado alfabéticamente.
        """
        if not self.disponible:
            return []

        disponibles = []
        for modelo in self._client.models.list():
            acciones = modelo.supported_actions or []
            if acciones and 'generateContent' not in acciones:
                continue
            nombre = (modelo.name or '').split('/')[-1]
            if not nombre.startswith('gemini'):
                continue
            if any(excluido in nombre for excluido in _EXCLUIR_DEL_LISTADO):
                continue
            disponibles.append(nombre)

        disponibles = sorted(set(disponibles))
        sugeridos = [m for m in MODELOS if m in disponibles]
        return sugeridos + [m for m in disponibles if m not in sugeridos]

    def _config_pensamiento(self, types):
        """
        Reduce el razonamiento extendido al mínimo que acepte cada familia.

        La tarea es redacción acotada sobre datos ya decididos: pensar más no
        mejora el argumentario y sí agrega segundos con el cliente delante. El
        parámetro cambió entre familias (2.5 usa `thinking_budget`, 3.x usa
        `thinking_level`), y para un modelo desconocido no se manda nada.
        """
        if self.modelo.startswith('gemini-3'):
            return types.ThinkingConfig(thinking_level='MINIMAL')
        if self.modelo.startswith('gemini-2.5') and not self.modelo.endswith('pro'):
            return types.ThinkingConfig(thinking_budget=0)
        return None

    def _generar(self, prompt: str, schema: dict, temperatura: float = 0.7) -> dict:
        from google.genai import types

        config_kwargs = dict(
            system_instruction=SYSTEM_INSTRUCTION,
            temperature=temperatura,
            response_mime_type='application/json',
            response_schema=schema,
            max_output_tokens=2048,
        )
        pensamiento = self._config_pensamiento(types)
        if pensamiento is not None:
            config_kwargs['thinking_config'] = pensamiento

        inicio = time.perf_counter()
        try:
            resp = self._client.models.generate_content(
                model=self.modelo,
                contents=prompt,
                config=types.GenerateContentConfig(**config_kwargs),
            )
        except Exception as exc:
            # El usuario puede elegir cualquier modelo que habilite su key,
            # incluidos los que no existían al escribir esto. Si el rechazo es
            # por el control de razonamiento, se reintenta sin él antes de dar
            # la llamada por perdida.
            if 'thinking' not in str(exc).lower() or pensamiento is None:
                raise
            config_kwargs.pop('thinking_config')
            resp = self._client.models.generate_content(
                model=self.modelo,
                contents=prompt,
                config=types.GenerateContentConfig(**config_kwargs),
            )
        latencia = (time.perf_counter() - inicio) * 1000

        texto = (resp.text or '').strip()
        if not texto:
            raise ValueError('El modelo no devolvió contenido (posible bloqueo por filtros de seguridad).')

        return json.loads(texto), latencia

    def _ejecutar(self, prompt, schema, plantilla, temperatura=0.7) -> dict:
        """Llama al modelo y, ante cualquier fallo, cae a la plantilla."""
        if not self.disponible:
            return {'origen': 'plantilla', 'data': plantilla, 'error': self.error,
                    'modelo': None, 'latencia_ms': 0}
        try:
            data, latencia = self._generar(prompt, schema, temperatura)
            return {'origen': 'gemini', 'data': data, 'error': None,
                    'modelo': self.modelo, 'latencia_ms': latencia}
        except Exception as exc:
            return {'origen': 'plantilla', 'data': plantilla,
                    'error': _mensaje_error(exc), 'modelo': self.modelo,
                    'latencia_ms': 0}

    def argumentario(self, ficha: dict) -> dict:
        """Speech de venta personalizado para la oferta recomendada."""
        return self._ejecutar(
            _prompt_argumentario(ficha),
            SCHEMA_ARGUMENTARIO,
            _plantilla_argumentario(ficha),
            temperatura=0.7,
        )

    def rebate(self, ficha: dict, motivo_label: str, rebate: dict) -> dict:
        """Speech de rebate para la objeción que el asesor escuchó."""
        return self._ejecutar(
            _prompt_rebate(ficha, motivo_label, rebate),
            SCHEMA_REBATE,
            _plantilla_rebate(ficha, motivo_label, rebate),
            temperatura=0.6,
        )
