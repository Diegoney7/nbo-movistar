# AI Telecom Challenge 2026: Motor NBO & Movistar Total

## 🎯 El Desafío
Movistar y la Universidad de Lima plantearon un reto claro: **Personalización Comercial Inteligente**. 
El objetivo es doble: 
1. Construir un recomendador general de **Next Best Offer (NBO)**.
2. Impulsar estratégicamente el producto **Movistar Total (MT)** como palanca de blindaje y retención.

## 🏗️ Arquitectura de la Solución

Nuestra solución no es solo un modelo de Machine Learning, es un **Producto de Datos E2E** diseñado para el asesor comercial bajo presión:

*   **DataLoader & Feature Engineering:** Preprocesa datos históricos, evitando el *data leakage* temporal y calculando métricas de "brecha a Movistar Total" y ratios de consumo.
*   **Segmentación de dos ejes:** Cruza un **macro-segmento de brecha convergente** (regla de negocio: `YA_MT` / `ELEGIBLE_MT` / `BRECHA_HOGAR` / `BRECHA_MOVIL` / `NO_CONVERGENTE`) con un **segmento comportamental** (KMeans, K=5 elegido por el codo de la curva de inercia). El eje de regla es el que más discrimina —los elegibles MT aceptan 58.9% vs ~34% del resto— y el cruce afina la priorización: dentro de los elegibles, el segmento *Plan al Límite* acepta ofertas MT al 73.0% frente al 64.6% de *Reciente*.
*   **Política de segmento (`segment_policy.py`):** Traduce el segmento en decisiones auditables sobre el ranking: empuje diferenciado de MT, priorización de la oferta que abre el camino a MT, filtro de ofertas no contratables y **freno ético de upsell** sobre clientes con mora.
*   **Acceptance Model (LightGBM + calibración isotónica):** Predice la probabilidad de que un cliente acepte una oferta (excluyendo casos incontactables para evitar ruido). Está **calibrado**: la probabilidad que ve el asesor es la tasa real esperada, no un score.
*   **Channel Recommender:** Evalúa de manera "contrafactual" qué canal (Tienda, Digital, Call In, Call Out) maximiza la probabilidad de aceptación. **Solo afirma un canal cuando la diferencia es material**; si el modelo no distingue —lo habitual en esta base, donde el canal tiene aceptación plana— cae al canal habitual del cliente y lo declara en la interfaz, en vez de presentar como recomendación algorítmica un empate resuelto por orden de lista.
*   **Motor de Rebate:** Si el cliente rechaza, el asesor registra la objeción que escuchó y el motor elige la contraoferta adecuada para ese motivo (más barata, de otra categoría, o reintento por otro canal) con el argumento correspondiente.
*   **Explainability (SHAP):** Traduce el peso matemático de las variables a argumentos comerciales accionables en lenguaje natural (ej. "Su consumo de datos está al límite de lo que incluye su plan"). La traducción depende del **valor** de la variable, no solo de que SHAP la haya elegido: `n_reclamos` puede empujar la predicción hacia arriba precisamente porque el cliente *no* tiene reclamos, y un texto fijo por variable le decía "le ofrecemos una mejora tras sus contactos recientes" a quien nunca reclamó. Cuando el valor no sostiene un argumento honesto, la variable se descarta.
*   **Asistente de IA generativa (Google Gemini, `llm_assistant.py`):** El motor decide *qué* ofrecer; el LLM resuelve *cómo decirlo*. Redacta el argumentario completo —apertura, argumento central, tres beneficios, objeción anticipada con su respuesta, cierre y mensaje para app/WhatsApp— y el speech de rebate para la objeción que el asesor marcó. **No decide nada comercial:** recibe la decisión ya tomada y una ficha cerrada de datos verificados, con prohibición explícita de usar cualquier dato fuera de ella, salida estructurada por esquema JSON y degradación a plantilla determinista si no hay credencial o la API falla.
*   **Modelo de valor (`value_model.py`):** Cuantifica el beneficio en las dos monedas que importan — ahorro mensual y anual para el cliente, y ARPU incremental más valor esperado (`probabilidad × ARPU`) para el negocio. Para los segmentos con brecha, el beneficio principal es lo que la oferta **desbloquea**: la ruta a Movistar Total.
*   **Dashboard (Streamlit):** Tres vistas para tres usuarios. La **ficha del asesor** está ordenada exactamente por el entregable mínimo del reto (perfil → oferta → motivo → probabilidad → canal y momento → beneficio) y termina en el argumentario y el rebate; para gerencia, un **funnel E2E** que traza clasificación → ofrecimiento → contactabilidad → resultado, respaldado por los medios probatorios de cada contacto; para Producto y Marketing, la **matriz de segmentación**.

## 📊 Impacto y Métricas

### El techo de información del dataset, medido

Antes de optimizar el modelo medimos cuánta señal contiene realmente el dataset. La aceptación se explica esencialmente por tres variables (`oferta_es_mt`, `meses_moroso`, `n_reclamos`); el resto del catálogo es plano frente al target: canal, edad, consumo, asequibilidad y contactabilidad dan tasas idénticas. Un estimador **oráculo** construido con las tasas empíricas de esas tres variables alcanza **AUC 0.583** en el split temporal de test.

Ese es el límite de Bayes: no hay algoritmo que lo supere de forma amplia sin sobreajustar o filtrar información del futuro. Nuestro modelo llega a **AUC 0.586** — está en el techo.

### Por eso la métrica que optimizamos es la calibración

Con el AUC limitado por los datos, el valor está en que la probabilidad sea **verdadera**: el motor rankea por `probabilidad × valor de negocio` y le muestra un porcentaje explícito al asesor.

| Métrica (test, meses 5-6) | Antes | Ahora | Óptimo de Bayes |
| --- | --- | --- | --- |
| AUC | 0.587 | **0.586** | 0.583 |
| Brier score | 0.237 | **0.223** | 0.223 |
| Error de calibración (ECE) | 0.0146 | **0.0040** | — |
| Prob. predicha en ofertas MT | 84.6% | **69.4%** | 69.2% (real) |

El modelo anterior usaba `class_weight='balanced'` sobre un target 37/63 que no está desbalanceado: eso inflaba las probabilidades **15 puntos** por encima de la realidad y distorsionaba el ranking.

### Valor comercial

| Targeting | Aceptación | Lift vs. base (37.5%) |
| --- | --- | --- |
| Top 5% del ranking | 72.1% | **×1.92** |
| Top 10% del ranking | 67.3% | ×1.80 |
| Top 20% del ranking | 51.9% | ×1.38 |

*   **Regla de Negocio MT:** El motor prioriza (multiplicador de ranking) las ofertas MT para los clientes elegibles. Dado que **la aceptación de MT en clientes elegibles llega al 69.7%**, empujar estas ofertas asegura cruzar la meta de negocio de que el 50% de la venta hogar sea convergente.
### Qué NO es predecible en estos datos (y por qué lo decimos)

Auditamos cada componente contra su baseline honesto. Tres resultan ser ruido, y preferimos declararlo antes que vender como IA algo que no lo es:

| Componente | Hallazgo | Qué hicimos |
| --- | --- | --- |
| Motivo de rechazo | Multinomial fija (precio 35%, no necesita 20%, ya tiene similar 15%, mal momento 15%, no confía 10%, otro 5%), idéntica para cualquier oferta, canal, mora o facturación. El clasificador daba 14.8% de accuracy, **por debajo del azar** | El asesor marca la objeción que **escuchó** y el motor elige la contraoferta para ese motivo. Predecirlo era resolver el problema equivocado |
| Canal óptimo | Aceptación plana por canal (37.3%–37.6%). La diferencia mediana que el modelo produce entre canales es de **0.000 pp**: el "ganador" salía por orden de lista (Tienda en 36 de 40 casos) | Solo se afirma un canal cuando la diferencia supera 1 pp; si no, se usa el canal habitual del cliente y se declara |
| Contactabilidad | Constante en 84.8% para todo canal, perfil y nivel de actividad | No se modela |
| Momento óptimo | Aceptación de 37.15%–37.61% según día de la semana, plana por mes y **literalmente constante** por día del mes (37.47%) | Heurística de negocio declarada (canal, uso de app, ciclo de cobranza), marcada como regla y no como predicción |

### El hallazgo del funnel

Los **47.572 rebates** registrados en el histórico tienen una tasa de cierre del **0.00%**, frente al 46.1% de los primeros ofrecimientos. Tal como se ejecuta hoy, el rebate no recupera ninguna venta. Es la mayor oportunidad de mejora del proceso, y la razón de que nuestro motor de rebate elija la contraoferta según la objeción real en lugar de repetir la siguiente del ranking.

---

## 🎤 Guion para el Pitch (3 Minutos)

**[Hook - 30s]**
> *"Imaginen a un asesor en tienda, con una fila de 10 personas, teniendo exactamente 2 minutos para convencer a un cliente de llevarse un plan más caro. No necesita un dashboard complejo, necesita una bala de plata. Y eso es exactamente lo que construimos: un Motor NBO que no solo predice la mejor oferta, sino que le dice al asesor **por qué** ofrecerla y **cómo** rebatir si le dicen que no."*

**[El Modelo y MT - 1m]**
> *"Analizamos los datos y encontramos una mina de oro: solo el 13% de la base es elegible para Movistar Total, pero cuando se lo ofrecen, ¡el 70% acepta! Nuestro motor cruza todo el catálogo y empuja estratégicamente a estos clientes elegibles hacia el combo MT, simulando los 4 canales de venta para recomendar por dónde es más probable que firme.*
> *Y algo que nos diferencia: medimos el techo de información del dataset. Un oráculo perfecto sobre estos datos da 0.583 de AUC — nuestro modelo da 0.586, está en el techo. Así que en vez de perseguir un AUC que los datos no pueden dar, optimizamos lo que sí importa: que la probabilidad sea verdadera. Cuando el sistema dice 69%, la tasa real es 69%. El modelo anterior decía 85%."*

**[Explicabilidad y Cierre - 1m 30s]**
> *"Pero predecir no basta, hay que vender. Integramos SHAP para traducir matemáticas a lenguaje humano. En pantalla, el asesor no ve un 'feature importance de 0.15', ve un mensaje que dice: 'Ofrécelo porque el cliente está al límite de sus gigas'.*
> *Y damos el último paso: con Gemini, el asesor pulsa un botón y tiene el guion completo — cómo abrir, qué argumentar, qué objeción le van a poner y cómo responderla, más el mensaje listo para WhatsApp. El modelo no elige la oferta ni inventa un precio: solo redacta, sobre una ficha cerrada de datos ya verificados. Si se cae la API, el guion sale igual por plantilla.*
> *Y si el cliente rechaza, no adivinamos por qué: el asesor ya lo escuchó. Él marca la objeción, el motor elige la contraoferta para ese motivo —si fue precio, la alternativa más barata con el ahorro exacto; si fue mal momento, mantener la oferta y reintentar por otro canal— y la IA redacta el rebate.*
> *Con esta herramienta, pasamos de vender a ciegas a una personalización comercial verdaderamente inteligente."*

---

## 🚀 Cómo ejecutar

1. Instalar dependencias: `pip install -r requirements.txt`. Las versiones están **fijadas a las que produjeron los modelos** de `models_artifacts/`, porque un cambio de versión de scikit-learn o LightGBM puede romper la deserialización de los `.pkl`. (En particular se requiere `scikit-learn >= 1.6`: el modelo de aceptación usa `sklearn.frozen.FrozenEstimator` para la calibración isotónica). Las librerías de gráficos del EDA están aparte, en `requirements-notebooks.txt`.
2. Entrenar los modelos (opcional, ya vienen entrenados en `models_artifacts/`):

   ```bash
   python train_all.py
   ```

   Ejecuta segmentación → aceptación → rebate **en ese orden**, que no es intercambiable: el modelo de rebate consume el contrato de features (`model_features.pkl`, `model_categories.pkl`) que produce el de aceptación.
3. Configurar la API key de Gemini (opcional: sin ella la app funciona con argumentarios de plantilla). Se obtiene en [Google AI Studio](https://aistudio.google.com/apikey) y se puede entregar por cualquiera de estas tres vías:

   ```bash
   export GEMINI_API_KEY="tu_api_key"          # variable de entorno
   cp .streamlit/secrets.toml.example .streamlit/secrets.toml   # archivo de secretos
   ```

   …o pegarla en la barra lateral de la app (*Asistente de IA generativa → Configurar credencial*), donde queda solo en la sesión del navegador. **La credencial nunca se escribe en el código ni se sube al repositorio** (`.streamlit/secrets.toml` está en `.gitignore`).

   Para verificar la conexión desde la terminal antes de una demo:

   ```bash
   python probar_gemini.py            # cliente elegible MT de ejemplo
   python probar_gemini.py CLI000001  # un cliente concreto
   ```

4. Levantar la aplicación: `streamlit run app/streamlit_app.py`.

Para publicarla en una URL que pueda usar todo el equipo, ver [`DESPLIEGUE.md`](./DESPLIEGUE.md).

### Estructura

```
src/
  data_loader.py          Carga y normalización de los 3 CSV
  feature_engineering.py  Features derivadas + split temporal sin leakage
  segment_policy.py       Eje A (brecha MT), política de ranking y momento sugerido
  rebate_policy.py        Contraoferta según la objeción real del cliente
  value_model.py          Beneficio para el cliente y para el negocio
  nbo_engine.py           Orquestador: candidatos → scoring → política → explicación
  explainability.py       SHAP → argumentos comerciales en lenguaje natural
  llm_assistant.py        IA generativa (Gemini): argumentario de venta y speech de rebate
  models/
    segmentation.py       Eje B (KMeans) con validación de K
    acceptance_model.py   Probabilidad calibrada + análisis del techo de Bayes
    rebate_model.py       Prior de motivos de rechazo
    channel_recommender.py Evaluación contrafactual de canales
app/streamlit_app.py      Ficha del asesor + funnel E2E + vista de segmentos
.streamlit/config.toml    Tema de marca Movistar
probar_gemini.py          Verificación de la integración con Gemini
train_all.py              Pipeline de entrenamiento completo
```

### Cómo se controla el riesgo del LLM

Un modelo de lenguaje suelto frente a un cliente es un riesgo comercial y reputacional. Cuatro controles lo acotan:

| Control | Qué garantiza |
| --- | --- |
| El LLM no decide | Recibe la oferta, el canal, el momento y los beneficios ya calculados por el motor. Nunca elige la oferta ni calcula un precio |
| Ficha cerrada de datos verificados | Todas las cifras salen del catálogo y de `value_model.py`; la instrucción de sistema prohíbe usar cualquier dato fuera de la ficha |
| Salida estructurada por esquema | La interfaz nunca parsea texto libre: el modelo devuelve JSON conforme a un esquema declarado |
| Degradación a plantilla | Sin credencial o ante un fallo de la API, el mismo argumentario se construye de forma determinista y la interfaz lo declara. El asesor nunca se queda sin guion |

Además, el beneficio para el negocio (ARPU, valor esperado) se excluye a propósito del contexto que recibe el modelo, y la instrucción de sistema le prohíbe mencionar mora, reclamos, probabilidades o cualquier tecnicismo interno al cliente.
