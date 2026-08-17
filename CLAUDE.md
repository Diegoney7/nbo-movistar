# Desafío 02 — Personalización comercial inteligente

**Hackatón "AI Telecom Challenge"** · Organizado por Movistar y Universidad de Lima
*Motor de Next Best Offer con Movistar Total (MT) como caso de uso prioritario de blindaje.*

---

## 1. Objetivo del desafío

Diseñar una solución basada en IA que recomiende, de manera inteligente y personalizada, la **mejor oferta comercial para cada cliente** (*Next Best Offer*), considerando su perfil, comportamiento, necesidades, historial de consumo y el contexto y canal de atención.

Como caso de uso prioritario, la solución debe **impulsar y masificar Movistar Total (MT)** como palanca para blindar la planta de clientes. El motor debe identificar:

1. El **cliente potencial**
2. El **canal** por el que es más propenso a comprar
3. El **mensaje / oferta idónea** según su necesidad
4. La **probabilidad de aceptación**
5. El **motivo de rechazo** (con solución de rebate)
6. El **seguimiento E2E** del ofrecimiento hasta el resultado de venta

---

## 2. Tipo de propuestas que se valorarán

- Modelos de **recomendación de ofertas personalizadas** según el perfil del cliente, combinando datos de consumo, comportamiento, historial comercial y preferencias.
- **Priorización por probabilidad de aceptación** y **explicabilidad**: que la solución indique *por qué* se sugiere determinada oferta.
- Recomendación del **canal y momento idóneo**, y del **mensaje / speech** más adecuado según la necesidad del cliente.
- **Seguimiento E2E del ofrecimiento (funnel)**: clasificación del cliente → medios de contacto → mensaje → speech de rebate → contactabilidad real → medios probatorios (audios, registros de plataformas) → resultado de venta.
- **Aplicación al caso MT**: incrementar el conocimiento de la oferta, mejorar el mensaje de comunicación e incrementar las ventas sobre clientes potenciales, incluyendo la mejora del flujo de ofrecimiento digital.
- **Prototipos funcionales, mockups o dashboards** que permitan visualizar la recomendación generada por IA, y asistentes inteligentes para asesores comerciales o canales digitales.
- **Segmentación inteligente** de clientes basada en IA.
- Propuestas que mejoren la conversión comercial **sin afectar negativamente la experiencia del cliente**, con enfoques éticos y responsables en el uso de datos.

---

## 3. Contexto del negocio

- En telecomunicaciones se atiende diariamente a miles de clientes por múltiples canales (tiendas, call center, canales digitales, WhatsApp, apps móviles, campañas y postventa). En cada interacción se pueden ofrecer planes, beneficios, upgrades, equipos, servicios adicionales o paquetes, pero **no siempre es sencillo identificar cuál es la mejor oferta para cada cliente en el momento correcto**.
- Hoy muchas recomendaciones dependen de **reglas generales, campañas masivas, criterios manuales o la experiencia del asesor**. Una buena oferta no solo debe vender más: debe mejorar la experiencia, anticipar necesidades, reducir fricciones, evitar bajas y aumentar la fidelización.

### Caso de uso prioritario — Movistar Total (MT)

- Es el producto que **diferencia a Movistar de la competencia**: oferta de ahorro (hasta 50% vs. comprar productos por separado), más GB y descuentos por segmentación.
- El potencial es **toda la planta móvil y hogar** (ofrecer a cada cliente el producto que le falta para convertirse en MT).
- **Metas**: >50% de la venta hogar y >10% de la venta móvil con MT. Incrementar la participación de MT reduce el churn, mejora la morosidad y la permanencia, y genera mayor rentabilidad.
- MT se vende por todos los canales; plataformas internas: **DITO** (venta) y **Visor** (postventa / cross). El flujo de ofrecimiento (contacto → speech → vista de oferta → negociación → cierre) varía por canal.

---

## 4. Problema / oportunidad

### Dolores actuales

- Ofertas poco personalizadas y campañas masivas que no responden al perfil individual.
- Baja conversión en algunos canales.
- Dificultad para identificar venta cruzada o mejora de plan.
- Falta de priorización por probabilidad de aceptación.
- Asesores con información fragmentada y riesgo de ofrecer productos poco adecuados.

### Específicos del caso MT

- Aunque genera más beneficios y mejor comisión, **no es la primera opción de ofrecimiento del asesor**.
- El ofrecimiento es **reactivo** (no todos los clientes preguntan por MT).
- Buena parte de la planta (**+50 años**) es poco digital, lo que dificulta digitalizar o mejorar flujos.
- **No existen reportes unificados ni plataformas automáticas de escucha** que permitan identificar si un producto fue efectivamente ofrecido, ni trazar el ofrecimiento de extremo a extremo.

### La oportunidad

Una solución de IA que analice la información disponible del cliente y recomiende la oferta más adecuada, **explicando el motivo**, priorizando por probabilidad de aceptación, sugiriendo canal y mensaje, y dando **seguimiento E2E** del ofrecimiento hasta la venta.

---

## 5. Impacto esperado

Una solución conceptual o prototipo (motor de recomendación, modelo predictivo, asistente para asesores, dashboard o simulador de NBO) que demuestre cómo la IA recomienda la mejor oferta para cada cliente y, en particular, **cuándo esa mejor oferta es Movistar Total**.

### Entregable mínimo — la solución debe mostrar

| # | Elemento |
| --- | --- |
| 1 | Perfil resumido del cliente |
| 2 | Oferta recomendada |
| 3 | Motivo de la recomendación (explicabilidad) |
| 4 | Probabilidad estimada de aceptación o prioridad |
| 5 | Canal y momento sugerido para presentar la oferta |
| 6 | Beneficio esperado para cliente y negocio |

**Idealmente**, además: un reporte de funnel que trace el ofrecimiento desde la clasificación del cliente hasta el resultado de venta, incluyendo medios probatorios.

### Indicadores que podrían mejorar

- Tasa de conversión comercial e incremento de ventas
- Participación de MT en la venta (% venta hogar y % venta móvil con MT)
- ARPU (ingreso promedio por usuario)
- Reducción de churn / bajas y mejora de la permanencia
- NPS o satisfacción del cliente y efectividad de campañas
- Mayor personalización de la experiencia del cliente

---

## 6. Usuarios involucrados

| Usuario | Rol / necesidad |
| --- | --- |
| **Canales de venta y atención** | Tiendas, Call Out, Call In y Digital |
| **Cliente** | Experimenta la personalización directa: recibe ofertas alineadas a su consumo real vía App o Web |
| **Asesores** (incluye call center) | Necesitan que el sistema les indique de forma **rápida y clara** qué cliente necesita qué producto |
| **Producto y Marketing** | Usan los patrones descubiertos por la IA para diseñar nuevos paquetes y estrategias de precios |

---

## 7. Datos disponibles

### Qué existe actualmente

- **Perfil y relación comercial**: tipo de cliente, antigüedad, plan actual, monto mensual facturado, equipos adquiridos, ubicación geográfica aproximada.
- **Comportamiento y consumo**: consumo de datos, voz o servicios; información de producto por cliente (plan, producto, oferta sugerida, consumo de datos).
- **Historiales**: pagos, reclamos, compras o renovaciones, campañas recibidas, ofertas aceptadas o rechazadas, probabilidad de churn.
- **Actividad por canal**: ventas por canal, visitas y tipo de visitas, llamadas y tipo de llamadas, adquisición de la App.

### Qué se comparte con los participantes

Datasets simulados / anonimizados (sin DNI ni teléfono), catálogo ficticio de ofertas, historial ficticio de aceptación de campañas y reportes de actividad por canal en versión de ejemplo.

📄 **Detalle campo por campo:** ver [`DICCIONARIO.MD`](./DICCIONARIO.MD)

---

## 8. Enfoque esperado de IA

- **Motor de recomendación / scoring de ofertas**, combinado con motor de reglas cuando corresponda.
- **Machine learning supervisado**, segmentación inteligente y clustering de clientes.
- **Predicción** de probabilidad de aceptación de la oferta y predicción de churn.
- **IA generativa** para construir argumentos comerciales, mensajes y speech de rebate personalizados.
- **Asistente virtual / inteligente** para asesores y dashboard con reporte de funnel del ofrecimiento (clasificación → contacto → mensaje → contactabilidad → medio probatorio → resultado de venta).
- **Explicabilidad** de las recomendaciones y un **plan de implementación** (cronograma).

---

## 9. Consideraciones clave para el jurado

> El desafío **no consiste únicamente en lograr el algoritmo con mayor accuracy** en el dataset, sino en demostrar **cómo ese algoritmo resuelve el problema planteado**.

- Se valorará especialmente la **creación de variables ingeniosas** combinando datos.
- Y la **simplicidad para traducir los resultados técnicos** a una interfaz limpia, intuitiva y fácil de usar para un **asesor bajo presión de tiempo**.
- MT funciona como caso de uso prioritario y medible (blindaje / churn / permanencia), pero se valorará que el motor de recomendación sea **generalizable a otras ofertas del portafolio**.
- Las soluciones deben ser **simples, escalables y aplicables al contexto real** del sector, con un uso **ético y responsable** de los datos del cliente.