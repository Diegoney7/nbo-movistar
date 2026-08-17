# Diccionario de Datos — Desafío 02

**Hackathon AI Telecom Challenge 2026**
Desafío 02: **Personalización comercial inteligente (NBO + Movistar Total)**

> Guía de arranque para equipos participantes.

---

## 1. Qué se te pide (resumen del desafío)

Construir una solución de IA que recomiende, para cada cliente, la **mejor oferta comercial** (*Next Best Offer*):

- **Qué** ofrecerle
- **Por qué canal**
- **Con qué probabilidad** de aceptación
- **Qué hacer si la rechaza** (rebate)

Como caso de uso prioritario, la solución debe **impulsar Movistar Total (MT)**.

---

## 2. Qué se te entrega

3 archivos CSV, **100% sintéticos y anonimizados** (sin DNI, teléfono, nombre real ni dirección exacta), con **100,000 clientes** y **6 meses de historial** (enero–junio 2026).

| Archivo | Qué es | Filas |
| --- | --- | --- |
| `dataset_clientes.csv` | Perfil y comportamiento de cada cliente | 100,000 |
| `catalogo_ofertas_entrega.csv` | Portafolio de productos que se pueden ofrecer | 22 |
| `historial_campanias.csv` | Historial de ofrecimientos ya realizados, con su resultado | 300,112 |

---

## 3. `dataset_clientes.csv`

*(100,000 filas — 1 fila por cliente)*

Perfil de cada cliente: quién es, qué servicios tiene, y un resumen (promedios/conteos) de su comportamiento en los últimos 6 meses.

| Columna | Tipo | Descripción |
| --- | --- | --- |
| `cliente_id` | string (PK) | Identificador único del cliente |
| `tipo_cliente` | categórico | `prepago` / `postpago`. Nulo si el cliente no tiene línea móvil (el concepto solo aplica a la línea móvil) |
| `antiguedad_meses` | int | Antigüedad del cliente en meses |
| `tiene_movil` | bool | Tiene línea móvil |
| `tiene_hogar` | bool | Tiene algún servicio hogar (internet, TV y/o telefonía fija — ver `oferta_hogar_id` para el detalle) |
| `oferta_hogar_id` | string (FK → `catalogo_ofertas_entrega.oferta_id`) | Qué paquete hogar específico tiene el cliente. Nulo si `tiene_hogar = False` |
| `tiene_internet_hogar` | bool | Si ese paquete hogar incluye internet. **No es lo mismo que `tiene_hogar`**: un cliente puede tener hogar (ej. solo TV) sin tener internet |
| `es_movistar_total` | bool | El cliente **ya tiene** el paquete convergente Movistar Total |
| `elegible_mt` | bool | El cliente **cumple los requisitos** para Movistar Total (móvil + internet hogar + postpago) pero **todavía no lo tiene** — **este es el segmento prioritario del desafío** |
| `plan_actual_id` | string (FK → `catalogo_ofertas_entrega.oferta_id`) | Producto/plan principal actual del cliente |
| `monto_facturado_prom` | float | Monto mensual facturado promedio, en soles |
| `edad_rango` | categórico | Rango de edad (`18-25`, `26-35`, `36-45`, `46-55`, `56-65`, `65+`) |
| `ubicacion_departamento` | categórico | Departamento aproximado |
| `es_usuario_app` | bool | Usó la app Movistar en alguno de los últimos 3 meses |
| `consumo_datos_gb_prom` | float | Promedio mensual de GB consumidos (6 meses) |
| `consumo_voz_min_prom` | float | Promedio mensual de minutos de voz consumidos |
| `consumo_sms_prom` | float | Promedio mensual de SMS enviados |
| `uso_app_movistar_prom` | float | Promedio mensual de sesiones de app |
| `monto_facturado_prom_6m` | float | Promedio de monto facturado en los 6 meses (histórico de facturación, distinto de `monto_facturado_prom`, que es del plan actual) |
| `dias_mora_prom` | float | Promedio de días de mora por mes |
| `meses_moroso` | int | Cantidad de meses (de 6) en que el cliente cayó en mora (>15 días) |
| `n_reclamos` | int | Cantidad total de reclamos registrados en los 6 meses |
| `n_actividad_canal` | int | Cantidad total de interacciones (visitas, llamadas, sesiones de app, etc.) en los 6 meses |
| `canal_mas_usado` | categórico | Canal (`Tienda` / `Call In` / `Call Out` / `Digital`) donde el cliente tuvo más interacciones en los 6 meses. Nulo si no tuvo ninguna interacción registrada |

---

## 4. `catalogo_ofertas_entrega.csv`

*(22 filas — 1 fila por oferta)*

El portafolio completo de productos que se le puede ofrecer a un cliente: planes móviles, planes hogar, upgrades, equipos, paquetes adicionales y Movistar Total.

| Columna | Tipo | Descripción |
| --- | --- | --- |
| `oferta_id` | string (PK) | Identificador de la oferta |
| `nombre_oferta` | string | Nombre comercial |
| `tipo_oferta` | categórico | `plan_movil` / `plan_hogar` / `upgrade` / `equipo` / `paquete_adicional` / `movistar_total` |
| `segmento_objetivo` | categórico | `movil` / `hogar` / `ambos` |
| `es_movistar_total` | bool | Si esta oferta es una de las 3 variantes de Movistar Total |
| `precio_mensual` | float | Precio mensual en soles |
| `ahorro_pct` | int | % de ahorro estimado vs. comprar los productos por separado (solo aplica a las 3 ofertas Movistar Total). **Valor ilustrativo**, no representa una tarifa oficial confirmada — MT ofrece "hasta 50% de ahorro" según el material del desafío, pero no hay un desglose oficial por tier |
| `gb_incluidos` | int | GB incluidos en el plan (`9999` = ilimitado) |
| `cluster_hogar` | categórico | Solo para ofertas hogar: `mono` (1 servicio), `duo` (2), `trio` (3: internet + TV + fijo). Nulo para el resto |
| `descripcion_bundle` | string | Solo para ofertas hogar: qué incluye el bundle (ej. "Internet + TV"). Nulo para el resto. Nota: la telefonía fija nunca se vende sola — solo aparece combinada en un duo o trio |
| `descripcion_corta` | string | Texto descriptivo generado (nombre + GB + precio) |

---

## 5. `historial_campanias.csv`

*(300,112 filas — 1 fila por ofrecimiento realizado)*

Cada fila es una oferta que se le presentó a un cliente en una fecha y canal específicos, y el resultado de ese ofrecimiento. **Esta es la tabla pensada para entrenar/evaluar tu modelo de Next Best Offer.**

| Columna | Tipo | Descripción |
| --- | --- | --- |
| `ofrecimiento_id` | string (PK) | Identificador del evento de ofrecimiento |
| `cliente_id` | string (FK → `dataset_clientes.cliente_id`) | A qué cliente se le ofreció |
| `oferta_id` | string (FK → `catalogo_ofertas_entrega.oferta_id`) | Qué oferta se le presentó |
| `fecha` | fecha | Fecha del ofrecimiento |
| `canal` | categórico | Canal del ofrecimiento (`Tienda` / `Call In` / `Call Out` / `Digital`) |
| `resultado` | categórico | `aceptada` / `rechazada` / `pendiente` (pendiente = no hubo contactabilidad real, no se pudo saber el resultado). **Target sugerido para el modelo de probabilidad de aceptación** |
| `motivo_rechazo` | categórico | Solo si `resultado = rechazada`: `precio` / `no_necesita` / `ya_tiene_similar` / `mal_momento` / `no_confia` / `otro`. Nulo en cualquier otro caso |
| `es_rebate` | bool | Si hubo una contraoferta tras un rechazo |
| `contactabilidad` | categórico | `contactado` / `no_contactado` |
| `medio_probatorio` | categórico | `registro_plataforma` / `audio_llamada` / `chat_log` — evidencia de que el ofrecimiento ocurrió (trazabilidad end-to-end) |
| `tipo_cliente` | categórico | Copiado de `dataset_clientes` (evita tener que cruzar si solo necesitas esta columna) |
| `antiguedad_meses` | int | Copiado de `dataset_clientes` |
| `elegible_mt` | bool | Copiado de `dataset_clientes` |
| `es_movistar_total` | bool | Copiado de `dataset_clientes`. **Ojo**: indica si el **cliente** ya tenía MT *antes* de este ofrecimiento, no si la oferta presentada era MT (para eso ver `oferta_es_mt`) |
| `nombre_oferta` | string | Copiado de `catalogo_ofertas_entrega` |
| `tipo_oferta` | categórico | Copiado de `catalogo_ofertas_entrega` |
| `oferta_es_mt` | bool | Si la oferta presentada en este evento **es** una de las 3 variantes de Movistar Total |

> ⚠️ **No confundir `es_movistar_total` (atributo del cliente) con `oferta_es_mt` (atributo de la oferta presentada)** — es el error de lectura más fácil de cometer en esta tabla.

---

## 6. Relaciones entre los 3 archivos

```text
catalogo_ofertas_entrega (oferta_id)
        │
        ├── dataset_clientes.plan_actual_id
        ├── dataset_clientes.oferta_hogar_id
        └── historial_campanias.oferta_id

dataset_clientes (cliente_id)
        └── historial_campanias.cliente_id   (1 cliente : N ofrecimientos)
```