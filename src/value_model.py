"""
Beneficio esperado de una recomendación, para el cliente y para el negocio.

Cubre el punto 6 del entregable mínimo del desafío. Traduce la recomendación a
las dos monedas que importan en la mesa: cuánto gana el cliente y cuánto gana la
compañía.

Todo lo que hay acá son cuentas explícitas sobre el catálogo, no estimaciones de
modelo. Cuando un supuesto es una convención comercial (por ejemplo, qué ofertas
reemplazan la factura y cuáles se suman), está declarado y es auditable.
"""

# Cómo impacta cada tipo de oferta sobre la factura del cliente:
#   'reemplaza' -> sustituye lo que el cliente paga hoy (planes y convergencia)
#   'suma'      -> se cobra además de la factura actual (add-ons y equipos)
IMPACTO_FACTURA = {
    'movistar_total': 'reemplaza',
    'plan_movil': 'reemplaza',
    'plan_hogar': 'reemplaza',
    'upgrade': 'suma',
    'paquete_adicional': 'suma',
    'equipo': 'suma',
}

GB_ILIMITADO = 9999


# Ahorro de la variante MT más accesible, para dimensionar el premio que espera
# al cliente que hoy está a un servicio de ser elegible.
MT_ENTRADA_PRECIO = 149.9
MT_ENTRADA_AHORRO_PCT = 20


def beneficio_cliente(oferta, cliente, macro=None):
    """
    Qué gana el cliente. Devuelve una lista de beneficios concretos y, cuando
    aplica, el ahorro mensual en soles.
    """
    beneficios = []
    ahorro_mensual = 0.0

    precio = float(oferta.get('precio_mensual') or 0)
    ahorro_pct = float(oferta.get('ahorro_pct') or 0)
    gb_oferta = float(oferta.get('gb_incluidos') or 0)
    consumo = float(cliente.get('consumo_datos_gb_prom') or 0)
    factura_actual = float(cliente.get('monto_facturado_prom') or 0)

    # Ahorro de convergencia: ahorro_pct es el descuento frente a contratar los
    # productos por separado, así que el precio suelto se reconstruye desde el
    # precio del paquete.
    if ahorro_pct > 0 and ahorro_pct < 100:
        precio_separado = precio / (1 - ahorro_pct / 100)
        ahorro_mensual = precio_separado - precio
        beneficios.append(
            f"Ahorra S/ {ahorro_mensual:.2f} al mes (S/ {ahorro_mensual * 12:.2f} al año) "
            f"frente a contratar los mismos servicios por separado."
        )

    # Datos: ilimitado, o cuántos GB gana sobre lo que hoy consume.
    if gb_oferta >= GB_ILIMITADO:
        beneficios.append("Datos ilimitados: deja de preocuparse por quedarse sin GB.")
    elif gb_oferta > 0:
        delta = gb_oferta - consumo
        if delta > 0:
            beneficios.append(
                f"{gb_oferta:.0f} GB incluidos: {delta:.1f} GB más de los que consume hoy "
                f"({consumo:.1f} GB al mes)."
            )
        else:
            beneficios.append(
                f"{gb_oferta:.0f} GB incluidos (hoy consume {consumo:.1f} GB al mes)."
            )

    if oferta.get('tipo_oferta') == 'movistar_total':
        beneficios.append("Una sola factura y un solo punto de atención para móvil y hogar.")

    # Si reemplaza la factura y cuesta menos que lo que paga hoy, es ahorro directo.
    if IMPACTO_FACTURA.get(oferta.get('tipo_oferta')) == 'reemplaza' and factura_actual > 0:
        diferencia = factura_actual - precio
        if diferencia > 0:
            beneficios.append(
                f"Paga S/ {diferencia:.2f} menos al mes que su factura actual "
                f"(S/ {factura_actual:.2f})."
            )

    # Camino a la convergencia: para quien está a un servicio de ser elegible MT,
    # el beneficio principal no es el producto en sí, es lo que desbloquea.
    if macro == 'BRECHA_HOGAR' and oferta.get('tipo_oferta') == 'plan_hogar':
        ahorro_futuro = MT_ENTRADA_PRECIO * (MT_ENTRADA_AHORRO_PCT / 100) / (1 - MT_ENTRADA_AHORRO_PCT / 100)
        beneficios.append(
            f"Al contratar hogar queda habilitado para Movistar Total, con un ahorro "
            f"de al menos S/ {ahorro_futuro:.2f} al mes sobre el paquete convergente."
        )
    elif macro == 'BRECHA_MOVIL' and oferta.get('tipo_oferta') == 'plan_movil':
        ahorro_futuro = MT_ENTRADA_PRECIO * (MT_ENTRADA_AHORRO_PCT / 100) / (1 - MT_ENTRADA_AHORRO_PCT / 100)
        beneficios.append(
            f"Con una línea móvil postpago queda habilitado para Movistar Total, con un "
            f"ahorro de al menos S/ {ahorro_futuro:.2f} al mes sobre el paquete convergente."
        )

    bundle = oferta.get('descripcion_bundle')
    if bundle and bundle not in ('no_aplica', None):
        beneficios.append(f"Incluye {bundle} en un solo paquete.")

    if not beneficios:
        beneficios.append("Complementa los servicios que ya tiene contratados.")

    return {'beneficios': beneficios, 'ahorro_mensual': ahorro_mensual}


def beneficio_negocio(oferta, cliente, probabilidad):
    """
    Qué gana el negocio. ARPU incremental y su valor esperado, que es la cifra
    con la que se prioriza una cartera: lo que se espera capturar por contacto.
    """
    precio = float(oferta.get('precio_mensual') or 0)
    factura_actual = float(cliente.get('monto_facturado_prom') or 0)
    tipo = oferta.get('tipo_oferta')

    if IMPACTO_FACTURA.get(tipo) == 'suma':
        arpu_incremental = precio
        nota = 'Se cobra adicional a la factura actual.'
    else:
        arpu_incremental = precio - factura_actual
        nota = f'Reemplaza la factura actual de S/ {factura_actual:.2f}.'

    valor_esperado = probabilidad * arpu_incremental

    return {
        'arpu_incremental': arpu_incremental,
        'arpu_incremental_anual': arpu_incremental * 12,
        'valor_esperado_mensual': valor_esperado,
        'valor_esperado_anual': valor_esperado * 12,
        'nota': nota,
        'es_convergente': tipo == 'movistar_total',
    }


def resumen_beneficio(oferta, cliente, probabilidad, macro=None):
    """Beneficio para ambos lados, listo para mostrar en la ficha del asesor."""
    return {
        'cliente': beneficio_cliente(oferta, cliente, macro),
        'negocio': beneficio_negocio(oferta, cliente, probabilidad),
    }
