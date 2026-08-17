"""
Motor de contraoferta (rebate).

Por qué esto no es un modelo predictivo:

`motivo_rechazo` en estos datos sigue una multinomial fija —precio 35%,
no_necesita 20%, ya_tiene_similar 15%, mal_momento 15%, no_confia 10%, otro 5%—
idéntica para cualquier tipo de oferta, canal, nivel de mora o facturación. Un
clasificador entrenado sobre ella no puede superar a "responder siempre precio".

Pero además el problema estaba mal planteado: en una interacción real el asesor
ESCUCHA la objeción, no necesita que se la predigan. Lo que sí necesita, y en
segundos, es qué responder y qué ofrecer a cambio. Eso es lo que resuelve este
módulo: dado el motivo real, elige la contraoferta adecuada dentro de las
recomendaciones ya rankeadas y entrega el argumento de rebate.
"""

MOTIVOS = ['precio', 'no_necesita', 'ya_tiene_similar', 'mal_momento', 'no_confia', 'otro']

MOTIVO_LABEL = {
    'precio': 'Precio — "está muy caro"',
    'no_necesita': 'No lo necesita — "no me hace falta"',
    'ya_tiene_similar': 'Ya tiene algo similar',
    'mal_momento': 'Mal momento — "ahora no"',
    'no_confia': 'Desconfianza — "no me convence"',
    'otro': 'Otro motivo',
}

# Estrategia por objeción: qué buscar en la alternativa y qué decir.
ESTRATEGIA = {
    'precio': {
        'criterio': 'mas_barata',
        'speech': 'Reconocer el presupuesto y bajar el ticket sin perder el beneficio '
                  'principal. Anclar en el ahorro mensual concreto, no en el precio total.',
    },
    'no_necesita': {
        'criterio': 'otra_categoria',
        'speech': 'No insistir con el mismo producto: cambiar de categoría y conectar '
                  'con un consumo que el cliente sí reconoce como propio.',
    },
    'ya_tiene_similar': {
        'criterio': 'otra_categoria',
        'speech': 'Aceptar que ya está cubierto en esa necesidad y mover la conversación '
                  'a una categoría que hoy no tiene contratada.',
    },
    'mal_momento': {
        'criterio': 'misma_oferta_otro_canal',
        'speech': 'No forzar el cierre. Acordar un segundo contacto y dejar la oferta '
                  'registrada; cambiar a un canal menos intrusivo para el reintento.',
    },
    'no_confia': {
        'criterio': 'misma_oferta_presencial',
        'speech': 'La objeción es de confianza, no de producto: sostener la misma oferta '
                  'y llevarla a un canal presencial, con condiciones y permanencia por escrito.',
    },
    'otro': {
        'criterio': 'siguiente_mejor',
        'speech': 'Sin objeción clara: ofrecer la siguiente mejor alternativa del ranking '
                  'e indagar antes de volver a proponer.',
    },
}

CANAL_ALTERNATIVO = {
    'Call Out': 'Digital',
    'Call In': 'Digital',
    'Digital': 'Tienda',
    'Tienda': 'Digital',
}


def elegir_rebate(motivo, oferta_rechazada, alternativas):
    """
    Elige la contraoferta dado el motivo real de rechazo.

    `alternativas` son las recomendaciones restantes, ya rankeadas por el motor.
    Devuelve un dict con la contraoferta, el canal y el argumento; si no hay
    alternativa que cumpla el criterio, cae a la siguiente mejor del ranking.
    """
    estrategia = ESTRATEGIA.get(motivo, ESTRATEGIA['otro'])
    criterio = estrategia['criterio']
    canal_original = oferta_rechazada.get('canal_sugerido')

    # Reintentar la misma oferta por otro canal: la objeción no es del producto.
    if criterio in ('misma_oferta_otro_canal', 'misma_oferta_presencial'):
        canal = 'Tienda' if criterio == 'misma_oferta_presencial' else CANAL_ALTERNATIVO.get(canal_original, 'Digital')
        return {
            'tipo': 'reintento',
            'oferta': oferta_rechazada,
            'canal': canal,
            'speech': estrategia['speech'],
            'detalle': f"Mantener {oferta_rechazada['nombre_oferta']} y reintentar por {canal}.",
        }

    if not alternativas:
        return {
            'tipo': 'sin_alternativa',
            'oferta': None,
            'canal': canal_original,
            'speech': estrategia['speech'],
            'detalle': 'No hay alternativas elegibles en el catálogo para este cliente.',
        }

    if criterio == 'mas_barata':
        candidatas = [a for a in alternativas if a['precio_mensual'] < oferta_rechazada['precio_mensual']]
        if candidatas:
            elegida = min(candidatas, key=lambda a: a['precio_mensual'])
            ahorro = oferta_rechazada['precio_mensual'] - elegida['precio_mensual']
            detalle = (f"{elegida['nombre_oferta']} — S/ {elegida['precio_mensual']:.2f} al mes, "
                       f"S/ {ahorro:.2f} menos que la oferta rechazada.")
        else:
            # Sin nada más barato, no se puede rebatir por precio: hay que decirlo,
            # no disfrazar la siguiente del ranking de alternativa económica.
            elegida = alternativas[0]
            detalle = (f"No hay ninguna alternativa más barata para este cliente. "
                       f"Negociar condiciones sobre la oferta original o proponer "
                       f"{elegida['nombre_oferta']} explicando el diferencial de valor.")

    elif criterio == 'otra_categoria':
        candidatas = [a for a in alternativas if a.get('tipo_oferta') != oferta_rechazada.get('tipo_oferta')]
        if candidatas:
            elegida = candidatas[0]
            detalle = f"{elegida['nombre_oferta']} — cambia de categoría respecto a lo que rechazó."
        else:
            elegida = alternativas[0]
            detalle = (f"Todas las alternativas elegibles son de la misma categoría. "
                       f"Proponer {elegida['nombre_oferta']} o cerrar la conversación "
                       f"y reabrir con otra necesidad en el próximo contacto.")

    else:  # siguiente_mejor
        elegida = alternativas[0]
        detalle = f"{elegida['nombre_oferta']} — siguiente mejor opción del ranking."

    return {
        'tipo': 'contraoferta',
        'oferta': elegida,
        'canal': elegida['canal_sugerido'],
        'speech': estrategia['speech'],
        'detalle': detalle,
    }
