"""
Verificación de la integración con Google Gemini, fuera de la interfaz.

Sirve para confirmar antes de una demo que la credencial funciona y que el
argumentario se genera con datos reales del motor:

    export GEMINI_API_KEY="tu_api_key"
    python probar_gemini.py                 # cliente de ejemplo elegible MT
    python probar_gemini.py CLI000001       # un cliente concreto

Si la key no está configurada, el script muestra el argumentario de plantilla,
que es exactamente lo que vería el asesor en ese caso.
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from nbo_engine import NBOEngine
from llm_assistant import AsistenteComercial, construir_ficha, MODELO_POR_DEFECTO
from rebate_policy import MOTIVO_LABEL

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def titulo(texto):
    print(f'\n{texto}\n' + '-' * len(texto))


def main():
    asistente = AsistenteComercial(modelo=MODELO_POR_DEFECTO)

    titulo('1. Credencial')
    if asistente.disponible:
        ok, mensaje = asistente.verificar()
        print(f'   {"OK" if ok else "ERROR"}: {mensaje}')
    else:
        print(f'   Sin IA: {asistente.error}')
        print('   El motor seguirá funcionando con argumentarios de plantilla.')

    titulo('2. Motor NBO')
    motor = NBOEngine(
        data_dir=os.path.join(BASE_DIR, 'data'),
        artifacts_dir=os.path.join(BASE_DIR, 'models_artifacts'),
    )
    nombres = dict(zip(motor.ofertas['oferta_id'], motor.ofertas['nombre_oferta']))

    if len(sys.argv) > 1:
        cliente_id = sys.argv[1].strip().upper()
    else:  # por defecto, un elegible MT: es el caso de uso prioritario del reto
        elegibles = motor.clientes_feat[motor.clientes_feat['macro_segmento'] == 'ELEGIBLE_MT']
        cliente_id = elegibles.iloc[0]['cliente_id']

    fila = motor.clientes_feat[motor.clientes_feat['cliente_id'] == cliente_id]
    if fila.empty:
        print(f'   Cliente {cliente_id} no encontrado.')
        return 1

    cliente = fila.iloc[0]
    recs = motor.get_recommendations(cliente_id, top_n=8)
    mejor = recs[0]
    segmento = motor.get_client_segment(cliente_id)
    print(f'   Cliente {cliente_id} · {segmento["macro_etiqueta"]} · {segmento["segmento_comportamental"]}')
    print(f'   Oferta recomendada: {mejor["nombre_oferta"]} · S/ {mejor["precio_mensual"]:.2f} '
          f'· probabilidad {mejor["probabilidad"]:.1%} · canal {mejor["canal_sugerido"]}')

    ficha = construir_ficha(cliente.to_dict(), mejor, segmento,
                            plan_actual_nombre=nombres.get(cliente['plan_actual_id']))

    titulo('3. Ficha enviada al modelo (única fuente de datos permitida)')
    print(json.dumps(ficha, ensure_ascii=False, indent=2))

    titulo('4. Argumentario de venta')
    resultado = asistente.argumentario(ficha)
    print(f'   origen: {resultado["origen"]} · modelo: {resultado["modelo"]} '
          f'· {resultado["latencia_ms"] / 1000:.1f} s')
    if resultado['error']:
        print(f'   aviso: {resultado["error"]}')
    for clave, valor in resultado['data'].items():
        if isinstance(valor, list):
            print(f'\n   {clave}:')
            for item in valor:
                print(f'     - {item}')
        else:
            print(f'\n   {clave}:\n     {valor}')

    titulo('5. Speech de rebate (objeción: precio)')
    rebate = motor.get_rebate(recs, 'precio')
    resultado_reb = asistente.rebate(ficha, MOTIVO_LABEL['precio'], rebate)
    print(f'   origen: {resultado_reb["origen"]} · contraoferta del motor: {rebate["detalle"]}')
    if resultado_reb['error']:
        print(f'   aviso: {resultado_reb["error"]}')
    for clave, valor in resultado_reb['data'].items():
        print(f'\n   {clave}:\n     {valor}')

    return 0


if __name__ == '__main__':
    sys.exit(main())
