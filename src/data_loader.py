import pandas as pd
import os

# Columnas de texto con pocos valores distintos. Guardarlas como `category` en
# vez de `object` es lo que hace que la app entre en el límite de memoria de un
# hosting gratuito: en historial_campanias.csv (300k filas) estas columnas
# ocupan ~160 MB como texto suelto y ~2 MB como categorías, sin cambiar ningún
# resultado. Los identificadores de alta cardinalidad (cliente_id,
# ofrecimiento_id) se dejan como texto: no ahorran memoria y complican los
# merges.
CATEGORICAS_CLIENTES = [
    'tipo_cliente', 'oferta_hogar_id', 'plan_actual_id', 'edad_rango',
    'ubicacion_departamento', 'canal_mas_usado',
]
CATEGORICAS_CAMPANIAS = [
    'oferta_id', 'canal', 'resultado', 'motivo_rechazo', 'contactabilidad',
    'medio_probatorio', 'tipo_cliente', 'nombre_oferta', 'tipo_oferta',
]


def _rellenar_categoria(serie: pd.Series, valor: str) -> pd.Series:
    """Rellena nulos en una columna categórica agregando antes el valor como
    categoría: asignar uno que no exista levanta excepción."""
    if serie.isna().any():
        return serie.cat.add_categories([valor]).fillna(valor)
    return serie


def _leer(path, categoricas, usecols=None, parse_dates=None) -> pd.DataFrame:
    """
    Lee el CSV pidiendo las categorías directamente al parser.

    Convertir después de leer da el mismo resultado, pero obliga a materializar
    la columna de texto completa primero: en dataset_clientes.csv eso son ~90 MB
    de pico que acá no se pagan.
    """
    dtype = {c: 'category' for c in categoricas
             if usecols is None or c in usecols}
    return pd.read_csv(path, dtype=dtype, usecols=usecols, parse_dates=parse_dates)


class DataLoader:
    def __init__(self, data_dir='../data'):
        self.data_dir = data_dir

    def load_clientes(self) -> pd.DataFrame:
        path = os.path.join(self.data_dir, 'dataset_clientes.csv')
        df = _leer(path, CATEGORICAS_CLIENTES)
        df['tipo_cliente'] = _rellenar_categoria(df['tipo_cliente'], 'sin_movil')
        df['oferta_hogar_id'] = _rellenar_categoria(df['oferta_hogar_id'], 'sin_hogar')
        df['canal_mas_usado'] = _rellenar_categoria(df['canal_mas_usado'], 'ninguno')
        return df

    def load_ofertas(self) -> pd.DataFrame:
        path = os.path.join(self.data_dir, 'catalogo_ofertas_entrega.csv')
        df = pd.read_csv(path)
        # Handle nulls in hogar specific columns
        df['cluster_hogar'] = df['cluster_hogar'].fillna('no_aplica')
        df['descripcion_bundle'] = df['descripcion_bundle'].fillna('no_aplica')
        return df

    def load_campanias(self, usecols=None) -> pd.DataFrame:
        """
        Historial de campañas. `usecols` permite leer solo las columnas que
        necesita quien llama: la app no usa la fecha ni el motivo de rechazo, y
        evitar esas columnas ahorra memoria en el pico de carga.
        """
        path = os.path.join(self.data_dir, 'historial_campanias.csv')
        parse_dates = ['fecha'] if usecols is None or 'fecha' in usecols else None
        return _leer(path, CATEGORICAS_CAMPANIAS, usecols=usecols, parse_dates=parse_dates)

    def load_all(self):
        return self.load_clientes(), self.load_ofertas(), self.load_campanias()

if __name__ == "__main__":
    loader = DataLoader()
    clientes, ofertas, campanias = loader.load_all()
    print(f"Loaded {len(clientes)} clientes, {len(ofertas)} ofertas, {len(campanias)} campanias.")
