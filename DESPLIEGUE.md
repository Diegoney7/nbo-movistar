# Despliegue — publicar el asistente para el equipo

Guía para dejar la app corriendo en una URL que puedan usar los compañeros del
equipo, sin que nadie tenga que instalar nada.

---

## Por qué no en Vercel

Vercel ejecuta **funciones serverless** de request/response. Streamlit es un
**servidor persistente**: el navegador mantiene abierta una conexión WebSocket
mientras dura la sesión, y esa conexión es la que redibuja la pantalla cuando el
asesor marca una objeción o pide el argumentario. Sin ese canal, la app no
funciona.

Aun resolviendo eso, hay dos límites duros:

| Límite de Vercel | Lo que necesita esta app |
| --- | --- |
| ~250 MB de dependencias por función | `lightgbm` + `shap` + `scikit-learn` + `pandas` pesan bastante más |
| Sin estado entre invocaciones | El motor carga 515 MB de datos y modelos una vez y los reutiliza |

Llevarla a Vercel significaría reescribir la interfaz en Next.js y montar el
motor en otro servidor aparte. La plataforma correcta para una app Streamlit es
un servicio que ejecute un proceso permanente.

---

## Opción elegida: Streamlit Community Cloud

Gratis, pensada para esto, y permite dejar la app **privada** invitando a los
compañeros por correo.

### Paso 1 — Subir el repositorio a GitHub

El repositorio es la carpeta `proyecto_nbo/` (ya está inicializada con su primer
commit). Crear un repositorio **privado** en GitHub —por ejemplo `nbo-movistar`,
sin README ni .gitignore, para que quede vacío— y luego:

```bash
cd proyecto_nbo
git remote add origin https://github.com/<tu-usuario>/nbo-movistar.git
git push -u origin main
```

Los datos pesan 62 MB en total y el archivo más grande son 45 MB, así que entran
en GitHub sin necesidad de Git LFS.

### Paso 2 — Crear la app en Community Cloud

1. Entrar a [share.streamlit.io](https://share.streamlit.io) con la misma cuenta
   de GitHub y autorizar el acceso a repositorios privados.
2. **Create app → Deploy a public app from GitHub** y completar:

   | Campo | Valor |
   | --- | --- |
   | Repository | `<tu-usuario>/nbo-movistar` |
   | Branch | `main` |
   | Main file path | `app/streamlit_app.py` |
   | Python version (*Advanced settings*) | **3.12** |

   La versión de Python importa: los modelos de `models_artifacts/` se
   serializaron con Python 3.12 y las versiones exactas que fija
   `requirements.txt`.

3. El primer despliegue tarda unos minutos: instala `packages.txt` (`libgomp1`,
   que LightGBM necesita en Linux) y luego `requirements.txt`.

### Paso 3 — Cargar la API key de Gemini

En *Advanced settings* al crear la app, o después en **Settings → Secrets**,
pegar:

```toml
GEMINI_API_KEY = "AIza..."

# Opcional: fija el modelo. Si se omite, usa el recomendado.
GEMINI_MODEL = "gemini-3.5-flash"
```

La key **no está** en el repositorio: `.streamlit/secrets.toml` está en
`.gitignore` y solo se versiona el ejemplo `secrets.toml.example`.

#### Qué cambia en pantalla cuando la key está en el servidor

La app detecta que la credencial viene del entorno o de los secrets y entra en
**modo asesor**: desaparecen el campo de API key, el botón de probar conexión y
el selector de modelo. El asesor entra, busca a su cliente y genera el
argumentario; no ve —ni puede tocar— nada de configuración.

Para abrir ese panel puntualmente, sin cambiar el código ni el despliegue,
agregar `?config=1` a la URL:

```
https://tu-app.streamlit.app/?config=1
```

Sobre el modelo: si la API key no habilita el que está fijado, la app **elige
sola el mejor disponible** en vez de mostrarle un error al asesor. Por eso el
selector no hace falta en el día a día.

### Paso 4 — Restringir el acceso al equipo

En **Settings → Sharing**, desactivar el acceso público y agregar los correos de
los compañeros. Solo ellos podrán abrir la URL, y solo ellos consumirán la cuota
de la API key.

### Actualizar la app

Cada `git push` a `main` redespliega automáticamente. Pero hay dos casos en los
que hace falta **Reboot app** desde *Manage app*, porque el redespliegue solo no
alcanza:

| Qué cambiaste | Por qué hay que reiniciar |
| --- | --- |
| `requirements.txt` o `packages.txt` | Para forzar la reinstalación limpia del entorno |
| Cualquier módulo de `src/` | Python cachea los módulos ya importados en `sys.modules`: Cloud recarga `app/streamlit_app.py` pero puede seguir usando la versión anterior de `src/`, y aparece un `ImportError` de una función que sí existe en el repositorio |

El segundo caso se reconoce fácil: el error menciona una función nueva y el
traceback apunta a un archivo de `/mount/src/...` que en GitHub sí la tiene.

---

## Consumo de recursos (medido)

| Momento | Memoria |
| --- | --- |
| Motor cargado (clientes + modelos + SHAP) | 474 MB |
| Con el funnel E2E y la segmentación calculados | 515 MB |

El límite de Community Cloud ronda 1 GB, así que entra con margen. Las
librerías se llevan la mayor parte (≈280 MB: pandas, scikit-learn y shap); los
datos ya están optimizados —las columnas de texto de baja cardinalidad se cargan
como `category`, lo que ahorró ~160 MB— y el histórico de 300 mil campañas se
lee bajo demanda y solo con las columnas de cada vista.

Si en algún momento la app se reinicia por memoria (por ejemplo, al abrirla
varias personas a la vez), la alternativa directa es **Hugging Face Spaces** con
SDK Streamlit: mismo repositorio, mismos archivos, 16 GB de RAM en el plan
gratuito.

---

## Sin desplegar: correrla en local

```bash
pip install -r requirements.txt
export GEMINI_API_KEY="tu_api_key"
streamlit run app/streamlit_app.py
```
