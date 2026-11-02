Predicción de recupero de automotores robados
==============================

# 🚗 Modelado Predictivo de Recupero de Automotores Sustraídos en Argentina

## 📋 Descripción del Proyecto
Este proyecto aplica técnicas de **Aprendizaje Automático Supervisado** (Clasificación Binaria) para predecir la probabilidad de recupero de automotores denunciados por robo o hurto en la República Argentina, utilizando datos públicos oficiales de la Dirección Nacional de los Registros Nacionales de la Propiedad del Automotor y Créditos Prendarios (DNRPA).

El desarrollo se organiza bajo la estructura estándar de ciencia de datos **Cookiecutter Data Science** y utiliza la librería `scikit-learn` para el modelado predictivo.

---

## 🗃️ Origen del Dataset y Metadatos
* **Fuente Oficial:** Dirección Nacional de los Registros Nacionales de la Propiedad del Automotor y Créditos Prendarios (DNRPA), Ministerio de Justicia de la Nación.
* **Portal de Acceso:** Portal Nacional de Datos Abiertos (`datos.gob.ar`).
**Enlace de Descarga:** [Dataset Oficial - Robos y Recuperos de Autos](https://www.datos.gob.ar/dataset/robos-y-recuperos-de-autos/resource/938f7cde-c8fe-597f-82d5-163241691c9b)
* **Catálogo:** Trámites Registrales de Automotores — Seccionales de todo el país (`dnrpa-robos-recuperos-autos-2026.zip`).
* **Licencia:** Datos Abiertos Públicos (Gobierno de la República Argentina).

---

## 🔬 Consideraciones Metodológicas y Diseño del Dataset

Para garantizar el rigor metodológico y la prevención de sesgos en el entrenamiento, se implementaron las siguientes decisiones de diseño en el Notebook 1 (`eda1.ipynb`):

### 1. Consolidación de Datos
* Se procesan de forma automatizada los archivos mensuales en formato CSV contenidos en el comprimido comprimido oficial, unificando los registros históricos en un único DataFrame.

### 2. Horizonte Temporal de Seguimiento (180 días)
* **Filtro Anti-sesgo:** Para evitar etiquetar como "No recuperado" a un vehículo cuya denuncia sea reciente y aún se encuentre dentro del período habitual de búsqueda, se estableció una ventana de seguimiento de **180 días**.
* Se calculó la fecha límite (`FECHA_LIMITE_DENUNCIA = FECHA_CORTE_DATASET - 180 días`), excluyendo del entrenamiento aquellos trámites recientes que no hayan completado dicho horizonte temporal.

### 3. Construcción de la Variable Objetivo (`recuperado`)
* **Variable Objetivo (\\(y\\)):** Se generó a partir de la columna `tramite_tipo`:
  * \\(y = 1\\) (*Recuperado*): Trámites que contienen la denominación `'RECUPERO'`.
  * \\(y = 0\\) (*No recuperado*): Denuncias activas sin registro de recupero posterior.

### 4. Selección de Predictores y Prevención de *Data Leakage*
* **Predictores Válidos (\\(X\\)):** Se seleccionaron únicamente variables conocidas al momento de realizar la denuncia:
  * `automotor_anio_modelo` (Numérica / Antigüedad)
  * `automotor_origen` (Categórica: Nacional / Importado)
  * `automotor_tipo_descripcion` (Categórica: Sedán, Pick-up, Utilitario, Carga, etc.)
  * `registro_seccional_provincia` (Categórica: Provincia de radicación registral)
* **Preprocesamiento Aislado:** Todas las transformaciones (imputación con `SimpleImputer`, escalado con `StandardScaler` y codificación con `OneHotEncoder`) se integran dentro de un `Pipeline` y un `ColumnTransformer` que se ajustan (`fit`) **exclusivamente sobre el conjunto de entrenamiento (\\(X_{train}\\))**.
🗂️ Diccionario de Variables del Proyecto
1. Variables del Dataset Crudo (data/raw/)
Corresponde a los campos extraídos directamente del archivo comprimido de la DNRPA (dnrpa-robos-recuperos-autos-2026.zip).
Columna Original
Tipo de Dato Original
Descripción del Campo
Ejemplos / Valores
tramite_fecha
datetime64[ns]
Fecha de realización del trámite en la seccional registral.
2026-01-10, 2026-06-18
tramite_tipo
object (string)
Denominación del trámite registral (base para la variable objetivo).
'DENUNCIA DE ROBO O HURTO', 'RECUPERO'
automotor_anio_modelo
float64
Año de fabricación del automotor.
2015.0, 2021.0
automotor_origen
object (string)
Origen de fabricación del vehículo.
'Nacional', 'Importado'
automotor_tipo_descripcion
object (string)
Tipo de carrocería o formato del automotor.
'SEDAN 4 PUERTAS', 'PICK-UP', 'COUPE'
registro_seccional_provincia
object (string)
Provincia de radicación de la seccional registral.
'Buenos Aires', 'Córdoba', 'CABA'
registro_seccional_codigo
int64
Código de identificación interna del Registro Seccional.
1002, 2015

---

## 📊 Ficha Técnica del Dataset Procesado
Variables del Dataset Procesado (data/processed/dataset_recupero_automotores.csv)Variables seleccionadas para el modelado en la Notebook 1 tras aplicar el filtro de horizonte temporal de 180 días y eliminar nulos:VariableRol en ModeloTipo de DatoNaturaleza EstadísticaDescripción y Tratamiento en el Pipelineautomotor_anio_modeloPredictor ($X$)float64Numérica DiscretaRefleja la antigüedad del vehículo. Se imputa la mediana y se estandariza con StandardScaler.automotor_origenPredictor ($X$)objectCategórica NominalOrigen de fabricación (Nacional/Importado). Se codifica mediante OneHotEncoder.automotor_tipo_descripcionPredictor ($X$)objectCategórica NominalCarrocería/formato. Se imputa por la moda y se codifica con OneHotEncoder.registro_seccional_provinciaPredictor ($X$)objectCategórica NominalJurisdicción registral. Se codifica con OneHotEncoder.recuperadoObjetivo ($y$)int64Categórica BinariaClase $1$ (Positiva): Recuperado ($6.7\%$).<br>Clase $0$ (Negativa): No recuperado ($93.3\%$).

* **Distribución de Clases:** **93.3% No Recuperados (\\(y=0\\))** vs. **6.7% Recuperados (\\(y=1\\))** (Desbalance de clases severo).
* **Estrategia de Evaluación:** División 80/20 estratificada (`stratify=y`) evaluando modelos de **Regresión Logística** y **Árboles de Decisión** mediante métricas sensibles al desbalance (**Recall, Precision, F1-Score y ROC-AUC**).


Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
