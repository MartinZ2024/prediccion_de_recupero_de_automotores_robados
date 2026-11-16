
# 🚗 Modelado Predictivo de Trámites de Recupero Automotor en Argentina

## 📋 Descripción del Proyecto y Contexto
Este proyecto de **Aprendizaje Automático Supervisado** tiene como objetivo desarrollar, entrenar y evaluar modelos predictivos de clasificación binaria (**Regresión Logística** y **Árbol de Decisión**) para identificar características vehiculares y territoriales asociadas administrativamente a trámites registrales de **'RECUPERO'** en la República Argentina, utilizando datos públicos abiertos provenientes de la **Dirección Nacional de los Registros Nacionales de la Propiedad del Automotor y de Créditos Prendarios (DNRPA)**, dependiente del Ministerio de Justicia de la Nación.

El robo y sustracción de automotores constituye una problemática patrimonial y de seguridad ciudadana relevante. Contar con modelos predictivos en el ámbito registral permite caracterizar qué tipos de vehículos y qué regiones concentran determinadas dinámicas de recupero administrativo, facilitando el análisis de patrones del sector.

---

## 🎯 Formulación de Objetivos

### Objetivo General
Desarrollar, entrenar y validar modelos de aprendizaje automático supervisado con la librería `scikit-learn` para predecir la probabilidad de que un trámite registral corresponda a un **Recupero Automotor** a partir de atributos técnicos del vehículo y variables geográficas registrales.

### Objetivos Específicos
1. **Adquisición y Caracterización:** Descargar y documentar formalmente el dataset público oficial desde el portal `datos.gob.ar`.
2. **Análisis Exploratorio de Datos (EDA):** Examinar distribuciones, cuantificar el desbalance de clases y analizar el porcentaje de nulos y la proporción de trámites por provincia.
3. **Preprocesamiento y Prevención de *Data Leakage*:** Construir un flujo automatizado mediante `Pipeline` y `ColumnTransformer` que aplique imputación, escalado (`StandardScaler`) y codificación categórica (`OneHotEncoder`) aprendiendo parámetros exclusivamente sobre los datos de entrenamiento.
4. **Entrenamiento y Comparación:** Implementar clasificadores supervisados con enfoques teóricos complementarios (Regresión Logística y Árbol de Decisión con `max_depth=5`), incorporando `class_weight='balanced'`.
5. **Evaluación Rigurosa:** Validar sobre un conjunto de prueba independiente y estratificado ($X_{test}$) mediante matrices de confusión, **Recall**, **Precision**, **F1-Score** y **ROC-AUC**, interpretando sus alcances y limitaciones de acuerdo con las pautas docentes.

---
   
## 📁 Estructura del Repositorio
```text
El proyecto está estructurado siguiendo la plantilla estándar de ciencia de datos (*Cookiecutter Data Science*):
    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- Dataset limpio post-EDA (dataset_recupero_automotores.csv)
    │   └── raw            <- Dataset crudo original (CSV consolidados / ZIP)
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks/
    │    ├── 1_eda_exploracion_v2.ipynb       <- Análisis Exploratorio de Datos (EDA) y exportación del CSV limpio
    │    │  
    │    └── 2_preprocesamiento_modelos.ipynb <- Preprocesamiento con Pipelines, entrenamiento y evaluación    
    │                         
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │   │
    │   Informe_tecnico.md <- Descripción data set y modelo.
    ├── requirements.txt   <- Librerías y dependencias necesarias para reproducir el entorno
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
```


# 🗃️ Origen del Dataset, Metadatos y Accesos

* **Fuente Oficial:** Dirección Nacional de los Registros Nacionales de la Propiedad del Automotor y Créditos Prendarios (DNRPA), Ministerio de Justicia de la Nación.
* **Portal de Acceso:** Portal Nacional de Datos Abiertos (`datos.gob.ar`).
* **Catálogo:** [Trámites de Robos y Recuperos de Autos](https://www.datos.gob.ar/dataset/robos-y-recuperos-de-autos) (`dnrpa-robos-recuperos-autos-2026.zip`).
* **Licencia:** Datos Abiertos Públicos (Gobierno de la República Argentina).
* **Enlaces Internos a Datos del Repositorio:**
  * 📁 Dataset Crudo: [`data/raw/`](data/raw/) (24.718 registros, 25 columnas).
  * 📁 Dataset Nuevos_Datos Crudos: [`data/raw/`](data/raw/)(dnrpa-robos-recuperos-autos-202512.csv) (2.801 registros, 25 columnas)
  * 📁 Dataset Procesado: [`data/processed/dataset_recupero_automotores.csv`](data/processed/dataset_recupero_automotores.csv) (24.178 registros, 5 columnas).
  * 📁 Reports: Informe_tecnico.md (Informe técnico del modelo y prueba) 
```

## Enlace al video explicativo del proyecto



