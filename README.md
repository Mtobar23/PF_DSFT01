# 🛒 Sistema de Recomendación de Productos - Olist E-commerce
### *"La Parceira de los Emprendedores"*

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

[![OLIST.png](https://i.postimg.cc/RqxzdPRT/OLIST.png)](https://postimg.cc/gLs5kH1w)

Sistema de Machine Learning para predecir si un producto recibirá una reseña positiva (score ≥ 4) en la plataforma de e-commerce brasileña Olist, desarrollado como Proyecto Final del programa **Data Science Full Time 01**.


---

## 👥 Equipo de Desarrollo

| Integrante | Rol |
|------------|-----|
| **Santiago Joaquín Mozo** | Data Scientist |
| **José Ramírez Montoya** | Data Scientist |
| **Manuel Eduardo Tobar Barreto** | Data Scientist |
| **Alejandro Carrillo Vásquez** | Data Scientist |

<p align="center">
  <strong>DAVA - Financial & Data Consulting</strong>
</p>

---

## 📋 Tabla de Contenidos

- [Descripción del Proyecto](#-descripción-del-proyecto)
- [Dataset](#-dataset)
- [Estructura del Proyecto](#-estructura-del-proyecto)
- [Pipeline del Proyecto](#-pipeline-del-proyecto)
- [Modelos Evaluados](#-modelos-evaluados)
- [Resultados](#-resultados)
- [Dashboard Interactivo](#-dashboard-interactivo)
- [Instalación](#-instalación)
- [Uso](#-uso)
- [Tecnologías](#-tecnologías)
- [Conclusiones](#-conclusiones)
- [Autor](#-autores)

---

## 🎯 Descripción del Proyecto

Olist es la **"Parceira"** (socia) del emprendedor brasileño para fortalecer su negocio e impulsar su vida. Este proyecto implementa una solución técnica que permite crear un **sistema de recomendación de productos inteligente** basado en información de compras históricas.

### Objetivos

- Consolidar múltiples fuentes de datos mediante un pipeline ETL robusto
- Realizar análisis exploratorio para entender patrones de comportamiento
- Entrenar y comparar múltiples modelos de clasificación
- Desplegar un dashboard interactivo para predicciones en tiempo real

### Problema de Negocio

Olist conecta pequeñas empresas con grandes marketplaces en Brasil. El sistema predice si un producto será recomendado (review ≥ 4), permitiendo:
- Identificar productos con alto potencial de satisfacción
- Optimizar estrategias de inventario y marketing
- Mejorar la experiencia del cliente
- Aumentar la conversión de ventas y reducir el abandono del carrito

### Alcance del Sistema

| Aspecto | Descripción |
|---------|-------------|
| **Tipo de recomendación** | Productos similares, productos comprados por otros clientes, ranking por zona |
| **Entrada** | Información de 100,000 órdenes (2016-2018) |
| **Salida** | Recomendaciones basadas en tendencias, zona geográfica, período e ítems relacionados |

### KPIs del Proyecto

- **Recall**: ¿Cuántos productos que el usuario quería aparecieron en nuestras recomendaciones?
- **Precision**: ¿Qué porcentaje de productos recomendados resultó relevante?
- **F1-Score**: Balance óptimo entre Precision y Recall
- **Tiempo de Respuesta**: Velocidad de predicción para producción

---

## 📊 Dataset

El proyecto utiliza el dataset público de Olist disponible en [Kaggle](https://www.kaggle.com/olistbr/brazilian-ecommerce), que contiene ~100,000 pedidos realizados entre 2016 y 2018.

> **Referencia**: Olist, and André Sionek. (2018). Brazilian E-Commerce Public Dataset by Olist [Data set]. Kaggle. https://doi.org/10.34740/KAGGLE/DSV/195341

### Datasets Utilizados

| Dataset | Registros | Columnas | Descripción |
|---------|-----------|----------|-------------|
| orders | 99,441 | 8 | Transacciones principales (1 fila = 1 pedido) |
| customers | 99,441 | 5 | Datos demográficos de clientes |
| order_items | 112,650 | 7 | Detalle de productos por pedido |
| payments | 103,886 | 5 | Información financiera y métodos de pago |
| reviews | 99,224 | 7 | Satisfacción del cliente (reviews) |
| products | 32,951 | 9 | Catálogo de productos y categorías |

### Calidad de Datos

- **Total registros cargados**: 647,493
- **Datasets sin valores nulos**: customers, order_items, payments
- **Datasets con valores nulos**: orders (3% fechas), reviews (88% comentarios), products (2% categorías)

### Variable Objetivo

- **recommend_product**: Variable binaria (1 = review ≥ 4, 0 = review < 4)
- Distribución: **77% positivos**, 23% negativos

### Hallazgos del EDA

| Dimensión | Hallazgo Clave |
|-----------|----------------|
| **Patrones de Compra** | 74 categorías, top: `cama_mesa_banho` (10,953 ventas) |
| **Tasa de Recompra** | 12.44% (11,610 de 93,358 clientes) |
| **Satisfacción** | Review promedio: 4.09/5, 77% positivas |
| **Geografía** | Sudeste domina (68.7%), São Paulo lidera (42.1%) |
| **Temporalidad** | Lunes más activo (17,973 órdenes), tendencia creciente |
| **Pagos** | Tarjeta de crédito preferido (76.4%) |
| **Correlación** | Precio vs satisfacción: 0.003 (débil) |

---

## 📁 Estructura del Proyecto

```
olist-recommendation-system/
│
├── data/
│   ├── raw/                    # Datos originales de Kaggle
│   └── processed/              # Datos procesados (olist_clean_for_model.csv)
│
├── notebooks/
│   ├── ETL.ipynb              # Pipeline de extracción y transformación
│   └── EDA_Olist.ipynb        # Análisis exploratorio de datos
│
├── src/
│   ├── entrenamiento_modelos.py   # Sistema de entrenamiento
│   └── Dashboard.py               # Dashboard Streamlit
│
├── exported_model/
│   ├── best_model.pkl         # Modelo entrenado
│   ├── preprocessor.pkl       # Pipeline de preprocesamiento
│   ├── model_metadata.json    # Metadatos del modelo
│   └── example_usage.py       # Ejemplo de uso
│
├── images/
│   ├── model_comparison.png
│   ├── modelo_xgboost.png
│   ├── modelo_random_forest.png
│   ├── modelo_gradient_boosting.png
│   ├── modelo_decision_tree.png
│   └── modelo_logistic_regression.png
│
├── requirements.txt
└── README.md
```

---

## 🔄 Pipeline del Proyecto

### 1. ETL (Extract, Transform, Load)

El notebook `ETL.ipynb` implementa un pipeline robusto:

#### Extract
- Carga de 6 datasets del ecosistema Olist
- Validación de calidad de datos inicial

#### Transform
| Transformación | Descripción | Ejemplo |
|----------------|-------------|---------|
| **Filtrado Estratégico** | Solo pedidos `status='delivered'` | 96,478 pedidos (97.02%) |
| **Formato de Fecha** | Convertir a DD-MM-YYYY | 2017-10-02 10:30:00 → 02-10-2017 |
| **Numeración Secuencial** | Crear `order_number` | 1 a 96,478 |
| **Expansión de Estados** | Siglas a nombres completos | SP → Sao Paulo |
| **Renombrado** | Estandarizar nomenclatura | order_purchase_timestamp → order_purchase_datetime |

#### Arquitectura de Joins
```
ORDERS (delivered)
       |
       +----------------+----------------+
       |                |                |
   CUSTOMERS      ORDER_ITEMS        REVIEWS
                       |
                   PRODUCTS
                       |
                   PAYMENTS
```

#### Load - Resultado Final
- **Archivo**: `tabla_principal_etl.csv`
- **Registros**: 110,197
- **Columnas**: 12
- **Tamaño**: 75.53 MB

### 2. Análisis Exploratorio (EDA)

El notebook `EDA_Olist.ipynb` incluye:
- Análisis de distribuciones temporales y geográficas
- Top categorías por volumen y satisfacción
- Correlaciones entre variables
- Identificación de outliers y limitaciones

### 3. Entrenamiento de Modelos

El script `entrenamiento_modelos.py` implementa una clase `ModelTrainingSystem` que:
- Preprocesa datos (StandardScaler + OneHotEncoder)
- Entrena 5 modelos con validación cruzada
- Evalúa métricas de rendimiento
- Exporta el mejor modelo con metadatos

### 4. Dashboard

El archivo `Dashboard.py` despliega una interfaz Streamlit con:
- Predicción individual y por lotes
- Visualización de métricas del modelo
- Modo local y API

---

## 🤖 Modelos Evaluados

Se evaluaron 5 algoritmos de clasificación:

| Modelo | Descripción |
|--------|-------------|
| **XGBoost** | Gradient boosting optimizado |
| **Random Forest** | Ensemble de árboles de decisión |
| **Gradient Boosting** | Boosting secuencial |
| **Decision Tree** | Árbol de decisión simple |
| **Logistic Regression** | Modelo lineal baseline |

---

## 📈 Resultados

### Comparación de Métricas

| Modelo | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Tiempo (s) |
|--------|----------|-----------|--------|----------|---------|------------|
| **XGBoost** | **0.772** | **0.774** | **0.994** | **0.871** | 0.615 | 8 |
| Random Forest | 0.73 | 0.83 | 0.83 | 0.83 | 0.671 | 44 |
| Gradient Boosting | 0.77 | 0.77 | 0.99 | 0.87 | 0.614 | - |
| Decision Tree | 0.66 | 0.78 | 0.73 | 0.77 | 0.587 | 2 |
| Logistic Regression | 0.57 | 0.78 | 0.57 | 0.68 | 0.574 | 10 |

### 🏆 Mejor Modelo: XGBoost

![XGBoost Results](images/modelo_xgboost.png)

#### ¿Por qué XGBoost?

| Ventaja | Descripción |
|---------|-------------|
| ✅ **Alto Recall (99.4%)** | Captura casi todas las oportunidades de venta |
| ✅ **F1-Score Óptimo (0.871)** | Mejor balance precisión-recall |
| ✅ **Velocidad (13s)** | Entrenamiento rápido vs Random Forest (44s) |
| ✅ **Escalable** | Listo para producción |
| ✅ **Manejo de Desbalance** | Parámetro `scale_pos_weight` |

#### Impacto en el Negocio

| Mejora | Impacto |
|--------|---------|
| ↑ 99.4% | Oportunidades de venta capturadas |
| ↑ | Conversión de ventas |
| ↑ | Satisfacción del cliente |
| ↑ | Ventas cruzadas |
| ↓ | Abandono del carrito |
| ↓ | Tiempo de búsqueda del usuario |

### Visualización de Resultados por Modelo

<details>
<summary>Random Forest</summary>

![Random Forest](images/modelo_random_forest.png)
</details>

<details>
<summary>Gradient Boosting</summary>

![Gradient Boosting](images/modelo_gradient_boosting.png)
</details>

<details>
<summary>Decision Tree</summary>

![Decision Tree](images/modelo_decision_tree.png)
</details>

<details>
<summary>Logistic Regression</summary>

![Logistic Regression](images/modelo_logistic_regression.png)
</details>

---

## 🖥️ Dashboard Interactivo

Dashboard desarrollado en **Streamlit** con tres funcionalidades principales:

### Funcionalidades

| Tab | Descripción |
|-----|-------------|
| **Predicción Individual** | Ingresar características de un producto y obtener recomendación con probabilidad |
| **Predicción por Lotes** | Cargar CSV con múltiples productos para predicciones masivas |
| **Métricas del Modelo** | Visualizar rendimiento, características y comparación de métricas |

### Características del Dashboard

- **Modo Local**: Usa el modelo entrenado directamente
- **Modo API**: Conecta con servidor FastAPI para predicciones
- **Visualización**: Gauge de probabilidad e importancia de características
- **Exportación**: Descarga de resultados en CSV

### Características de Entrada

| Tipo | Variables |
|------|-----------|
| **Numéricas** | Precio, número de orden, año, mes, día, hora |
| **Categóricas** | Categoría del producto, estado, región, método de pago |

### Screenshot del Dashboard

<p align="center">
  <i>Sistema de Recomendación Olist - Predicción Individual</i>
</p>

El dashboard muestra:
- Información del modelo (XGBoost, F1-Score: 0.871, Exactitud: 0.772)
- Resultado de predicción con nivel de confianza
- Gráfico de probabilidad tipo gauge
- Importancia relativa de características

---

## ⚙️ Instalación

### Requisitos Previos

- Python 3.8+
- pip

### Pasos

1. Clonar el repositorio:
```bash
git clone https://github.com/tu-usuario/olist-recommendation-system.git
cd olist-recommendation-system
```

2. Crear entorno virtual:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Instalar dependencias:
```bash
pip install -r requirements.txt
```

### requirements.txt

```
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.3.0
xgboost>=2.0.0
streamlit>=1.28.0
plotly>=5.15.0
matplotlib>=3.7.0
seaborn>=0.12.0
```

---

## 🚀 Uso

### Entrenar el Modelo

```bash
python src/entrenamiento_modelos.py
```

### Ejecutar el Dashboard

```bash
streamlit run src/Dashboard.py
```

### Usar el Modelo Programáticamente

```python
import pickle
import pandas as pd

# Cargar modelo
with open('exported_model/best_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('exported_model/preprocessor.pkl', 'rb') as f:
    preprocessor = pickle.load(f)

# Preparar datos
input_data = {
    'price': 149.90,
    'product_category_name': 'cama_mesa_banho',
    'customer_state': 'SP',
    'payment_type': 'credit_card'
}

# Predecir
df = pd.DataFrame([input_data])
X = preprocessor.transform(df)
prediction = model.predict(X)
probability = model.predict_proba(X)[0][1]

print(f"Recomendación: {'Sí' if prediction[0] == 1 else 'No'}")
print(f"Probabilidad: {probability:.2%}")
```

---

## 🛠️ Tecnologías

- **Python 3.8+**: Lenguaje principal
- **Pandas & NumPy**: Manipulación de datos
- **Scikit-learn**: Pipeline de ML y modelos
- **XGBoost**: Modelo de clasificación
- **Streamlit**: Dashboard interactivo
- **Plotly & Matplotlib**: Visualizaciones
- **Jupyter Notebook**: Desarrollo y documentación

---

## 📌 Conclusiones

### Del ETL
- Base de datos amplia (~100k órdenes) estructurada en 9 datasets
- Datos limpios: sin duplicados, bajo porcentaje de nulos, tipos de datos correctos
- Se seleccionaron columnas clave para análisis y modelado

### Del EDA
- **Desbalance de categorías**: Oportunidad de mejora con técnicas de balanceo
- **Datos faltantes**: Tratamiento adecuado de valores nulos
- **Correlación débil precio-satisfacción**: El precio no determina la satisfacción

### Del Modelo
1. **XGBoost**

   a. *Exactitud (0.772)*
       77% de todas las predicciones son correctas
       De cada 100 recomendaciones, 77 están bien y 23 mal
      
   b. *Precisión (0.774)*
       Cuando el modelo dice "recomienda esto", 77% de las veces acierta
       De cada 100 cosas que recomienda, 77 son realmente buenas
        
   c.  *Recall (0.994)*
       El modelo detecta casi todo lo bueno (99.4%)
       De cada 100 productos que debería recomendar, encuentra 99
       Muy bueno para no perderse oportunidades

   d. *F1-Score (0.871)*
      Equilibrio entre precisión y recall (escala 0-1)
      0.871 es un buen puntaje general

   e. *ROC-AUC (0.615)*
      Capacidad para distinguir entre bueno y malo
      0.615 es moderado: mejor que adivinar (0.5) pero no excelente
   
2. **Dashboard** democratiza el acceso a las recomendaciones
  
3. **Arquitectura** lista para producción
  
4. **Impacto medible** en métricas de negocio

---

## 🚀 Próximos Pasos

| Fase | Acción |
|------|--------|
| **Corto plazo** | A/B testing en producción |
| **Mediano plazo** | Personalización avanzada por usuario |
| **Largo plazo** | Integración con catálogo en tiempo real |
| **Mejoras técnicas** | Optimización de hiperparámetros con Optuna, SMOTE para desbalance |

---

## 📦 Entregables

- ✅ Informe técnico del desarrollo del modelo
- ✅ Dashboard interactivo (Streamlit)
- ✅ API lista para integración
- ✅ Demo funcional

---

## 👥 Autores

**DAVA - Financial & Data Consulting**

| Nombre | GitHub | LinkedIn |
|--------|--------|----------|
| Alejandro Carrillo Vázquez | [@Tomsakoch0605](https://github.com/Tomsakoch0605) | [LinkedIn](https://www.linkedin.com/in/michel-alejandro-carrillo-vázquez-93658977) |
| Santiago Joaquín Mozo | [@SJMozo](https://github.com/SJMozo) | [LinkedIn](https://www.linkedin.com/in/santiago-joaquín-m-83323a37a) |
| José Ramírez Montoya | [@JoseMontoya21](https://github.com/JoseMontoya21) | [LinkedIn](https://www.linkedin.com/in/jose-montoya-03696321a/) |
| Manuel Eduardo Tobar Barreto | [@Mtobar23](https://github.com/Mtobar23) | [LinkedIn](https://www.linkedin.com/in/manueltobar/) |

📧 **Contacto**: infodatateam@DataVivaConsulting.com

---

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo [LICENSE](LICENSE) para más detalles.

---

⭐ Si este proyecto te resultó útil, ¡no olvides darle una estrella!
