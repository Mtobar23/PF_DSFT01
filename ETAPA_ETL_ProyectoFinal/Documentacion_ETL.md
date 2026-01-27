# 📊 Reporte Técnico - Pipeline ETL de Olist

## 📋 Resumen Ejecutivo

Este documento presenta el proceso ETL (Extract, Transform, Load) implementado para consolidar datos del ecosistema de e-commerce Olist en una tabla analítica unificada.

**Objetivo Principal:** Integrar 6 fuentes de datos diferentes y generar una tabla maestra que permita análisis estratégicos sobre pedidos entregados exitosamente.

**Resultado:** Dataset consolidado de **110,197 registros** y **12 columnas**, exportado como `tabla_principal_etl.csv`.

---

## 🎯 Contexto del Proyecto

**Olist** es una plataforma brasileña de e-commerce que conecta pequeñas empresas con grandes marketplaces. Los datos están distribuidos en múltiples archivos que requieren integración para obtener valor analítico.

### Preguntas de Negocio que Responde el Dataset
- ¿Qué categorías de productos generan mejores reviews?
- ¿Cuál es la preferencia de pago por estado?
- ¿Cómo se relaciona el precio con la satisfacción del cliente?
- ¿Dónde está concentrado el negocio geográficamente?

---

## 📦 1. EXTRACT - Extracción de Datos

Se cargaron **6 datasets** del ecosistema Olist:

| Dataset | Registros | Columnas | Descripción |
|---------|-----------|----------|-------------|
| `orders` | 99,441 | 8 | Transacciones principales (1 fila = 1 pedido) |
| `customers` | 99,441 | 5 | Datos demográficos de clientes |
| `order_items` | 112,650 | 7 | Detalle de productos por pedido |
| `payments` | 103,886 | 5 | Información financiera y métodos de pago |
| `reviews` | 99,224 | 7 | Satisfacción del cliente (reviews) |
| `products` | 32,951 | 9 | Catálogo de productos y categorías |

### Calidad de Datos Inicial
- **Total de registros cargados:** 647,493
- **Datasets sin valores nulos:** `customers`, `order_items`, `orders`
- **Datasets con valores nulos:**
  - `payments`: 1% en método de pago
  - `reviews`: 0.03% en reviews
  - `products`: 2% en categorías

---

## 🔧 2. TRANSFORM - Transformación de Datos

### 2.1 Filtrado Estratégico

**Decisión de Negocio:** Solo incluir pedidos con `status = 'delivered'`

**Justificación:**
- Representan transacciones completas (revenue real)
- Tienen reviews válidas (feedback genuino)
- Reflejan la experiencia completa del cliente

**Resultado del Filtrado:**
- Pedidos originales: **99,441**
- Pedidos delivered: **96,478** (97.02%)
- Pedidos excluidos: **2,963** (cancelados, en proceso, etc.)

### 2.2 Arquitectura de Joins

```
                ORDERS (delivered)
                      |
     +----------------+----------------+
     |                |                |
CUSTOMERS      ORDER_ITEMS         REVIEWS
                      |
                 PRODUCTS
                      |
                  PAYMENTS
```

**Estrategia:** Left joins para preservar todos los pedidos delivered.

**Secuencia de Joins:**
1. `Orders + Customers` → 96,478 filas
2. `+ Order Items` → 110,197 filas (expansión por múltiples ítems)
3. `+ Products` → 110,197 filas
4. `+ Reviews` → 110,197 filas
5. `+ Payments` → 110,197 filas (final)


### 2.3 Transformaciones Adicionales

| Transformación | Descripción | Ejemplo |
|----------------|-------------|---------|
| **Formato de Fecha** | Convertir a DD-MM-YYYY (sin hora) | 2017-10-02 10:30:00 → 02-10-2017 |
| **Numeración Secuencial** | Crear `order_number` basado en fecha de compra | 1 a 96,478 |
| **Expansión de Estados** | Convertir siglas a nombres completos | SP → Sao Paulo |
| **Renombrado** | Estandarizar nomenclatura de columnas | `order_purchase_timestamp` → `order_purchase_datetime` |

---

## 📊 3. LOAD - Resultado Final

### Estructura del Dataset Final

**Archivo:** `tabla_principal_etl.csv`
- **Registros:** 110,197
- **Columnas:** 13
- **Tamaño:** 75.53 MB en memoria

### Esquema de Columnas

| Columna | Tipo | Descripción | Nulos |
|---------|------|-------------|-------|
| `order_number` | int64 | Numeración secuencial (1-96,478) | 0% |
| `order_id` | object | ID único del pedido | 0% |
| `product_id` | object | ID único del producto | 0% |
| `price` | float64 | Precio del producto | 0% |
| `product_category_name` | object | Categoría del producto | 1.4% |
| `order_purchase_datetime` | object | Fecha de compra (DD-MM-YYYY) | 0% |
| `orders_status` | object | Estado del pedido (100% delivered) | 0% |
| `orders_customer_id` | object | ID del cliente en el pedido | 0% |
| `customer_unique_id` | object | ID único del cliente | 0% |
| `customer_state` | object | Estado del cliente (nombre completo) | 0% |
| `review_score` | float64 | Calificación del cliente (1-5) | 0.8% |
| `payment_type` | object | Método de pago | 0.0% |


---

## 🔍 4. Insights Principales

### 4.1 Categorías de Productos

**Top 5 Categorías por Volumen:**
1. `cama_mesa_banho` - 10,953 pedidos (9.9%)
2. `beleza_saude` - 9,465 pedidos (8.6%)
3. `esporte_lazer` - 8,431 pedidos (7.7%)
4. `moveis_decoracao` - 8,160 pedidos (7.4%)
5. `informatica_acessorios` - 7,644 pedidos (6.9%)

**Hallazgo:** Estas 5 categorías representan el **43.5%** del volumen total.

### 4.2 Satisfacción del Cliente

**Review Score Promedio:** 4.08/5.0

**Categorías con MAYOR Satisfacción (min. 50 reviews):**
- `livros_importados` - ⭐ 4.51
- `livros_interesse_geral` - ⭐ 4.51
- `construcao_ferramentas_ferramentas` - ⭐ 4.44

**Categorías con MENOR Satisfacción:**
- `moveis_escritorio` - ⭐ 3.51 (1,654 reviews)
- `telefonia_fixa` - ⭐ 3.76
- `fashion_roupa_masculina` - ⭐ 3.76

**Correlación Precio-Satisfacción:** 0.003 (casi nula)
> 💡 **Insight:** El precio NO determina la satisfacción. La calidad del servicio y producto son más importantes.

### 4.3 Métodos de Pago

**Distribución General:**
- `credit_card` - 76.4%
- `boleto` - 20.3%
- `voucher` - 1.8%
- `debit_card` - 1.5%

**Preferencia por Estado (Top 5):**
- Todos los estados principales prefieren tarjeta de crédito (71-79%)

### 4.4 Distribución Geográfica

**Top 3 Estados:**
1. São Paulo - 42.1%
2. Rio de Janeiro - 12.8%
3. Minas Gerais - 11.7%

**Total Top 3:** 66.7% del negocio

> ⚠️ **Alerta:** ALTA concentración geográfica - Riesgo de dependencia regional.

### 4.5 Análisis de Precio

**Distribución por Segmento de Precio:**
- R$ 0-50: 38,530 pedidos (35.0%) | ⭐ 4.08
- R$ 50-100: 32,376 pedidos (29.4%) | ⭐ 4.06
- R$ 100-200: 26,356 pedidos (23.9%) | ⭐ 4.11
- R$ 200-500: 9,845 pedidos (8.9%) | ⭐ 4.11
- R$ 500+: 3,090 pedidos (2.8%) | ⭐ 4.10

---

## 💡 5. Recomendaciones Estratégicas

### Para Marketing
1. **Priorizar stock y campañas** en las categorías top (cama/mesa/baño, belleza/salud)
2. **Promover tarjeta de crédito** como método principal de pago (ya es dominante)
3. **Desarrollar estrategia de expansión** hacia estados subrepresentados

### Para Operaciones
1. **Investigar causas de baja satisfacción** en `moveis_escritorio` (1,654 reviews negativas)
2. **Implementar controles de calidad** más estrictos en categorías problemáticas
3. **Usar categorías top como benchmarks** de excelencia operativa

### Para Producto
1. **Expandir catálogo de libros** (alta satisfacción, bajo volumen - oportunidad)
2. **Revisar propuesta de valor** en muebles de oficina y telefonía
3. **Enfocarse en experiencia de entrega** (el precio no afecta la satisfacción)

### Para Análisis Futuro
1. **Análisis de retención:** Identificar clientes recurrentes
2. **Análisis temporal:** Detectar estacionalidad por categoría
3. **Modelado predictivo:** Predecir probabilidad de review positivo
4. **Optimización logística:** Analizar tiempos de entrega por región

---

## 📈 6. Métricas Clave del Proceso ETL

| Métrica | Valor |
|---------|-------|
| **Datasets integrados** | 6 |
| **Registros procesados (total)** | 647,493 |
| **Registros en dataset final** | 110,197 |
| **Tasa de completitud** | 97.0% (solo delivered) |
| **Productos únicos** | 32,216 |
| **Clientes únicos** | 93,358 |
| **Pedidos únicos** | 96,478 |
| **Categorías de productos** | 73 |
| **Estados cubiertos** | 27 |
| **Rango de fechas** | 01-01-2018 a 31-12-2017 |

---

## 🛠️ 7. Stack Tecnológico

**Lenguaje:** Python 3.10  
**Librerías Principales:**
- `pandas` ≥ 1.3.0 - Manipulación de datos
- `numpy` ≥ 1.21.0 - Operaciones numéricas
- `datetime` - Manejo de fechas

**Formato de Salida:** CSV  
**Optimización:** Configurado para datasets < 10M registros

---

## 📚 8. Notas Técnicas

### Decisiones de Diseño
- **Left joins:** Preservan todos los pedidos delivered como eje principal
- **Agregaciones:** Evitan duplicados por múltiples pagos/reviews
- **Formato de fecha:** Facilita lectura humana (DD-MM-YYYY)
- **Valores nulos:** Solo 1.4% en categorías (aceptable para análisis)

### Limitaciones Conocidas
- Reviews ausentes en 0.8% de los pedidos
- Categorías de producto no disponibles en 1.4% de los ítems
- Dataset concentrado en pedidos delivered (excluye 3% cancelados)

### Escalabilidad
Para volúmenes mayores (>10M registros), considerar:
- Procesamiento por chunks: `pd.read_csv(chunksize=100000)`
- Uso de Dask o PySpark para procesamiento distribuido
- Almacenamiento en formato Parquet (más eficiente que CSV)

---

## ✅ 9. Finalización

El pipeline ETL ha consolidado exitosamente **6 fuentes de datos** en una tabla analítica unificada de **110,197 registros**, lista para análisis avanzados, visualizaciones y modelado predictivo.

**Principales Logros:**
- ✅ Integración completa de datos dispersos
- ✅ Filtrado preciso de pedidos delivered (97% de completitud)
- ✅ Identificación de insights accionables
- ✅ Dataset limpio y estandarizado

**Valor de Negocio:**
El dataset resultante permite tomar decisiones basadas en datos sobre:
- Optimización de inventario por categoría
- Estrategias de expansión geográfica
- Mejora de satisfacción del cliente
- Personalización de métodos de pago por región

---

**Fecha de Ejecución:** 16-01-2026  
**Archivo Generado:** `tabla_principal_etl.csv`  
**Autor del Proceso ETL: Santiago Joaquin Mozo  
**Versión del Reporte:** 1.0
