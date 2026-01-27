"""
DASHBOARD PARA SISTEMA DE RECOMENDACIÓN DE PRODUCTOS OLIST
Dashboard interactivo para predecir recomendaciones de productos
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import requests
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from pathlib import Path

# Configuración de la página
st.set_page_config(
    page_title="Sistema de Recomendación Olist",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS
st.markdown("""
<style>
    .main-title {
        text-align: center;
        color: #1E3A8A;
        font-size: 2.8rem;
        margin-bottom: 2rem;
    }
    .section-header {
        color: #2563EB;
        border-bottom: 2px solid #E5E7EB;
        padding-bottom: 0.5rem;
        margin-top: 2rem;
    }
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .metric-box {
        background: #F3F4F6;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #3B82F6;
        margin: 0.5rem 0;
    }
    .stButton > button {
        width: 100%;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Título principal
st.markdown('<h1 class="main-title">🛒 Sistema de Recomendación Olist</h1>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2231/2231692.png", width=80)
    st.title("⚙️ Configuración")
    
    # Selección de modo
    mode = st.radio(
        "Modo de operación:",
        ["Modelo Local", "API FastAPI"]
    )
    
    if mode == "API FastAPI":
        api_url = st.text_input("URL de la API:", "http://localhost:8000")
    
    st.divider()
    
    # Información del modelo
    st.subheader("📊 Información del Modelo")
    
    try:
        # Intenta cargar metadatos
        metadata_path = Path("exported_model/model_metadata.json")
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            st.success(f"**Modelo:** {metadata['best_model_name']}")
            st.info(f"**Fecha entrenamiento:** {metadata['training_date']}")
            st.info(f"**F1-Score:** {metadata['performance']['f1_score']:.3f}")
            st.info(f"**Exactitud:** {metadata['performance']['accuracy']:.3f}")
        else:
            st.warning("Metadatos no encontrados")
    except Exception as e:
        st.error(f"Error: {e}")
    
    st.divider()
    
    # Información del proyecto
    with st.expander("📋 Acerca del proyecto"):
        st.markdown("""
        Este sistema predice si un producto recibirá una 
        reseña positiva (score ≥ 4) basándose en:
        
        - **Características del producto:** precio, categoría
        - **Datos del cliente:** estado, región
        - **Información de compra:** fecha, hora, método de pago
        
        **Modelo:** XGBoost entrenado con datos históricos de Olist
        **Precisión:** 77%
        **F1-Score:** 0.87
        """)

# Clase para manejar predicciones
class PredictionSystem:
    def __init__(self, use_api=False, api_url=None):
        self.use_api = use_api
        self.api_url = api_url
        self.model = None
        self.preprocessor = None
        self.metadata = None
        
        if not use_api:
            self.load_model()
    
    def load_model(self):
        """Carga el modelo entrenado"""
        try:
            # Cargar modelo
            with open('exported_model/best_model.pkl', 'rb') as f:
                self.model = pickle.load(f)
            
            # Cargar preprocesador
            with open('exported_model/preprocessor.pkl', 'rb') as f:
                self.preprocessor = pickle.load(f)
            
            # Cargar metadatos
            with open('exported_model/model_metadata.json', 'r') as f:
                self.metadata = json.load(f)
            
            return True
        except Exception as e:
            st.sidebar.error(f"Error cargando modelo: {e}")
            return False
    
    def predict_local(self, input_data):
        """Predicción usando modelo local"""
        try:
            # Convertir a DataFrame
            df = pd.DataFrame([input_data])
            
            # Asegurar columnas
            features = self.metadata['features']['numeric'] + self.metadata['features']['categorical']
            for col in features:
                if col not in df.columns:
                    if col in self.metadata['features']['numeric']:
                        df[col] = 0
                    else:
                        df[col] = 'unknown'
            
            # Preprocesar
            X = self.preprocessor.transform(df)
            
            # Predecir
            prediction = self.model.predict(X)[0]
            probability = self.model.predict_proba(X)[0][1]
            
            # Determinar confianza
            if probability > 0.8:
                confidence = "Muy Alta"
            elif probability > 0.7:
                confidence = "Alta"
            elif probability > 0.6:
                confidence = "Moderada"
            elif probability > 0.5:
                confidence = "Baja"
            else:
                confidence = "Muy Baja"
            
            return {
                'prediction': int(prediction),
                'probability': float(probability),
                'recommend': bool(prediction),
                'confidence': confidence
            }
        except Exception as e:
            st.error(f"Error en predicción: {e}")
            return None
    
    def predict_api(self, input_data):
        """Predicción usando API"""
        try:
            response = requests.post(
                f"{self.api_url}/predict",
                json=input_data,
                timeout=10
            )
            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"Error API: {response.status_code}")
                return None
        except Exception as e:
            st.error(f"Error conectando con API: {e}")
            return None

# Inicializar sistema
if mode == "API FastAPI":
    predictor = PredictionSystem(use_api=True, api_url=api_url)
else:
    predictor = PredictionSystem(use_api=False)

# Pestañas principales
tab1, tab2, tab3 = st.tabs(["📊 Predicción Individual", "📁 Predicción por Lotes", "📈 Métricas del Modelo"])

# TAB 1: Predicción individual
with tab1:
    st.markdown('<h2 class="section-header">Predicción Individual</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Información del producto
        st.subheader("📦 Información del Producto")
        
        price = st.number_input("Precio (R$)", min_value=0.0, max_value=10000.0, value=149.90, step=10.0, format="%.2f")
        
        categories = [
            "cama_mesa_banho", "beleza_saude", "esporte_lazer",
            "moveis_decoracao", "informatica_acessorios", "utilidades_domesticas",
            "relogios_presentes", "telefonia", "ferramentas_jardim", "automotivo"
        ]
        category = st.selectbox("Categoría del Producto", categories)
        
        order_num = st.number_input("Número de Orden", min_value=1, max_value=100000, value=1000)
    
    with col2:
        # Información del cliente
        st.subheader("👤 Información del Cliente")
        
        states = ["SP", "RJ", "MG", "RS", "PR", "SC", "BA", "DF", "GO", "PE"]
        state = st.selectbox("Estado", states)
        
        regions = ["Sudeste", "Sul", "Nordeste", "Centro-Oeste", "Norte"]
        region = st.selectbox("Región", regions)
        
        payments = ["credit_card", "boleto", "voucher", "debit_card", "other"]
        payment = st.selectbox("Método de Pago", payments)
        
        # Fecha y hora
        st.subheader("📅 Fecha y Hora de Compra")
        col_date1, col_date2 = st.columns(2)
        with col_date1:
            year = st.number_input("Año", 2016, 2024, 2023)
            month = st.selectbox("Mes", range(1, 13), index=5)
            day = st.selectbox("Día", range(1, 32), index=14)
        with col_date2:
            weekday = st.selectbox("Día de la Semana", 
                                 ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"],
                                 index=3)
            hour = st.slider("Hora", 0, 23, 14)
    
    # Convertir día de semana a número
    weekday_map = {"Lunes": 0, "Martes": 1, "Miércoles": 2, "Jueves": 3, 
                   "Viernes": 4, "Sábado": 5, "Domingo": 6}
    weekday_num = weekday_map[weekday]
    
    # Botón de predicción
    if st.button("🎯 Predecir Recomendación", type="primary", use_container_width=True):
        # Preparar datos
        data = {
            "order_number": int(order_num),
            "price": float(price),
            "product_category_name": category,
            "customer_state": state,
            "customer_region": region,
            "payment_type": payment,
            "purchase_year": int(year),
            "purchase_month": int(month),
            "purchase_day": int(day),
            "purchase_weekday": int(weekday_num),
            "purchase_hour": int(hour)
        }
        
        # Realizar predicción
        with st.spinner("Realizando predicción..."):
            if mode == "API FastAPI":
                result = predictor.predict_api(data)
            else:
                result = predictor.predict_local(data)
        
        if result:
            # Mostrar resultados
            st.markdown("### 📋 Resultado de la Predicción")
            
            # Tarjeta de resultado
            if result['recommend']:
                color = "#10B981"
                icon = "✅"
                title = "RECOMENDAR PRODUCTO"
                message = "Este producto tiene alta probabilidad de recibir una reseña positiva (score ≥ 4)"
            else:
                color = "#EF4444"
                icon = "❌"
                title = "NO RECOMENDAR PRODUCTO"
                message = "Este producto tiene baja probabilidad de recibir una reseña positiva"
            
            st.markdown(f"""
            <div style="background: {color}; color: white; padding: 1.5rem; border-radius: 10px; margin: 1rem 0;">
                <h3 style="margin: 0; color: white;">{icon} {title}</h3>
                <p style="margin: 0.5rem 0 0 0; opacity: 0.9;">{message}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Métricas
            col_metric1, col_metric2, col_metric3 = st.columns(3)
            
            with col_metric1:
                st.metric(
                    "Probabilidad",
                    f"{result['probability']:.1%}",
                    delta=f"Confianza: {result.get('confidence', 'N/A')}"
                )
            
            with col_metric2:
                st.metric(
                    "Recomendación",
                    "Sí" if result['recommend'] else "No",
                    delta="Recomendado" if result['recommend'] else "No recomendado"
                )
            
            with col_metric3:
                pred_value = result['prediction']
                st.metric(
                    "Valor Predicho",
                    pred_value,
                    delta="Positivo" if pred_value == 1 else "Negativo"
                )
            
            # Visualizaciones
            st.markdown("#### 📈 Visualización de Resultados")
            
            col_viz1, col_viz2 = st.columns(2)
            
            with col_viz1:
                # Gauge de probabilidad
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=result['probability'] * 100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Probabilidad de Recomendación"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': color},
                        'steps': [
                            {'range': [0, 50], 'color': "#EF4444"},
                            {'range': [50, 70], 'color': "#F59E0B"},
                            {'range': [70, 100], 'color': "#10B981"}
                        ],
                        'threshold': {
                            'line': {'color': "black", 'width': 4},
                            'thickness': 0.75,
                            'value': 70
                        }
                    }
                ))
                fig_gauge.update_layout(height=300)
                st.plotly_chart(fig_gauge, use_container_width=True)
            
            with col_viz2:
                # Gráfico de importancia de características (ejemplo)
                feature_data = {
                    'Característica': ['Precio', 'Categoría', 'Región', 'Método Pago', 'Hora', 'Estado'],
                    'Importancia': [0.25, 0.20, 0.15, 0.15, 0.10, 0.15]
                }
                df_features = pd.DataFrame(feature_data)
                
                fig_bar = px.bar(
                    df_features,
                    x='Importancia',
                    y='Característica',
                    orientation='h',
                    color='Importancia',
                    color_continuous_scale='Blues',
                    title="Importancia Relativa de Características"
                )
                fig_bar.update_layout(height=300)
                st.plotly_chart(fig_bar, use_container_width=True)
            
            # Resumen de datos
            with st.expander("📝 Ver datos ingresados"):
                st.json(data)

# TAB 2: Predicción por lotes
with tab2:
    st.markdown('<h2 class="section-header">Predicción por Lotes</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    Sube un archivo CSV con múltiples productos para obtener predicciones en lote.
    
    **Formato requerido:** El CSV debe contener las siguientes columnas:
    - `order_number` (entero)
    - `price` (decimal)
    - `product_category_name` (texto)
    - `customer_state` (texto: SP, RJ, MG, etc.)
    - `customer_region` (texto: Sudeste, Sul, etc.)
    - `payment_type` (texto: credit_card, boleto, etc.)
    - `purchase_year`, `purchase_month`, `purchase_day` (enteros)
    - `purchase_weekday` (entero: 0=Lunes, 6=Domingo)
    - `purchase_hour` (entero: 0-23)
    """)
    
    uploaded_file = st.file_uploader("Selecciona un archivo CSV", type=['csv'])
    
    if uploaded_file is not None:
        try:
            # Leer CSV
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Archivo cargado: {len(df)} registros")
            
            # Mostrar vista previa
            with st.expander("👁️ Vista previa de datos"):
                st.dataframe(df.head())
            
            # Verificar columnas requeridas
            required_columns = [
                'order_number', 'price', 'product_category_name',
                'customer_state', 'customer_region', 'payment_type',
                'purchase_year', 'purchase_month', 'purchase_day',
                'purchase_weekday', 'purchase_hour'
            ]
            
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                st.error(f"❌ Faltan columnas: {', '.join(missing_columns)}")
                st.info("Por favor, asegúrate de que tu CSV tenga todas las columnas requeridas.")
            else:
                if st.button("🚀 Ejecutar Predicciones por Lote", type="primary"):
                    # Convertir a lista de diccionarios
                    data_list = df[required_columns].to_dict('records')
                    
                    if mode == "API FastAPI":
                        # Para modo API
                        try:
                            with st.spinner("Enviando datos a la API..."):
                                response = requests.post(
                                    f"{api_url}/predict-batch",
                                    json={"data": data_list},
                                    timeout=30
                                )
                                
                                if response.status_code == 200:
                                    results = response.json()
                                    predictions = []
                                    probabilities = []
                                    
                                    for result in results['results']:
                                        predictions.append(result['recommend'])
                                        probabilities.append(result['probability'])
                                    
                                    # Agregar resultados al DataFrame
                                    df['prediction'] = predictions
                                    df['probability'] = probabilities
                                    df['recommend'] = df['prediction'].map({True: '✅ Sí', False: '❌ No'})
                                    
                                    st.success(f"✅ {len(results['results'])} predicciones completadas")
                                    
                        except Exception as e:
                            st.error(f"Error con la API: {e}")
                    else:
                        # Para modo local
                        predictions = []
                        probabilities = []
                        confidences = []
                        
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        for i, data in enumerate(data_list):
                            status_text.text(f"Procesando registro {i+1}/{len(data_list)}...")
                            result = predictor.predict_local(data)
                            
                            if result:
                                predictions.append(result['recommend'])
                                probabilities.append(result['probability'])
                                confidences.append(result.get('confidence', 'N/A'))
                            
                            progress_bar.progress((i + 1) / len(data_list))
                        
                        # Agregar resultados al DataFrame
                        df['prediction'] = predictions
                        df['probability'] = probabilities
                        df['confidence'] = confidences
                        df['recommend'] = df['prediction'].map({True: '✅ Sí', False: '❌ No'})
                        
                        st.success(f"✅ {len(df)} predicciones completadas localmente")
                    
                    # Mostrar resumen
                    if 'prediction' in df.columns:
                        col_sum1, col_sum2, col_sum3 = st.columns(3)
                        
                        with col_sum1:
                            recommend_count = df['prediction'].sum()
                            st.metric("Recomendados", f"{recommend_count}/{len(df)}")
                        
                        with col_sum2:
                            avg_prob = df['probability'].mean()
                            st.metric("Probabilidad Promedio", f"{avg_prob:.1%}")
                        
                        with col_sum3:
                            if 'confidence' in df.columns:
                                high_conf = sum(1 for c in df['confidence'] if c in ['Alta', 'Muy Alta'])
                                st.metric("Alta Confianza", high_conf)
                        
                        # Descargar resultados
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label="📥 Descargar Resultados CSV",
                            data=csv,
                            file_name=f"predicciones_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                        
                        # Mostrar tabla de resultados
                        with st.expander("📊 Ver todas las predicciones"):
                            display_cols = ['product_category_name', 'price', 'customer_state', 
                                          'probability', 'confidence', 'recommend']
                            st.dataframe(df[display_cols])
        
        except Exception as e:
            st.error(f"❌ Error procesando archivo: {e}")

# TAB 3: Métricas del modelo
with tab3:
    st.markdown('<h2 class="section-header">Métricas del Modelo</h2>', unsafe_allow_html=True)
    
    if predictor.metadata:
        col_info1, col_info2 = st.columns(2)
        
        with col_info1:
            st.markdown("#### Información del Modelo")
            info_data = {
                "Modelo": predictor.metadata['best_model_name'],
                "Tipo": predictor.metadata['best_model_type'],
                "Fecha Entrenamiento": predictor.metadata['training_date'],
                "Muestras Entrenamiento": f"{predictor.metadata['dataset_info']['train_samples']:,}",
                "Muestras Prueba": f"{predictor.metadata['dataset_info']['test_samples']:,}"
            }
            
            for key, value in info_data.items():
                st.markdown(f"**{key}:** {value}")
        
        with col_info2:
            st.markdown("#### Métricas de Rendimiento")
            metrics_data = {
                "Exactitud": f"{predictor.metadata['performance']['accuracy']:.3f}",
                "Precisión": f"{predictor.metadata['performance']['precision']:.3f}",
                "Recall": f"{predictor.metadata['performance']['recall']:.3f}",
                "F1-Score": f"{predictor.metadata['performance']['f1_score']:.3f}",
                "ROC-AUC": f"{predictor.metadata['performance']['roc_auc']:.3f}"
            }
            
            for key, value in metrics_data.items():
                st.markdown(f"**{key}:** {value}")
        
        # Gráfico de métricas
        st.markdown("#### Comparación de Métricas")
        
        metrics_df = pd.DataFrame({
            'Métrica': ['Exactitud', 'Precisión', 'Recall', 'F1-Score', 'ROC-AUC'],
            'Valor': [
                predictor.metadata['performance']['accuracy'],
                predictor.metadata['performance']['precision'],
                predictor.metadata['performance']['recall'],
                predictor.metadata['performance']['f1_score'],
                predictor.metadata['performance']['roc_auc']
            ]
        })
        
        fig_metrics = px.bar(
            metrics_df,
            x='Métrica',
            y='Valor',
            color='Valor',
            color_continuous_scale='Viridis',
            text='Valor',
            title="Métricas de Rendimiento del Modelo"
        )
        fig_metrics.update_traces(texttemplate='%{text:.3f}', textposition='outside')
        fig_metrics.update_layout(height=400)
        st.plotly_chart(fig_metrics, use_container_width=True)
        
        # Características utilizadas
        st.markdown("#### Características Utilizadas")
        
        col_feat1, col_feat2 = st.columns(2)
        
        with col_feat1:
            st.markdown("**Numéricas:**")
            for feat in predictor.metadata['features']['numeric']:
                st.markdown(f"- {feat}")
        
        with col_feat2:
            st.markdown("**Categóricas:**")
            for feat in predictor.metadata['features']['categorical']:
                st.markdown(f"- {feat}")
    
    else:
        st.warning("No se encontró información del modelo. Carga los metadatos para ver las métricas.")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #6B7280; padding: 1rem;'>
        <p>🛍️ Sistema de Recomendación de Productos Olist • Proyecto Final Data Science</p>
        <p>📧 Contacto: infodatateam@DataVivaConsulting.com • 📅 {}</p>
    </div>
    """.format(datetime.now().strftime("%Y")),
    unsafe_allow_html=True
)

# Instrucciones para ejecutar
with st.expander("📋 Instrucciones de Ejecución"):
    st.markdown("""
    ### Para usar este dashboard:
    
    1. **Modo Local (Recomendado):**
       - Asegúrate de que los archivos del modelo estén en `exported_model/`
       - Selecciona "Modelo Local" en la barra lateral
       - Usa las pestañas para hacer predicciones
    
    2. **Modo API:**
       - Primero ejecuta el servidor FastAPI (crea un archivo api_server.py)
       - En la barra lateral, selecciona "API FastAPI"
       - Ingresa la URL de la API (por defecto: http://localhost:8000)
    
    3. **Requerimientos:**
    ```bash
    pip install streamlit pandas numpy plotly requests scikit-learn xgboost
    ```
    
    4. **Ejecutar Streamlit:**
    ```bash
    streamlit run Dashboard.py
    ```
    """)

# Verificación de archivos del modelo
if not Path("exported_model").exists():
    st.error("⚠️ No se encontró el directorio 'exported_model'")
    st.info("Asegúrate de que el modelo entrenado esté en la ruta correcta.")