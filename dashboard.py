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

# Inicializar estado del umbral
if 'threshold' not in st.session_state:
    st.session_state.threshold = 0.5

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
    .threshold-indicator {
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.5rem 0;
        text-align: center;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Título principal
st.markdown('<h1 class="main-title">🛒 Sistema de Recomendación Olist</h1>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("Olist_icon.jpeg", width=80)
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
            
            # Mostrar umbral óptimo si existe
            if 'best_threshold' in metadata:
                st.info(f"**Umbral óptimo:** {metadata['best_threshold']:.3f}")
        else:
            st.warning("Metadatos no encontrados")
    except Exception as e:
        st.error(f"Error: {e}")
    
    st.divider()
    
    # CONTROL DE UMBRAL
    st.subheader("🎚️ Control de Umbral")
    
    # Cargar el mejor umbral del modelo si existe
    try:
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            best_threshold = metadata.get('best_threshold', 0.5)
        else:
            best_threshold = 0.5
    except:
        best_threshold = 0.5
    
    # Slider de umbral
    threshold = st.slider(
        "Umbral de decisión",
        min_value=0.1,
        max_value=0.9,
        value=st.session_state.threshold,
        step=0.05,
        key='threshold_slider',
        help="""Ajusta el umbral para las predicciones:
        • < 0.5: Más sensible (recomienda más productos)
        • > 0.5: Más estricto (recomienda menos productos)
        • 0.5: Valor por defecto"""
    )
    
    # Actualizar el estado
    st.session_state.threshold = threshold
    
    # Mostrar indicador visual del umbral
    if threshold < 0.4:
        st.markdown(
            f'<div class="threshold-indicator" style="background-color: #FCA5A5; color: #7F1D1D;">'
            f'⚠️ Umbral BAJO ({threshold:.2f}) - Muy sensible</div>',
            unsafe_allow_html=True
        )
    elif threshold < 0.6:
        st.markdown(
            f'<div class="threshold-indicator" style="background-color: #FDE68A; color: #92400E;">'
            f'⚖️ Umbral MODERADO ({threshold:.2f}) - Balanceado</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'<div class="threshold-indicator" style="background-color: #A7F3D0; color: #065F46;">'
            f'✅ Umbral ALTO ({threshold:.2f}) - Muy estricto</div>',
            unsafe_allow_html=True
        )
    
    st.divider()
    
    # Información del proyecto
    with st.expander("📋 Acerca del proyecto"):
        st.markdown("""
        Este sistema predice si un producto recibirá una 
        reseña positiva (score ≥ 4) basándose en:
        
        - **Características del producto:** precio, categoría
        - **Datos del cliente:** estado, región
        - **Información de compra:** fecha, hora, método de pago
        
        **Umbral ajustable:** Controla la sensibilidad de las recomendaciones
        
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
    
    def predict_local(self, input_data, custom_threshold=0.5):
        """Predicción usando modelo local con umbral personalizable"""
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
            
            # Obtener probabilidad
            probability = self.model.predict_proba(X)[0][1]
            
            # Aplicar umbral personalizado
            prediction = 1 if probability >= custom_threshold else 0
            
            # Determinar confianza
            if probability >= 0.8:
                confidence = "Muy Alta"
            elif probability >= 0.7:
                confidence = "Alta"
            elif probability >= 0.6:
                confidence = "Moderada"
            elif probability >= 0.5:
                confidence = "Baja"
            else:
                confidence = "Muy Baja"
            
            return {
                'prediction': int(prediction),
                'probability': float(probability),
                'recommend': bool(prediction),
                'confidence': confidence,
                'threshold_used': custom_threshold
            }
        except Exception as e:
            st.error(f"Error en predicción: {e}")
            return None
    
    def predict_local_batch(self, data_list, custom_threshold=0.5):
        """Predicción por lotes con umbral personalizable"""
        predictions = []
        probabilities = []
        confidences = []
        
        for data in data_list:
            result = self.predict_local(data, custom_threshold)
            if result:
                predictions.append(result['recommend'])
                probabilities.append(result['probability'])
                confidences.append(result.get('confidence', 'N/A'))
        
        return predictions, probabilities, confidences
    
    def predict_api(self, input_data):
        """Predicción usando API"""
        try:
            # Incluir umbral en la solicitud
            input_data['threshold'] = st.session_state.threshold
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
    
    # Mostrar umbral actual
    st.info(f"**Umbral actual:** {st.session_state.threshold:.2f} - "
            f"{'Muy sensible' if st.session_state.threshold < 0.4 else 'Moderado' if st.session_state.threshold < 0.6 else 'Estricto'}")
    
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
        with st.spinner(f"Realizando predicción (umbral={st.session_state.threshold:.2f})..."):
            if mode == "API FastAPI":
                result = predictor.predict_api(data)
            else:
                result = predictor.predict_local(data, custom_threshold=st.session_state.threshold)
        
        if result:
            # Mostrar resultados
            st.markdown("### 📋 Resultado de la Predicción")
            
            # Tarjeta de resultado
            if result['recommend']:
                color = "#10B981"
                icon = "✅"
                title = "RECOMENDAR PRODUCTO"
                message = "Este producto supera el umbral de probabilidad"
            else:
                color = "#EF4444"
                icon = "❌"
                title = "NO RECOMENDAR PRODUCTO"
                message = "Este producto no supera el umbral de probabilidad"
            
            st.markdown(f"""
            <div style="background: {color}; color: white; padding: 1.5rem; border-radius: 10px; margin: 1rem 0;">
                <h3 style="margin: 0; color: white;">{icon} {title}</h3>
                <p style="margin: 0.5rem 0 0 0; opacity: 0.9;">{message}</p>
                <p style="margin: 0.5rem 0 0 0; font-size: 0.9rem; opacity: 0.8;">
                    Umbral usado: {result.get('threshold_used', st.session_state.threshold):.2f}
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Métricas
            col_metric1, col_metric2, col_metric3, col_metric4 = st.columns(4)
            
            with col_metric1:
                st.metric(
                    "Probabilidad",
                    f"{result['probability']:.1%}",
                    delta=f"vs umbral {result.get('threshold_used', 0.5):.2f}"
                )
            
            with col_metric2:
                diff = result['probability'] - result.get('threshold_used', st.session_state.threshold)
                st.metric(
                    "Diferencia",
                    f"{diff:+.2%}",
                    delta="Supera umbral" if diff >= 0 else "No supera"
                )
            
            with col_metric3:
                st.metric(
                    "Recomendación",
                    "Sí" if result['recommend'] else "No",
                    delta="Recomendado" if result['recommend'] else "No recomendado"
                )
            
            with col_metric4:
                st.metric(
                    "Confianza",
                    result['confidence'],
                    delta="Alta" if result['confidence'] in ['Alta', 'Muy Alta'] else "Baja"
                )
            
            # Visualizaciones
            st.markdown("#### 📈 Visualización de Resultados")
            
            col_viz1, col_viz2 = st.columns(2)
            
            with col_viz1:
                # Gauge de probabilidad con umbral
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
                            'value': result.get('threshold_used', 0.5) * 100
                        }
                    }
                ))
                fig_gauge.update_layout(height=300)
                st.plotly_chart(fig_gauge, use_container_width=True)
            
            with col_viz2:
                # Gráfico de comparación con umbral
                threshold_value = result.get('threshold_used', st.session_state.threshold)
                comparison_data = {
                    'Métrica': ['Probabilidad', 'Umbral'],
                    'Valor': [result['probability'], threshold_value]
                }
                df_comparison = pd.DataFrame(comparison_data)
                
                fig_bar = px.bar(
                    df_comparison,
                    x='Métrica',
                    y='Valor',
                    color='Métrica',
                    color_discrete_map={
                        'Probabilidad': color,
                        'Umbral': '#6B7280'
                    },
                    text='Valor',
                    title="Comparación con Umbral"
                )
                fig_bar.update_traces(texttemplate='%{text:.2%}', textposition='outside')
                fig_bar.update_layout(height=300)
                fig_bar.add_hline(
                    y=threshold_value,
                    line_dash="dash",
                    line_color="red",
                    annotation_text=f"Umbral: {threshold_value:.2%}",
                    annotation_position="top right"
                )
                st.plotly_chart(fig_bar, use_container_width=True)
            
            # Resumen de datos
            with st.expander("📝 Ver datos ingresados y resultado completo"):
                st.json(result)

# TAB 2: Predicción por lotes
with tab2:
    st.markdown('<h2 class="section-header">Predicción por Lotes</h2>', unsafe_allow_html=True)
    
    st.markdown(f"""
    Sube un archivo CSV con múltiples productos para obtener predicciones en lote.
    
    **Umbral actual:** {st.session_state.threshold:.2f}
    
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
                            with st.spinner(f"Enviando datos a la API (umbral={st.session_state.threshold:.2f})..."):
                                response = requests.post(
                                    f"{api_url}/predict-batch",
                                    json={"data": data_list, "threshold": st.session_state.threshold},
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
                        predictions, probabilities, confidences = predictor.predict_local_batch(
                            data_list, custom_threshold=st.session_state.threshold
                        )
                        
                        # Agregar resultados al DataFrame
                        df['prediction'] = predictions
                        df['probability'] = probabilities
                        df['confidence'] = confidences
                        df['recommend'] = df['prediction'].map({True: '✅ Sí', False: '❌ No'})
                        
                        st.success(f"✅ {len(df)} predicciones completadas (umbral={st.session_state.threshold:.2f})")
                    
                    # Mostrar resumen
                    if 'prediction' in df.columns:
                        col_sum1, col_sum2, col_sum3, col_sum4 = st.columns(4)
                        
                        with col_sum1:
                            recommend_count = df['prediction'].sum()
                            st.metric(
                                "Recomendados", 
                                f"{recommend_count}/{len(df)}",
                                f"{recommend_count/len(df):.1%} del total"
                            )
                        
                        with col_sum2:
                            avg_prob = df['probability'].mean()
                            st.metric(
                                "Probabilidad Promedio", 
                                f"{avg_prob:.1%}",
                                f"vs umbral {st.session_state.threshold:.2f}"
                            )
                        
                        with col_sum3:
                            if 'confidence' in df.columns:
                                high_conf = sum(1 for c in df['confidence'] if c in ['Alta', 'Muy Alta'])
                                st.metric("Alta Confianza", high_conf)
                        
                        with col_sum4:
                            threshold_success = sum(1 for p in df['probability'] if p >= st.session_state.threshold)
                            st.metric(
                                "Superan Umbral",
                                f"{threshold_success}/{len(df)}",
                                f"Umbral: {st.session_state.threshold:.2f}"
                            )
                        
                        # Descargar resultados
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label="📥 Descargar Resultados CSV",
                            data=csv,
                            file_name=f"predicciones_umbral{st.session_state.threshold:.2f}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            help="Descarga los resultados con todas las predicciones"
                        )
                        
                        # Mostrar tabla de resultados
                        with st.expander("📊 Ver todas las predicciones"):
                            display_cols = ['product_category_name', 'price', 'customer_state', 
                                          'probability', 'confidence', 'recommend']
                            st.dataframe(df[display_cols].head(20))
                        
                        # Gráfico de distribución de probabilidades
                        st.markdown("#### 📊 Distribución de Probabilidades")
                        
                        fig_dist = px.histogram(
                            df, 
                            x='probability',
                            nbins=20,
                            title=f'Distribución de Probabilidades (Umbral: {st.session_state.threshold:.2f})',
                            labels={'probability': 'Probabilidad', 'count': 'Cantidad'},
                            color_discrete_sequence=['#3B82F6']
                        )
                        
                        # Añadir línea del umbral
                        fig_dist.add_vline(
                            x=st.session_state.threshold,
                            line_dash="dash",
                            line_color="red",
                            annotation_text=f"Umbral: {st.session_state.threshold:.2f}",
                            annotation_position="top right"
                        )
                        
                        fig_dist.update_layout(
                            xaxis_title="Probabilidad",
                            yaxis_title="Cantidad de Productos",
                            showlegend=False
                        )
                        
                        st.plotly_chart(fig_dist, use_container_width=True)
        
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
            
            # Mostrar umbral óptimo si existe
            if 'best_threshold' in predictor.metadata:
                st.markdown(f"**Umbral óptimo:** {predictor.metadata['best_threshold']:.3f}")
                st.caption(f"Umbral actual en uso: {st.session_state.threshold:.3f}")
        
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
        
        # Información sobre el umbral
        st.markdown("#### 🎚️ Información sobre el Umbral")
        
        col_umb1, col_umb2 = st.columns(2)
        
        with col_umb1:
            st.markdown("**Efecto del umbral:**")
            st.markdown("""
            - **Umbral bajo (< 0.5):** Más productos recomendados
            - **Umbral medio (0.5):** Balance recomendado
            - **Umbral alto (> 0.5):** Solo productos de alta calidad
            """)
        
        with col_umb2:
            if 'best_threshold' in predictor.metadata:
                optimal_threshold = predictor.metadata['best_threshold']
                st.markdown(f"**Umbral óptimo del modelo:** `{optimal_threshold:.3f}`")
                st.progress(optimal_threshold, text=f"Posición del umbral óptimo")
                
                # Comparación con umbral actual
                current = st.session_state.threshold
                difference = current - optimal_threshold
                
                if abs(difference) < 0.05:
                    st.success(f"✅ Umbral actual cercano al óptimo (diferencia: {difference:.3f})")
                elif difference > 0:
                    st.warning(f"⚠️ Umbral actual más estricto que el óptimo (diferencia: +{difference:.3f})")
                else:
                    st.warning(f"⚠️ Umbral actual más sensible que el óptimo (diferencia: {difference:.3f})")
    
    else:
        st.warning("No se encontró información del modelo. Carga los metadatos para ver las métricas.")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #6B7280; padding: 1rem;'>
        <p>🛍️ Sistema de Recomendación de Productos Olist • Proyecto Final Data Science</p>
        <p>🎚️ Umbral ajustable: {:.2f} • 📅 {}</p>
    </div>
    """.format(st.session_state.threshold, datetime.now().strftime("%Y")),
    unsafe_allow_html=True
)

# Instrucciones para ejecutar
with st.expander("📋 Instrucciones de Ejecución"):
    st.markdown("""
    ### Para usar este dashboard:
    
    1. **Modo Local (Recomendado):**
       - Asegúrate de que los archivos del modelo estén en `exported_model/`
       - Selecciona "Modelo Local" en la barra lateral
       - Ajusta el umbral según necesites
       - Usa las pestañas para hacer predicciones
    
    2. **Modo API:**
       - Primero ejecuta el servidor FastAPI
       - En la barra lateral, selecciona "API FastAPI"
       - Ingresa la URL de la API (por defecto: http://localhost:8000)
    
    3. **Control de Umbral:**
       - Usa el slider en la barra lateral para ajustar la sensibilidad
       - Umbral bajo (< 0.5): Más recomendaciones
       - Umbral alto (> 0.5): Menos pero más seguras
       - El modelo tiene un umbral óptimo que se muestra automáticamente
    
    4. **Requerimientos:**
    ```bash
    pip install streamlit pandas numpy plotly requests scikit-learn xgboost
    ```
    
    5. **Ejecutar Streamlit:**
    ```bash
    streamlit run Dashboard.py
    ```
    """)

# Verificación de archivos del modelo
if not Path("exported_model").exists():
    st.error("⚠️ No se encontró el directorio 'exported_model'")
    st.info("Asegúrate de que el modelo entrenado esté en la ruta correcta.")