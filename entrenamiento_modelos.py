"""
SISTEMA DE ENTRENAMIENTO DE MODELOS DE RECOMENDACIÓN
Script para entrenar y seleccionar el mejor modelo de recomendación
"""

import pandas as pd
import numpy as np
import pickle
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Librerías para gráficos
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn-v0_8')

# Librerías de machine learning
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Modelos a evaluar
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Métricas de evaluación
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

class ModelTrainingSystem:
    """
    Sistema completo para entrenamiento y evaluación de modelos
    """
    
    def __init__(self, data_path='olist_clean_for_model.csv'):
        """
        Inicializa el sistema con la ruta del dataset
        """
        self.data_path = data_path
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.preprocessor = None
        self.models_results = {}
        self.best_model = None
        self.best_model_name = ""
        
        print(f"Inicializando sistema con datos de: {data_path}")
    
    def load_and_prepare_data(self):
        """
        Carga y prepara los datos para el entrenamiento
        """
        print("\n" + "="*60)
        print("CARGANDO Y PREPARANDO DATOS")
        print("="*60)
        
        try:
            # Cargar el dataset
            print(f"Cargando archivo: {self.data_path}")
            self.data = pd.read_csv(self.data_path)
            
            # Información básica del dataset
            print(f"✓ Dataset cargado correctamente")
            print(f"  Registros totales: {self.data.shape[0]:,}")
            print(f"  Variables: {self.data.shape[1]}")
            
            # Mostrar primeras filas para entender la estructura
            print("\nPrimeras 5 filas del dataset:")
            print(self.data.head())
            
            # Verificar tipos de datos
            print("\nTipos de datos por columna:")
            for col, dtype in self.data.dtypes.items():
                print(f"  {col:25}: {dtype}")
            
            # Manejar valores nulos
            print("\nValores nulos por columna:")
            null_counts = self.data.isnull().sum()
            for col, count in null_counts.items():
                if count > 0:
                    print(f"  {col:25}: {count:,} ({count/len(self.data)*100:.1f}%)")
            
            # Preparar variables de fecha
            print("\nExtrayendo características de fecha...")
            self._extract_date_features()
            
            # Crear variable objetivo
            print("Creando variable objetivo...")
            self._create_target_variable()
            
            # Seleccionar características
            print("Seleccionando características para el modelo...")
            self._select_features()
            
            return True
            
        except Exception as e:
            print(f"✗ Error cargando datos: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _extract_date_features(self):
        """
        Extrae características de las columnas de fecha
        """
        # Verificar si existe la columna de fecha
        date_column = None
        possible_date_columns = ['order_purchase_datetime', 'order_purchase_timestamp', 'purchase_date']
        
        for col in possible_date_columns:
            if col in self.data.columns:
                date_column = col
                break
        
        if date_column:
            print(f"  Usando columna de fecha: {date_column}")
            self.data[date_column] = pd.to_datetime(self.data[date_column], errors='coerce')
            
            # Extraer componentes de fecha
            self.data['purchase_year'] = self.data[date_column].dt.year
            self.data['purchase_month'] = self.data[date_column].dt.month
            self.data['purchase_day'] = self.data[date_column].dt.day
            self.data['purchase_weekday'] = self.data[date_column].dt.weekday
            self.data['purchase_hour'] = self.data[date_column].dt.hour
            
            # Rellenar valores nulos
            for col in ['purchase_year', 'purchase_month', 'purchase_day', 'purchase_weekday', 'purchase_hour']:
                if col in self.data.columns:
                    median_val = self.data[col].median()
                    self.data[col] = self.data[col].fillna(median_val)
            
            print("  ✓ Características de fecha extraídas")
        else:
            print("  ⚠ No se encontró columna de fecha, usando valores predeterminados")
            # Crear características de fecha predeterminadas
            self.data['purchase_year'] = 2018
            self.data['purchase_month'] = 6
            self.data['purchase_day'] = 15
            self.data['purchase_weekday'] = 2
            self.data['purchase_hour'] = 12
    
    def _create_target_variable(self):
        """
        Crea la variable objetivo basada en el puntaje de reseña
        """
        if 'review_score' in self.data.columns:
            # Consideramos positiva una reseña con puntaje >= 4
            self.data['recommend_product'] = (self.data['review_score'] >= 4).astype(int)
            
            # Estadísticas de la variable objetivo
            positives = self.data['recommend_product'].sum()
            total = len(self.data)
            print(f"  ✓ Productos recomendados: {positives:,} ({positives/total:.1%})")
            print(f"  ✓ Productos no recomendados: {total-positives:,} ({1-positives/total:.1%})")
        else:
            raise ValueError("Columna 'review_score' no encontrada en los datos")
    
    def _select_features(self):
        """
        Selecciona las características para el modelo
        """
        # Primero, listar todas las columnas disponibles
        print("\nColumnas disponibles en el dataset:")
        for col in self.data.columns:
            print(f"  - {col}")
        
        # Definir posibles características basadas en el dataset
        possible_features = {
            'numeric': ['order_number', 'price', 'review_score', 
                       'purchase_year', 'purchase_month', 'purchase_day',
                       'purchase_weekday', 'purchase_hour'],
            'categorical': ['product_category_name', 'customer_state', 
                          'customer_region', 'payment_type']
        }
        
        # Seleccionar solo las características que existen
        self.numeric_features = []
        for feat in possible_features['numeric']:
            if feat in self.data.columns and feat != 'review_score':  # Excluimos la variable objetivo
                self.numeric_features.append(feat)
        
        self.categorical_features = []
        for feat in possible_features['categorical']:
            if feat in self.data.columns:
                self.categorical_features.append(feat)
        
        print(f"\n✓ Características seleccionadas:")
        print(f"  Numéricas ({len(self.numeric_features)}): {self.numeric_features}")
        print(f"  Categóricas ({len(self.categorical_features)}): {self.categorical_features}")
        
        if len(self.numeric_features) == 0:
            print("  ⚠ No se encontraron características numéricas")
            # Crear una característica numérica básica si no hay ninguna
            self.data['feature_count'] = 1
            self.numeric_features = ['feature_count']
        
        if len(self.categorical_features) == 0:
            print("  ⚠ No se encontraron características categóricas")
            # Crear una característica categórica básica si no hay ninguna
            self.data['dummy_category'] = 'default'
            self.categorical_features = ['dummy_category']
    
    def create_preprocessing_pipeline(self):
        """
        Crea el pipeline de preprocesamiento
        """
        print("\n" + "="*60)
        print("CREANDO PIPELINE DE PREPROCESAMIENTO")
        print("="*60)
        
        try:
            # Pipeline para características numéricas
            numeric_transformer = Pipeline(steps=[
                ('scaler', StandardScaler())
            ])
            
            # Pipeline para características categóricas
            categorical_transformer = Pipeline(steps=[
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ])
            
            # Combinar transformadores
            self.preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numeric_transformer, self.numeric_features),
                    ('cat', categorical_transformer, self.categorical_features)
                ]
            )
            
            print("✓ Pipeline de preprocesamiento creado")
            print(f"  Características a procesar: {len(self.numeric_features) + len(self.categorical_features)}")
            
            # Probar el preprocesador
            test_data = self.data[self.numeric_features + self.categorical_features].head(5)
            try:
                test_processed = self.preprocessor.fit_transform(test_data)
                print(f"  ✓ Pipeline probado exitosamente")
                print(f"  Forma después del preprocesamiento: {test_processed.shape}")
            except Exception as e:
                print(f"  ⚠ Error probando pipeline: {e}")
            
            return True
            
        except Exception as e:
            print(f"✗ Error creando pipeline: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def split_data(self, test_size=0.2):
        """
        Divide los datos en conjuntos de entrenamiento y prueba
        """
        print("\n" + "="*60)
        print("DIVIDIENDO DATOS EN ENTRENAMIENTO Y PRUEBA")
        print("="*60)
        
        try:
            # Preparar características y variable objetivo
            X = self.data[self.numeric_features + self.categorical_features]
            y = self.data['recommend_product']
            
            print(f"  Características originales: {X.shape}")
            print(f"  Variable objetivo: {y.shape}")
            
            # Aplicar preprocesamiento
            print("  Aplicando preprocesamiento...")
            X_processed = self.preprocessor.fit_transform(X)
            
            # Obtener nombres de características después del preprocesamiento
            feature_names = []
            feature_names.extend(self.numeric_features)
            
            if 'cat' in self.preprocessor.named_transformers_:
                encoder = self.preprocessor.named_transformers_['cat']['onehot']
                cat_features = encoder.get_feature_names_out(self.categorical_features)
                feature_names.extend(cat_features)
            
            # Convertir a DataFrame
            X_processed_df = pd.DataFrame(X_processed, columns=feature_names)
            
            print(f"  Características después de preprocesamiento: {X_processed_df.shape}")
            
            # Dividir datos
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X_processed_df, y,
                test_size=test_size,
                random_state=42,
                stratify=y
            )
            
            print(f"\n✓ Datos divididos correctamente")
            print(f"  Conjunto de entrenamiento: {self.X_train.shape[0]:,} registros")
            print(f"  Conjunto de prueba: {self.X_test.shape[0]:,} registros")
            print(f"  Características: {self.X_train.shape[1]}")
            
            # Verificar distribución de clases
            print(f"\n  Distribución en entrenamiento:")
            train_positives = self.y_train.sum()
            print(f"    Recomendados: {train_positives:,} ({train_positives/len(self.y_train):.1%})")
            
            print(f"  Distribución en prueba:")
            test_positives = self.y_test.sum()
            print(f"    Recomendados: {test_positives:,} ({test_positives/len(self.y_test):.1%})")
            
            return True
            
        except Exception as e:
            print(f"✗ Error dividiendo datos: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def train_models(self):
        """
        Entrena y evalúa múltiples modelos
        """
        print("\n" + "="*60)
        print("ENTRENANDO MODELOS")
        print("="*60)
        
        # Definir los modelos a entrenar
        models_to_train = {
            'Logistic_Regression': {
                'model': LogisticRegression(
                    random_state=42,
                    max_iter=1000,
                    class_weight='balanced'
                ),
                'params': {'C': [0.1, 1.0]}
            },
            'Decision_Tree': {
                'model': DecisionTreeClassifier(
                    random_state=42,
                    class_weight='balanced',
                    max_depth=10
                ),
                'params': {}
            },
            'Random_Forest': {
                'model': RandomForestClassifier(
                    random_state=42,
                    n_estimators=100,
                    class_weight='balanced',
                    n_jobs=-1
                ),
                'params': {'max_depth': [10, 20]}
            },
            'XGBoost': {
                'model': XGBClassifier(
                    random_state=42,
                    eval_metric='logloss',
                    use_label_encoder=False,
                    n_jobs=-1
                ),
                'params': {
                    'n_estimators': [100],
                    'max_depth': [3, 5]
                }
            }
        }
        
        print(f"Entrenando {len(models_to_train)} modelos diferentes...")
        
        best_f1_score = 0
        
        for model_name, model_config in models_to_train.items():
            print(f"\n{'—'*40}")
            print(f"Modelo: {model_name}")
            print('—'*40)
            
            try:
                # Entrenar modelo
                start_time = datetime.now()
                model = model_config['model']
                
                # Si hay parámetros para ajustar, usar GridSearchCV simple
                if model_config['params']:
                    from sklearn.model_selection import GridSearchCV
                    grid_search = GridSearchCV(
                        model, 
                        model_config['params'], 
                        cv=3, 
                        scoring='f1',
                        n_jobs=-1,
                        verbose=0
                    )
                    grid_search.fit(self.X_train, self.y_train)
                    model = grid_search.best_estimator_
                    print(f"  Mejores parámetros: {grid_search.best_params_}")
                else:
                    model.fit(self.X_train, self.y_train)
                
                training_time = (datetime.now() - start_time).total_seconds()
                
                # Realizar predicciones
                y_pred = model.predict(self.X_test)
                y_pred_proba = model.predict_proba(self.X_test)[:, 1]
                
                # Calcular métricas
                accuracy = accuracy_score(self.y_test, y_pred)
                precision = precision_score(self.y_test, y_pred)
                recall = recall_score(self.y_test, y_pred)
                f1 = f1_score(self.y_test, y_pred)
                roc_auc = roc_auc_score(self.y_test, y_pred_proba)
                
                # Validación cruzada
                cv_scores = cross_val_score(model, self.X_train, self.y_train, 
                                          cv=3, scoring='f1')
                cv_mean = cv_scores.mean()
                
                # Guardar resultados
                self.models_results[model_name] = {
                    'model': model,
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'roc_auc': roc_auc,
                    'cv_mean': cv_mean,
                    'training_time': training_time,
                    'predictions': y_pred,
                    'probabilities': y_pred_proba
                }
                
                # Mostrar resultados
                print(f"  Exactitud: {accuracy:.4f}")
                print(f"  Precisión: {precision:.4f}")
                print(f"  Recall: {recall:.4f}")
                print(f"  F1-Score: {f1:.4f}")
                print(f"  ROC-AUC: {roc_auc:.4f}")
                print(f"  CV F1-Score (media): {cv_mean:.4f}")
                print(f"  Tiempo entrenamiento: {training_time:.1f}s")
                
                # Actualizar mejor modelo
                if f1 > best_f1_score:
                    best_f1_score = f1
                    self.best_model = model
                    self.best_model_name = model_name
                    
            except Exception as e:
                print(f"  ✗ Error entrenando {model_name}: {e}")
                continue
        
        if self.best_model is not None:
            print(f"\n{'='*60}")
            print(f"✓ MEJOR MODELO SELECCIONADO: {self.best_model_name}")
            print(f"  F1-Score: {best_f1_score:.4f}")
            print('='*60)
            return True
        else:
            print(f"\n✗ No se pudo entrenar ningún modelo")
            return False
    
    def evaluate_and_compare_models(self):
        """
        Evalúa y compara los modelos entrenados
        """
        print("\n" + "="*60)
        print("EVALUACIÓN Y COMPARACIÓN DE MODELOS")
        print("="*60)
        
        if not self.models_results:
            print("No hay modelos para comparar")
            return False
        
        # Crear tabla de comparación
        comparison_data = []
        
        for model_name, results in self.models_results.items():
            comparison_data.append({
                'Modelo': model_name,
                'Accuracy': results['accuracy'],
                'Precision': results['precision'],
                'Recall': results['recall'],
                'F1-Score': results['f1_score'],
                'ROC-AUC': results['roc_auc'],
                'CV Score': results['cv_mean'],
                'Tiempo (s)': results['training_time']
            })
        
        # Crear DataFrame
        df_comparison = pd.DataFrame(comparison_data)
        df_comparison = df_comparison.sort_values('F1-Score', ascending=False)
        
        # Mostrar tabla
        print("\nTabla de comparación de modelos:")
        print("-" * 85)
        print(df_comparison.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        
        # Crear gráficos comparativos
        self._create_comparison_charts(df_comparison)
        
        return True  # Cambiado de return df_comparison a return True
    
    def _create_comparison_charts(self, df_comparison):
        """
        Crea gráficos para comparar los modelos
        """
        try:
            # Configurar figura
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle('Comparación de Modelos', fontsize=16, fontweight='bold')
            
            # Gráfico 1: F1-Score y ROC-AUC
            ax1 = axes[0, 0]
            x_pos = np.arange(len(df_comparison))
            width = 0.35
            
            ax1.bar(x_pos - width/2, df_comparison['F1-Score'], width, 
                   label='F1-Score', alpha=0.8, color='steelblue')
            ax1.bar(x_pos + width/2, df_comparison['ROC-AUC'], width, 
                   label='ROC-AUC', alpha=0.8, color='lightcoral')
            
            ax1.set_xlabel('Modelos')
            ax1.set_ylabel('Puntaje')
            ax1.set_title('F1-Score vs ROC-AUC')
            ax1.set_xticks(x_pos)
            ax1.set_xticklabels(df_comparison['Modelo'], rotation=45, ha='right')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Gráfico 2: Precisión y Recall
            ax2 = axes[0, 1]
            ax2.bar(x_pos - width/2, df_comparison['Precision'], width, 
                   label='Precisión', alpha=0.8, color='mediumseagreen')
            ax2.bar(x_pos + width/2, df_comparison['Recall'], width, 
                   label='Recall', alpha=0.8, color='gold')
            
            ax2.set_xlabel('Modelos')
            ax2.set_ylabel('Puntaje')
            ax2.set_title('Precisión vs Recall')
            ax2.set_xticks(x_pos)
            ax2.set_xticklabels(df_comparison['Modelo'], rotation=45, ha='right')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Gráfico 3: Tiempo de entrenamiento
            ax3 = axes[1, 0]
            colors = ['green' if name == self.best_model_name else 'skyblue' 
                     for name in df_comparison['Modelo']]
            
            ax3.bar(df_comparison['Modelo'], df_comparison['Tiempo (s)'], 
                   color=colors, alpha=0.8)
            ax3.set_xlabel('Modelos')
            ax3.set_ylabel('Segundos')
            ax3.set_title('Tiempo de Entrenamiento')
            ax3.tick_params(axis='x', rotation=45)
            ax3.grid(True, alpha=0.3)
            
            # Gráfico 4: Exactitud
            ax4 = axes[1, 1]
            ax4.bar(df_comparison['Modelo'], df_comparison['Accuracy'], 
                   color=colors, alpha=0.8)
            ax4.set_xlabel('Modelos')
            ax4.set_ylabel('Exactitud')
            ax4.set_title('Exactitud por Modelo')
            ax4.tick_params(axis='x', rotation=45)
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
            print(f"\n✓ Gráfico de comparación guardado: model_comparison.png")
            plt.close()
            
        except Exception as e:
            print(f"✗ Error creando gráficos: {e}")
    
    def export_best_model(self):
        """
        Exporta el mejor modelo y los componentes necesarios
        """
        print("\n" + "="*60)
        print("EXPORTANDO MEJOR MODELO")
        print("="*60)
        
        if self.best_model is None:
            print("✗ No hay modelo para exportar")
            return False
        
        try:
            # Crear directorio para exportar
            export_dir = 'exported_model'
            os.makedirs(export_dir, exist_ok=True)
            
            # Guardar el modelo
            model_path = os.path.join(export_dir, 'best_model.pkl')
            with open(model_path, 'wb') as f:
                pickle.dump(self.best_model, f)
            print(f"✓ Modelo guardado: {model_path}")
            
            # Guardar el preprocesador
            preprocessor_path = os.path.join(export_dir, 'preprocessor.pkl')
            with open(preprocessor_path, 'wb') as f:
                pickle.dump(self.preprocessor, f)
            print(f"✓ Preprocesador guardado: {preprocessor_path}")
            
            # Guardar metadatos
            metadata = {
                'best_model_name': self.best_model_name,
                'best_model_type': type(self.best_model).__name__,
                'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'features': {
                    'numeric': self.numeric_features,
                    'categorical': self.categorical_features
                },
                'performance': {
                    'accuracy': float(self.models_results[self.best_model_name]['accuracy']),
                    'precision': float(self.models_results[self.best_model_name]['precision']),
                    'recall': float(self.models_results[self.best_model_name]['recall']),
                    'f1_score': float(self.models_results[self.best_model_name]['f1_score']),
                    'roc_auc': float(self.models_results[self.best_model_name]['roc_auc'])
                },
                'dataset_info': {
                    'total_samples': len(self.data),
                    'train_samples': len(self.X_train),
                    'test_samples': len(self.X_test),
                    'positive_class_ratio': float(self.data['recommend_product'].mean())
                }
            }
            
            metadata_path = os.path.join(export_dir, 'model_metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=4, default=str)
            print(f"✓ Metadatos guardados: {metadata_path}")
            
            # Crear archivo de ejemplo de uso
            self._create_example_usage_file(export_dir)
            
            print(f"\n✓ Modelo exportado exitosamente en: {export_dir}/")
            print(f"  Archivos creados:")
            print(f"    • best_model.pkl")
            print(f"    • preprocessor.pkl")
            print(f"    • model_metadata.json")
            print(f"    • example_usage.py")
            
            return True
            
        except Exception as e:
            print(f"✗ Error exportando modelo: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _create_example_usage_file(self, export_dir):
        """
        Crea un archivo de ejemplo para usar el modelo exportado
        """
        example_code = '''"""
EJEMPLO DE USO DEL MODELO EXPORTADO
"""

import pickle
import pandas as pd
import numpy as np
import json

def load_model_components(model_dir='exported_model'):
    """
    Carga el modelo y componentes necesarios
    """
    # Cargar modelo
    with open(f'{model_dir}/best_model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    # Cargar preprocesador
    with open(f'{model_dir}/preprocessor.pkl', 'rb') as f:
        preprocessor = pickle.load(f)
    
    # Cargar metadatos
    with open(f'{model_dir}/model_metadata.json', 'r') as f:
        metadata = json.load(f)
    
    return model, preprocessor, metadata

def make_prediction(model, preprocessor, input_data, metadata):
    """
    Realiza una predicción con nuevos datos
    """
    # Convertir a DataFrame si es necesario
    if isinstance(input_data, dict):
        input_data = pd.DataFrame([input_data])
    
    # Asegurar que tenemos todas las columnas necesarias
    for col in metadata['features']['numeric'] + metadata['features']['categorical']:
        if col not in input_data.columns:
            input_data[col] = 0 if col in metadata['features']['numeric'] else 'unknown'
    
    # Ordenar columnas como en el entrenamiento
    input_data = input_data[metadata['features']['numeric'] + metadata['features']['categorical']]
    
    # Aplicar preprocesamiento
    processed_data = preprocessor.transform(input_data)
    
    # Realizar predicción
    prediction = model.predict(processed_data)
    probability = model.predict_proba(processed_data)[:, 1]
    
    return {
        'prediction': int(prediction[0]),
        'probability': float(probability[0]),
        'recommendation': 'RECOMENDAR' if prediction[0] == 1 else 'NO RECOMENDAR'
    }

# Ejemplo de uso
if __name__ == '__main__':
    import json
    
    # Cargar componentes
    model, preprocessor, metadata = load_model_components()
    
    print(f"Modelo cargado: {metadata['best_model_name']}")
    print(f"F1-Score del modelo: {metadata['performance']['f1_score']:.4f}")
    
    # Datos de ejemplo (ajustar según las características reales)
    example_input = {
        'order_number': 1000,
        'price': 149.90,
        'product_category_name': 'cama_mesa_banho',
        'customer_state': 'SP',
        'customer_region': 'Sudeste',
        'payment_type': 'credit_card'
    }
    
    # Agregar características de fecha si están en los metadatos
    if 'purchase_year' in metadata['features']['numeric']:
        example_input.update({
            'purchase_year': 2023,
            'purchase_month': 6,
            'purchase_day': 15,
            'purchase_weekday': 3,
            'purchase_hour': 14
        })
    
    # Realizar predicción
    result = make_prediction(model, preprocessor, example_input, metadata)
    
    print("\\nResultado de la predicción:")
    print(f"• Decisión: {result['recommendation']}")
    print(f"• Probabilidad: {result['probability']:.2%}")
    print(f"• Valor predicho: {result['prediction']}")
'''
        
        example_path = os.path.join(export_dir, 'example_usage.py')
        with open(example_path, 'w', encoding='utf-8') as f:
            f.write(example_code)
        
        print(f"✓ Archivo de ejemplo creado: {example_path}")
    
    def run_full_pipeline(self):
        """
        Ejecuta el pipeline completo de entrenamiento
        """
        print("="*60)
        print("INICIANDO PIPELINE DE ENTRENAMIENTO")
        print("="*60)
        
        steps = [
            ("Cargar datos", self.load_and_prepare_data),
            ("Crear pipeline", self.create_preprocessing_pipeline),
            ("Dividir datos", self.split_data),
            ("Entrenar modelos", self.train_models),
            ("Evaluar modelos", self.evaluate_and_compare_models),
            ("Exportar modelo", self.export_best_model)
        ]
        
        for step_name, step_function in steps:
            print(f"\n▶ {step_name}")
            print("-" * 40)
            
            success = step_function()
            if not success and step_name not in ["Evaluar modelos", "Exportar modelo"]:
                print(f"✗ Error en el paso: {step_name}")
                return False
        
        print("\n" + "="*60)
        print("PIPELINE COMPLETADO")
        print("="*60)
        
        if self.best_model is not None:
            print(f"\nRESUMEN FINAL:")
            print("-" * 40)
            print(f"• Mejor modelo: {self.best_model_name}")
            best_results = self.models_results[self.best_model_name]
            print(f"• F1-Score: {best_results['f1_score']:.4f}")
            print(f"• Exactitud: {best_results['accuracy']:.4f}")
            print(f"• ROC-AUC: {best_results['roc_auc']:.4f}")
            print(f"• Modelo exportado en: exported_model/")
            print(f"\nListo para implementar en dashboard.py")
        
        return True


def main():
    """
    Función principal para ejecutar el sistema
    """
    print("\n" + "="*60)
    print("SISTEMA DE ENTRENAMIENTO DE MODELOS")
    print("="*60)
    
    # Crear instancia del sistema
    training_system = ModelTrainingSystem('olist_clean_for_model.csv')
    
    # Ejecutar pipeline completo
    success = training_system.run_full_pipeline()
    
    if not success:
        print("\n⚠ El pipeline no se completó exitosamente")
        print("Revisa los errores anteriores y corrige los datos o configuración.")


if __name__ == "__main__":
    main()