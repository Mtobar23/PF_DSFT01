"""
API FASTAPI PARA SISTEMA DE RECOMENDACIÓN
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import pickle
import json
from typing import List, Dict, Any, Optional
from datetime import datetime
import uvicorn
from pathlib import Path

# Configurar FastAPI
app = FastAPI(
    title="Sistema de Recomendación Olist API",
    description="API para predecir recomendaciones de productos",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Cargar modelo y preprocesador
MODEL_PATH = "exported_model/best_model.pkl"
PREPROCESSOR_PATH = "exported_model/preprocessor.pkl"
METADATA_PATH = "exported_model/model_metadata.json"

try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    
    with open(PREPROCESSOR_PATH, 'rb') as f:
        preprocessor = pickle.load(f)
    
    with open(METADATA_PATH, 'r') as f:
        metadata = json.load(f)
    
    print("✅ Modelo cargado correctamente")
    
except Exception as e:
    print(f"❌ Error cargando modelo: {e}")
    model = None
    preprocessor = None
    metadata = None

# Definir esquemas de datos con ejemplos
class PredictionRequest(BaseModel):
    order_number: int = Field(
        default=1000,
        description="Número de orden único",
        example=1000
    )
    price: float = Field(
        default=149.90,
        description="Precio del producto en R$",
        example=149.90
    )
    product_category_name: str = Field(
        default="cama_mesa_banho",
        description="Categoría del producto",
        example="cama_mesa_banho"
    )
    customer_state: str = Field(
        default="SP",
        description="Estado del cliente (sigla de 2 letras)",
        example="SP"
    )
    customer_region: str = Field(
        default="Sudeste",
        description="Región del cliente",
        example="Sudeste"
    )
    payment_type: str = Field(
        default="credit_card",
        description="Método de pago",
        example="credit_card"
    )
    purchase_year: int = Field(
        default=2023,
        description="Año de la compra",
        example=2023
    )
    purchase_month: int = Field(
        default=6,
        description="Mes de la compra (1-12)",
        example=6
    )
    purchase_day: int = Field(
        default=15,
        description="Día del mes de la compra (1-31)",
        example=15
    )
    purchase_weekday: int = Field(
        default=3,
        description="Día de la semana (0=Lunes, 6=Domingo)",
        example=3
    )
    purchase_hour: int = Field(
        default=14,
        description="Hora de la compra (0-23)",
        example=14
    )
    
    class Config:
        schema_extra = {
            "example": {
                "order_number": 1000,
                "price": 149.90,
                "product_category_name": "cama_mesa_banho",
                "customer_state": "SP",
                "customer_region": "Sudeste",
                "payment_type": "credit_card",
                "purchase_year": 2023,
                "purchase_month": 6,
                "purchase_day": 15,
                "purchase_weekday": 3,
                "purchase_hour": 14
            }
        }

class BatchPredictionRequest(BaseModel):
    data: List[Dict[str, Any]] = Field(
        description="Lista de diccionarios con datos de productos",
        example=[
            {
                "order_number": 1001,
                "price": 199.90,
                "product_category_name": "beleza_saude",
                "customer_state": "RJ",
                "customer_region": "Sudeste",
                "payment_type": "boleto",
                "purchase_year": 2023,
                "purchase_month": 6,
                "purchase_day": 16,
                "purchase_weekday": 4,
                "purchase_hour": 10
            },
            {
                "order_number": 1002,
                "price": 299.90,
                "product_category_name": "esporte_lazer",
                "customer_state": "MG",
                "customer_region": "Sudeste",
                "payment_type": "credit_card",
                "purchase_year": 2023,
                "purchase_month": 6,
                "purchase_day": 17,
                "purchase_weekday": 5,
                "purchase_hour": 16
            }
        ]
    )

class PredictionResponse(BaseModel):
    prediction: int = Field(description="Predicción (0=No recomendar, 1=Recomendar)")
    probability: float = Field(description="Probabilidad de recomendación (0-1)")
    recommend: bool = Field(description="Si se recomienda o no el producto")
    confidence: str = Field(description="Nivel de confianza de la predicción")
    features_used: List[str] = Field(description="Lista de características utilizadas")
    
    class Config:
        schema_extra = {
            "example": {
                "prediction": 1,
                "probability": 0.87,
                "recommend": True,
                "confidence": "Alta",
                "features_used": ["price", "product_category_name", "customer_state", "customer_region", "payment_type", "purchase_year", "purchase_month", "purchase_day", "purchase_weekday", "purchase_hour"]
            }
        }

# Lista de opciones válidas para los campos
VALID_CATEGORIES = [
    "cama_mesa_banho", "beleza_saude", "esporte_lazer",
    "moveis_decoracao", "informatica_acessorios", "utilidades_domesticas",
    "relogios_presentes", "telefonia", "ferramentas_jardim", "automotivo"
]

VALID_STATES = ["SP", "RJ", "MG", "RS", "PR", "SC", "BA", "DF", "GO", "PE"]

VALID_REGIONS = ["Sudeste", "Sul", "Nordeste", "Centro-Oeste", "Norte"]

VALID_PAYMENTS = ["credit_card", "boleto", "voucher", "debit_card", "other"]

# Endpoint de salud
@app.get("/")
async def root():
    return {
        "message": "API de Recomendación Olist",
        "status": "active",
        "model_loaded": model is not None,
        "timestamp": datetime.now().isoformat(),
        "endpoints": [
            {"method": "GET", "path": "/", "description": "Página principal"},
            {"method": "GET", "path": "/health", "description": "Health check"},
            {"method": "GET", "path": "/model-info", "description": "Información del modelo"},
            {"method": "GET", "path": "/valid-values", "description": "Valores válidos para campos"},
            {"method": "POST", "path": "/predict", "description": "Predicción individual"},
            {"method": "POST", "path": "/predict-batch", "description": "Predicción por lotes"},
            {"method": "GET", "path": "/docs", "description": "Documentación Swagger UI"}
        ]
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy" if model is not None else "model_not_loaded",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": model is not None
    }

@app.get("/model-info")
async def get_model_info():
    if metadata is None:
        raise HTTPException(status_code=500, detail="Modelo no cargado")
    
    return {
        "model_name": metadata.get('best_model_name'),
        "training_date": metadata.get('training_date'),
        "performance": metadata.get('performance'),
        "features": metadata.get('features'),
        "dataset_info": metadata.get('dataset_info')
    }

@app.get("/valid-values")
async def get_valid_values():
    """Obtener valores válidos para los campos de entrada"""
    return {
        "product_category_name": VALID_CATEGORIES,
        "customer_state": VALID_STATES,
        "customer_region": VALID_REGIONS,
        "payment_type": VALID_PAYMENTS,
        "purchase_weekday": {
            "0": "Lunes",
            "1": "Martes", 
            "2": "Miércoles",
            "3": "Jueves",
            "4": "Viernes",
            "5": "Sábado",
            "6": "Domingo"
        },
        "ranges": {
            "price": {"min": 0, "max": 10000},
            "purchase_year": {"min": 2016, "max": 2024},
            "purchase_month": {"min": 1, "max": 12},
            "purchase_day": {"min": 1, "max": 31},
            "purchase_hour": {"min": 0, "max": 23}
        }
    }

# Endpoint de predicción individual con validación
@app.post("/predict", 
          response_model=PredictionResponse,
          summary="Predecir recomendación individual",
          description="""
Realiza una predicción individual para determinar si un producto será bien evaluado.

**Parámetros:**
- Todos los campos son requeridos
- Use valores válidos de /valid-values endpoint

**Respuesta:**
- prediction: 0 (no recomendar) o 1 (recomendar)
- probability: Probabilidad entre 0 y 1
- recommend: Booleano basado en prediction
- confidence: Nivel de confianza (Muy Baja, Baja, Moderada, Alta, Muy Alta)
- features_used: Características utilizadas
          """)
async def predict(request: PredictionRequest):
    try:
        # Validar valores de entrada
        if request.product_category_name not in VALID_CATEGORIES:
            raise HTTPException(
                status_code=400, 
                detail=f"Categoría inválida. Válidas: {VALID_CATEGORIES}"
            )
        
        if request.customer_state not in VALID_STATES:
            raise HTTPException(
                status_code=400,
                detail=f"Estado inválido. Válidos: {VALID_STATES}"
            )
        
        if request.customer_region not in VALID_REGIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Región inválida. Válidas: {VALID_REGIONS}"
            )
        
        if request.payment_type not in VALID_PAYMENTS:
            raise HTTPException(
                status_code=400,
                detail=f"Método de pago inválido. Válidos: {VALID_PAYMENTS}"
            )
        
        # Validar rangos
        if not (0 <= request.price <= 10000):
            raise HTTPException(
                status_code=400,
                detail="Precio debe estar entre 0 y 10000"
            )
        
        if not (0 <= request.purchase_weekday <= 6):
            raise HTTPException(
                status_code=400,
                detail="Día de semana debe estar entre 0 (Lunes) y 6 (Domingo)"
            )
        
        if not (0 <= request.purchase_hour <= 23):
            raise HTTPException(
                status_code=400,
                detail="Hora debe estar entre 0 y 23"
            )
        
        # Convertir a DataFrame
        input_data = request.dict()
        df = pd.DataFrame([input_data])
        
        # Asegurar que todas las características estén presentes
        if metadata and 'features' in metadata:
            features = metadata['features']['numeric'] + metadata['features']['categorical']
            for col in features:
                if col not in df.columns:
                    if col in metadata['features']['numeric']:
                        df[col] = 0
                    else:
                        df[col] = 'unknown'
        
        # Preprocesar
        X = preprocessor.transform(df)
        
        # Predecir
        prediction = model.predict(X)[0]
        probability = model.predict_proba(X)[0][1]
        
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
            "prediction": int(prediction),
            "probability": float(probability),
            "recommend": bool(prediction),
            "confidence": confidence,
            "features_used": features if metadata else []
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en predicción: {str(e)}")

# Endpoint de predicción por lotes
@app.post("/predict-batch",
          summary="Predicción por lotes",
          description="""
Realiza predicciones para múltiples productos a la vez.

**Parámetros:**
- data: Lista de diccionarios con los mismos campos que /predict

**Respuesta:**
- count: Número de predicciones realizadas
- results: Lista de resultados individuales
- timestamp: Fecha y hora de la predicción
          """)
async def predict_batch(request: BatchPredictionRequest):
    try:
        results = []
        errors = []
        
        for idx, item in enumerate(request.data):
            try:
                # Validar cada item
                if "product_category_name" in item and item["product_category_name"] not in VALID_CATEGORIES:
                    errors.append(f"Item {idx}: Categoría inválida")
                    continue
                
                if "customer_state" in item and item["customer_state"] not in VALID_STATES:
                    errors.append(f"Item {idx}: Estado inválido")
                    continue
                
                if "customer_region" in item and item["customer_region"] not in VALID_REGIONS:
                    errors.append(f"Item {idx}: Región inválida")
                    continue
                
                if "payment_type" in item and item["payment_type"] not in VALID_PAYMENTS:
                    errors.append(f"Item {idx}: Método de pago inválido")
                    continue
                
                # Convertir cada item a DataFrame
                df = pd.DataFrame([item])
                
                # Asegurar características
                if metadata:
                    features = metadata['features']['numeric'] + metadata['features']['categorical']
                    for col in features:
                        if col not in df.columns:
                            if col in metadata['features']['numeric']:
                                df[col] = 0
                            else:
                                df[col] = 'unknown'
                
                # Preprocesar
                X = preprocessor.transform(df)
                
                # Predecir
                prediction = model.predict(X)[0]
                probability = model.predict_proba(X)[0][1]
                
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
                
                results.append({
                    "item_index": idx,
                    "prediction": int(prediction),
                    "probability": float(probability),
                    "recommend": bool(prediction),
                    "confidence": confidence,
                    "order_number": item.get("order_number", idx)
                })
                
            except Exception as e:
                errors.append(f"Item {idx}: Error - {str(e)}")
        
        return {
            "count": len(results),
            "successful": len(results),
            "failed": len(errors),
            "results": results,
            "errors": errors if errors else None,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en predicción por lotes: {str(e)}")

# Endpoint de prueba rápida
@app.post("/predict-test")
async def predict_test():
    """Endpoint de prueba con datos predefinidos"""
    test_data = {
        "order_number": 9999,
        "price": 199.90,
        "product_category_name": "beleza_saude",
        "customer_state": "RJ",
        "customer_region": "Sudeste",
        "payment_type": "credit_card",
        "purchase_year": 2023,
        "purchase_month": 6,
        "purchase_day": 20,
        "purchase_weekday": 2,
        "purchase_hour": 15
    }
    
    try:
        # Convertir a DataFrame
        df = pd.DataFrame([test_data])
        
        # Asegurar características
        if metadata:
            features = metadata['features']['numeric'] + metadata['features']['categorical']
            for col in features:
                if col not in df.columns:
                    if col in metadata['features']['numeric']:
                        df[col] = 0
                    else:
                        df[col] = 'unknown'
        
        # Preprocesar
        X = preprocessor.transform(df)
        
        # Predecir
        prediction = model.predict(X)[0]
        probability = model.predict_proba(X)[0][1]
        
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
            "test_data": test_data,
            "prediction": int(prediction),
            "probability": float(probability),
            "recommend": bool(prediction),
            "confidence": confidence,
            "message": "✅ Prueba exitosa"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en prueba: {str(e)}")

# Si se ejecuta directamente
if __name__ == "__main__":
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )