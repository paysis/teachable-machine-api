import os
import logging
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import tensorflow as tf
from typing import Dict, Type
from preprocessors.audio_preprocessor import AudioPreprocessor
from preprocessors.image_preprocessor import ImagePreprocessor
import argparse
from enum import Enum
from pydantic import BaseModel
from PIL import Image, UnidentifiedImageError
import librosa
import numpy as np
import magic

# Disable scientific notation
np.set_printoptions(suppress=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelType(str, Enum):
    IMAGE = "image"
    AUDIO = "audio"

# Strategy pattern cuz it says they may add new models to their website
PREPROCESSORS: Dict[str, Type] = {
    ModelType.IMAGE: ImagePreprocessor,
    ModelType.AUDIO: AudioPreprocessor
}

# We will compare model input type to this map using *magic* :)
SUPPORTED_FILE_TYPES = {
    ModelType.IMAGE: ["image/jpeg", "image/png", "image/jpg"],
    ModelType.AUDIO: ["audio/wav", "audio/mpeg", "audio/mp3"]
}

parser = argparse.ArgumentParser(
    description="Teachable Machine Inference API"
)
parser.add_argument("--model-dir", type=str, default="models", help="Directory to store h5 models")
parser.add_argument("--host", type=str, default="127.0.0.1", help="IP to bind the API. Defaults to all interfaces cuz we will use this")
parser.add_argument("--port", type=int, default=8080, help="Port to bind the API")
args = parser.parse_args()

# Model directory must exist
os.makedirs(args.model_dir, exist_ok=True)

app = FastAPI(
    title="Teachable Machine Inference API",
    description="Bu API, TensorFlow modelleri kullanarak resim ve ses dosyaları üzerinde tahmin yapar.",
    version="0.1.0",
    license_info={
        "name": "MIT"
    },
    openapi_tags=[
        {
            "name": "Tahmin",
            "description": "Model çalıştırma ve tahmin sonuçları elde etme ile ilgili işlemler"
        }
    ]
)

# Load models into memory (lazy)
# We do not expect much load as the maximum number of students is ~30
MODELS: Dict[str, tf.keras.Model] = {}

def load_model(model_name: str) -> tf.keras.Model:
    """Load a Tensorflow model from disk."""
    model_path = os.path.join(args.model_dir, model_name)
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' bulunamadı!")
    if model_name not in MODELS:
        # H5 did not work so I tried SavedModel
        # But since Keras V3, SavedModel seems to be deprecated
        # Though we can use TFSM layer to make it work, so we did!
        MODELS[model_name] = tf.keras.layers.TFSMLayer(model_path, call_endpoint="serving_default")
        logger.info(f"Loaded model: {model_name}")
    return MODELS[model_name]

def load_labels(model_name: str) -> list:
    """Load the labels for a given model."""
    labels_path = os.path.join(args.model_dir, f"{model_name}_labels.txt")
    if not os.path.exists(labels_path):
        raise HTTPException(status_code=404, detail=f"'{model_name}' için label dosyası bulunamadı.")
    with open(labels_path, "r") as f:
        labels = [line.strip().split(" ")[1] for line in f.readlines()]
    return labels

def validate_file(file: UploadFile, model_type: ModelType):
    """Validate the uploaded file against the expected model type using *magic*. I love this name..."""
    file_type = magic.from_buffer(file.file.read(2048), mime=True)
    file.file.seek(0) # Reset file pointer back to the beginning of the file

    if file_type not in SUPPORTED_FILE_TYPES.get(model_type, []):
        raise HTTPException(
            status_code=400,
            detail=f"model_type={model_type} için desteklenmeyen bir dosya tipi yüklendi. Beklenen dosya tipleri: {SUPPORTED_FILE_TYPES[model_type]}"
        )
    
def map_prediction_to_label(prediction: np.ndarray, labels: list) -> dict:
    """Map prediction probabilities to class labels."""
    if len(prediction.shape) != 2 or prediction.shape[1] != len(labels):
        raise ValueError("Tahmin şekli, label sayısıyla eşleşmiyor.")
    return {label: float(prob) for label, prob in zip(labels, prediction[0])}

@app.post(
    "/predict",
    tags=["Tahmin"],
    summary="Tahmin Yap",
    description="""
    Bu uç nokta, belirli bir model ve yüklenen dosya kullanarak tahmin yapar.

    **UYARI**: Öncelikle modelinizi bildirilen Google Form aracılığıyla göndermeniz ve yetkilileri bilgilendirmeniz gerekmektedir!

    > Aynı anda birden fazla modelin çalıştırılmasını sağlamak için altyapı tarafından yönetilen bir iş parçacığı havuzu kullanır.
    """,
    response_description="Sınıf isimleri ve olasılıklarla birlikte tahmin sonucu.",
    responses={
        200: {
            "description": "Başarılı tahmin",
            "content": {
                "application/json": {
                    "example": {
                        "prediction": {
                            "kedi": 0.92,
                            "köpek": 0.08
                        }
                    }
                }
            }
        },
        400: {
            "description": "Geçersiz dosya türü veya model türü",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "model_type=image için desteklenmeyen bir dosya tipi yüklendi. Beklenen dosya tipleri: ['image/jpeg', 'image/png', 'image/jpg']"
                    }
                }
            }
        },
        404: {
            "description": "Model veya etiket dosyası bulunamadı",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "Model 'kedi_kopek_modeli' bulunamadı."
                    }
                }
            }
        },
        500: {
            "description": "Sunucu tarafında bir hata oluştuğunda bu hata meydana gelir. Böyle bir durum ile karşılaşırsanız lütfen durumu yetkililere bildirin.",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "Internal server error"
                    }
                }
            }
        }
    }
)
def predict(
    model_name: str,
    model_type: ModelType,
    file: UploadFile = File(...)
):
    """
    #### **Parametreler**
    - **`model_name` (str)**: Tahmin yapmak için kullanılacak modelin adı.
    - **`model_type` (ModelType)**: Modelin türü (`image` veya `audio`).
    - **`file` (UploadFile)**: Modele girdi olarak verilecek dosya (resim veya ses).

    #### **Örnekler**
    ##### **1. Resim Tahmini**
    - **Model Adı**: `kedi_kopek_modeli`
    - **Model Türü**: `image`
    - **Dosya**: Bir kedi veya köpek resmi (JPEG veya PNG formatında).

    ##### **2. Ses Tahmini**
    - **Model Adı**: `ses_tanima_modeli`
    - **Model Türü**: `audio`
    - **Dosya**: Bir ses dosyası (WAV veya MP3 formatında).
    """
    try:
        validate_file(file, model_type)

        model = load_model(model_name)

        labels = load_labels(model_name)

        preprocessor = PREPROCESSORS[model_type]()

        # preprocess
        try:
            input_data = preprocessor.preprocess(file.file)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Dosya önişleme hata ile sonuçlandı: {str(e)}")
        
        outputs = model(input_data) # predict using TFSM layer

        # TFSM spits out a dict containing the last layer as key and output tensor as value.
        prediction_with_labels = map_prediction_to_label(outputs[list(outputs.keys())[0]], labels)

        return JSONResponse(content={
            "prediction": prediction_with_labels
        })
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)