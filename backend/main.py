from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = FastAPI(title="Medical AI: Pneumonia Detection Suite")

# --- 1. CORS CONFIGURATION (Full-Stack/DevOps) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 2. MODEL LOADING ---
try:
    # Use compile=False if you only need inference (saves memory)
    model = tf.keras.models.load_model("pneumonia_model.h5", compile=False)
    print("✅ High-Performance Model Loaded")
except Exception as e:
    print(f"❌ Critical Error: Model file not found. {e}")

@app.get("/")
def health_check():
    return {"status": "online", "model": "MobileNetV2_Pneumonia_v1"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # --- 3. INPUT VALIDATION (Software Testing) ---
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(status_code=400, detail="Unsupported file format. Please use JPG/PNG.")

    try:
        # --- 4. PREPROCESSING PIPELINE (Deep Learning Sync) ---
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # High-quality resizing (LANCZOS) matches professional training pipelines
        image = image.resize((224, 224), Image.Resampling.LANCZOS)
        
        # Explicitly set to float32 and normalize to [0, 1]
        img_array = np.array(image).astype('float32') / 255.0
        
        # Expand dimensions to create a 'batch' of 1: (1, 224, 224, 3)
        img_array = np.expand_dims(img_array, axis=0)

        # --- 5. INFERENCE ---
        prediction = model.predict(img_array, verbose=0)
        raw_score = float(prediction[0][0])
        
        # Logic: 0 = Normal, 1 = Pneumonia (based on standard Kaggle class indices)
        is_pneumonia = raw_score > 0.5
        result = "PNEUMONIA" if is_pneumonia else "NORMAL"
        
        # Calculate confidence based on the distance from the 0.5 threshold
        confidence = raw_score if is_pneumonia else (1.0 - raw_score)

        return {
            "prediction": result,
            "confidence": f"{round(confidence * 100, 2)}%",
            "api_status": "certified",
            "metadata": {
                "model_architecture": "MobileNetV2",
                "input_shape": "224x224x3"
            }
        }

    except Exception as e:
        # Detailed error for the Testing Report
        raise HTTPException(status_code=500, detail=f"Inference Engine Error: {str(e)}")