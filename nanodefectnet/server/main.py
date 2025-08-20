from fastapi import FastAPI, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic.main import BaseModel

from nanodefectnet.utils.logger import LoggerConfig
from nanodefectnet.core.inference_classification_model import run_inference_model
from nanodefectnet.utils.image_utils import load_image_from_bytes

LOGGER = LoggerConfig().logger

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Prediction(BaseModel):
    label: str
    prob: float


@app.post("/api/predict-waferdefect", response_model=Prediction)
async def predict_waferdefect(
    file: UploadFile = File(...), model_name: str = Query(...)
):
    image_bytes = await file.read()
    image = load_image_from_bytes(image_bytes)
    predicted_class, predicted_score = run_inference_model(
        model_name=model_name,
        image=image,
        inference_config_file_path="configs/inference/infer.yaml",
    )
    prediction = Prediction(label=predicted_class, prob=predicted_score)
    LOGGER.info(prediction)
    return prediction
