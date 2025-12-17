import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from io import BytesIO
from PIL import Image

app = FastAPI()

# Load model from current folder
model = tf.keras.models.load_model('imageclassifier.keras')

templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(request: Request, file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(BytesIO(contents)).convert("RGB")

    image = image.resize((256, 256))
    img_array = np.array(image)
    img_array = np.expand_dims(img_array / 255.0, axis=0)

    prediction = model.predict(img_array)
    score = prediction[0][0]

    if score < 0.5:
        result = f"Happy 😄 ({score:.2f})"
        color = "green"
    else:
        result = f"Sad 😢 ({score:.2f})"
        color = "red"

    return templates.TemplateResponse(
        "index.html",
        {"request": request, "prediction": result, "color": color}
    )
