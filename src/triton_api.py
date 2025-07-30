import asyncio
import base64
from io import BytesIO

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from PIL import Image

from .exec_model import TritonRetinaFace

app = FastAPI()

triton_face = TritonRetinaFace()


@app.post("/predict/", description="model processing")
async def predict(
    model_name: str = Query(..., description="triton model"),
    file: UploadFile = File(..., description="image for processing"),
):
    try:
        image = await file.read()

        if isinstance(image, bytes):
            img_base64 = base64.b64encode(image).decode("utf-8")

        res = await triton_face.detect(
            img_base64, input_size=(640, 640), model_name=model_name
        )

        det = res.get("det")
        kpss = res.get("kpss")

        if det is not None:
            det = det.tolist() if hasattr(det, "tolist") else det
        if kpss is not None:
            kpss = kpss.tolist() if hasattr(kpss, "tolist") else kpss

        return {
            "det": det,
            "kpss": kpss,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
