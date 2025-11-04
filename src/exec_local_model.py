import asyncio
import json
import os
import sys
from io import BytesIO
from dotenv import load_dotenv

import aio_pika
import cv2
import numpy as np

from .vision_utils import get_model

current_directory = os.path.dirname(__file__)
rabbitmq_connector = os.path.join(
    current_directory, "..", "..", "sneaky-vaccine-backend"
)
sys.path.append(rabbitmq_connector)

from rabbit import RabbitMQ

rmq = RabbitMQ()

load_dotenv('.env', override=True)
OUTPUT_IMAGES_PATH = os.getenv("OUTPUT_IMAGES_PATH")


def model_exec(image_path: str) -> np.ndarray:

    base_name = os.path.basename(image_path)
    print(f"image base name: {base_name}")

    path_to_save = os.path.join(OUTPUT_IMAGES_PATH, base_name)
    print(f"image save path: {path_to_save}")

    image = cv2.imread(image_path)

    if not isinstance(image, np.ndarray):
        raise ValueError("image must be ndarray")

    model_path = os.path.join(os.path.dirname(__file__), "../model_zoo", "det_10g.onnx")

    detector = get_model(model_path)
    detector.prepare(ctx_id=0, input_size=(640, 640))

    if detector is None:
        raise Exception("Model is None")

    if image is None:
        raise ValueError(f"Could not read image: {image}")

    det, landmarks = detector.detect(image)

    # draw bbox
    # for f in det:
    #     x1, x2 = int(f[0]), int(f[1])
    #     y1, y2 = int(f[2]), int(f[3])
    #     image = cv2.rectangle(image, (x1,x2), (y1, y2), (0,0,255), 2)

    # draw landmarks
    for landmark in landmarks:
        for point in landmark:
            x, y = int(point[0]), int(point[1])
            cv2.circle(image, (x, y), 2, (0, 0, 255), -3)

    cv2.imwrite(path_to_save, image)

    return path_to_save


async def image_from_tg_callback(message: aio_pika.IncomingMessage):
    async with message.process():

        data = json.loads(message.body.decode())
        print(f"Neural net received: {data}")

        image_path = data.get("path_to_processing_image", None)
        user_id = data.get("user_id", None)

        output = model_exec(image_path)

        processed_result = {"processed_image_data": output, "user_id": user_id}

        await rmq.send_processed_image(processed_result)


async def processing_tg_images():
    await rmq.consume_images_from_tg(image_from_tg_callback)

    await asyncio.Event().wait()


if __name__ == "__main__":

    asyncio.run(processing_tg_images())
