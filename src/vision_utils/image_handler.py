import base64
from distutils.command.install import install
from io import BytesIO

import cv2
import numpy as np


class ImageHandler:
    @staticmethod
    def cv_image_read(image_data: object) -> np.array:
        if isinstance(image_data, str):
            img = cv2.imread(image_data)

        return img

    @staticmethod
    def image_to_bytesIO(image_data: str | np.ndarray) -> BytesIO:
        if isinstance(image_data, str):
            image = cv2.imread(image_data)
            _, buffer = cv2.imencode(".jpg", image)
            io_buf = BytesIO(buffer)

        elif isinstance(image_data, np.ndarray):
            _, buffer = cv2.imencode(".jpg", image_data)
            io_buf = BytesIO(buffer)

        else:
            raise ValueError("Input must be either a string (file path) or numpy array")

        return io_buf

    @staticmethod
    def bytesio_decode(image_data: BytesIO) -> np.array:
        if isinstance(image_data, BytesIO):
            decoded_image = cv2.imdecode(
                np.frombuffer(image_data.getbuffer(), np.uint8), -1
            )
        else:
            raise ValueError("image_data must be BytesIO")

        return decoded_image

    @staticmethod
    def decode_base64_to_numpy(img_base64: str) -> np.array:
        if img_base64.startswith("data:image/"):
            header, encoded = img_base64.split(",", 1)
        else:
            encoded = img_base64

        img_bytes = base64.b64decode(encoded)

        img_buffer = BytesIO(img_bytes)

        decoded_image = cv2.imdecode(
            np.frombuffer(img_buffer.getbuffer(), np.uint8), -1
        )

        return decoded_image
