import asyncio
import base64
import os
from io import BytesIO

import cv2
import grpc
import numpy as np
import tritonclient.grpc.aio as grpcclient
from PIL import Image
from torchvision import transforms
from tritonclient.grpc import service_pb2, service_pb2_grpc
from tritonclient.utils import triton_to_np_dtype

from src.vision_utils import get_model
from src.vision_utils.image_handler import ImageHandler


class TritonRetinaFace:
    def __init__(self):
        self.url = os.environ.get("TRITON_URL", "127.0.0.1:8001")
        self.triton_client = grpcclient.InferenceServerClient(url=self.url)
        self.nms_thresh = 0.4
        self.det_thresh = 0.5
        self._init_vars()

    def _init_vars(self):
        self.fmc = 3
        self._feat_stride_fpn = [8, 16, 32]
        self._num_anchors = 2
        self.use_kps = True
        self.input_mean = 127.5
        self.input_std = 128.0
        self.center_cache = {}

    async def forward(self, img: np.array, threshold: int, model_name: str) -> (list, list, list):
        scores_list = []
        bboxes_list = []
        kpss_list = []

        output_mapping = {
            0: {"scores": "448", "bbox": "451", "kps": "454"},
            1: {"scores": "471", "bbox": "474", "kps": "477"},
            2: {"scores": "494", "bbox": "497", "kps": "500"},
        }

        blob = cv2.dnn.blobFromImage(
            img,
            1.0 / self.input_std,
            (img.shape[1], img.shape[0]),
            (self.input_mean, self.input_mean, self.input_mean),
            swapRB=True,
        )

        model_meta, model_config = self.parse_model_metadata(model_name)
        dtype = model_meta.inputs[0].datatype

        if len(img.shape) == 3:
            img = np.expand_dims(img, axis=0)

        if img.shape[-1] in [3, 4]:
            img = np.transpose(img, (0, 3, 1, 2))

        inputs = [grpcclient.InferInput(model_meta.inputs[0].name, img.shape, dtype)]
        inputs[0].set_data_from_numpy(img)

        outputs = []
        for idx in range(self.fmc):
            outputs.append(
                grpcclient.InferRequestedOutput(output_mapping[idx]["scores"])
            )
            outputs.append(grpcclient.InferRequestedOutput(output_mapping[idx]["bbox"]))
            if self.use_kps:
                outputs.append(
                    grpcclient.InferRequestedOutput(output_mapping[idx]["kps"])
                )

        result = await self.triton_client.infer(
            model_name=model_name, inputs=inputs, outputs=outputs
        )

        input_height = blob.shape[2]
        input_width = blob.shape[3]

        for idx, stride in enumerate(self._feat_stride_fpn):
            scores = result.as_numpy(output_mapping[idx]["scores"]).reshape(-1)
            bbox_preds = (
                result.as_numpy(output_mapping[idx]["bbox"]).reshape(-1, 4) * stride
            )

            height = input_height // stride
            width = input_width // stride
            key = (height, width, stride)

            if key in self.center_cache:
                anchor_centers = self.center_cache[key]
            else:
                anchor_centers = np.stack(
                    np.mgrid[:height, :width][::-1], axis=-1
                ).astype(np.float32)
                anchor_centers = (anchor_centers * stride).reshape((-1, 2))
                if self._num_anchors > 1:
                    anchor_centers = np.stack(
                        [anchor_centers] * self._num_anchors, axis=1
                    ).reshape((-1, 2))
                if len(self.center_cache) < 100:
                    self.center_cache[key] = anchor_centers

            bboxes = self.distance2bbox(anchor_centers, bbox_preds)

            pos_inds = np.where(scores >= threshold)[0]
            if len(pos_inds) == 0:
                continue

            scores_list.append(scores[pos_inds])
            bboxes_list.append(bboxes[pos_inds])

        if self.use_kps:
            kps_preds = (
                result.as_numpy(output_mapping[idx]["kps"]).reshape(-1, 10) * stride
            )
            kpss = self.distance2kps(anchor_centers, kps_preds)
            kpss = kpss.reshape((kpss.shape[0], -1, 2))
            kpss_list.append(kpss[pos_inds])

        return scores_list, bboxes_list, kpss_list

    async def detect(self,
                 img: base64,
                 input_size = None,
                 max_num: int = 0,
                 model_name: str = None
        ) -> dict:

        model_name = model_name
        assert input_size is not None

        img = self.preprocess_image(img)

        im_ratio = float(img.shape[0]) / img.shape[1]
        model_ratio = float(input_size[1]) / input_size[0]

        if im_ratio > model_ratio:
            new_height = input_size[1]
            new_width = int(new_height / im_ratio)
        else:
            new_width = input_size[0]
            new_height = int(new_width * im_ratio)

        det_scale = float(new_height) / img.shape[0]
        resized_img = cv2.resize(img, (new_width, new_height))
        det_img = np.zeros((input_size[1], input_size[0], 3), dtype=np.float32)
        det_img[:new_height, :new_width, :] = resized_img

        scores_list, bboxes_list, kpss_list = await self.forward(
            det_img, self.det_thresh, model_name
        )

        if len(scores_list) == 0 or len(bboxes_list) == 0:
            return np.zeros((0, 5)), None

        scores = np.concatenate([s.flatten() for s in scores_list])
        bboxes = np.concatenate([b for b in bboxes_list])

        if self.use_kps and len(kpss_list) > 0:
            kpss = np.concatenate([k for k in kpss_list])
        else:
            kpss = None

        order = scores.argsort()[::-1]
        bboxes = bboxes[order] / det_scale
        scores = scores[order]

        if kpss is not None:
            kpss = kpss[order] / det_scale

        detections = np.column_stack([bboxes, scores])

        keep = self.nms(detections)
        final_detections = detections[keep]

        final_kpss = kpss[keep] if kpss is not None else None

        if max_num > 0 and final_detections.shape[0] > max_num:
            area = (final_detections[:, 2] - final_detections[:, 0]) * (
                final_detections[:, 3] - final_detections[:, 1]
            )
            img_center = np.array([img.shape[1] / 2, img.shape[0] / 2])
            centers = np.column_stack(
                [
                    (final_detections[:, 0] + final_detections[:, 2]) / 2,
                    (final_detections[:, 1] + final_detections[:, 3]) / 2,
                ]
            )
            offset_dist_sq = np.sum((centers - img_center) ** 2, axis=1)

            values = area - offset_dist_sq * 2.0
            bindex = np.argsort(values)[::-1][:max_num]

            final_detections = final_detections[bindex]
            if final_kpss is not None:
                final_kpss = final_kpss[bindex]

        return {"det": final_detections, "kpss": final_kpss}

    def parse_model_metadata(self, model_name: str) -> (object, object):
        channel = grpc.insecure_channel(self.url)
        grpc_stub = service_pb2_grpc.GRPCInferenceServiceStub(channel)
        metadata_request = service_pb2.ModelMetadataRequest(name=model_name)
        metadata_response = grpc_stub.ModelMetadata(metadata_request)

        config_request = service_pb2.ModelConfigRequest(name=model_name)
        config_response = grpc_stub.ModelConfig(config_request)

        return metadata_response, config_response

    @staticmethod
    def distance2bbox(points, distance) -> np.array:
        x1 = points[:, 0] - distance[:, 0]
        y1 = points[:, 1] - distance[:, 1]
        x2 = points[:, 0] + distance[:, 2]
        y2 = points[:, 1] + distance[:, 3]
        return np.stack([x1, y1, x2, y2], axis=-1)

    @staticmethod
    def distance2kps(points, distance) -> np.array:
        preds = []
        for i in range(0, distance.shape[1], 2):
            px = points[:, 0] + distance[:, i]
            py = points[:, 1] + distance[:, i + 1]
            preds.append(px)
            preds.append(py)
        return np.stack(preds, axis=-1).reshape(-1, 5, 2)

    def nms(self, dets: list) -> list[int]:
        thresh = self.nms_thresh
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scores = dets[:, 4]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]

        return keep

    def preprocess_image(self, image_data: object) -> np.array:
        if isinstance(image_data, str):
            image_data = ImageHandler.decode_base64_to_numpy(image_data)

        if image_data.dtype != np.float32:
            image_data = image_data.astype(np.float32)

        image_data = (image_data - self.input_mean) / self.input_std

        return image_data
