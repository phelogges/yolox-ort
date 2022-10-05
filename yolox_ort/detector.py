# AUTHOR: raichu
# CONTACT: 1012415660@qq.com
# FILE: detector.py
# DATE: 2022/10/5

__all__ = ["get_providers", "Detector"]

import time

# import .utils
from yolox_ort import utils

import numpy as np
import cv2
import onnxruntime
from loguru import logger


def get_providers():
    if onnxruntime.get_device() == "GPU":
        logger.info("Found GPU device, set both GPU and CPU devices and GPU has priority")
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    elif onnxruntime.get_device() == "CPU":
        logger.warning("Only found CPU device, you may get much lower inference with CPU, "
                       "we highly recommend to install Nvidia CUDA instead for faster inference")
        providers = ["CPUExecutionProvider"]
    else:
        msg = "Cannot found any available devices for inference, check your hardware and libraries first"
        logger.error(msg)
        raise ValueError(msg)
    return providers


class Detector:
    def __init__(self,
                 model_file: str,
                 nms_threshold: float = 0.45,
                 score_threshold: float = 0.3,
                 verbose: bool = False,
                 with_p6: bool = False):
        self.verbose = verbose
        logger.info("Initializing detector...")

        self.model_file = model_file
        self.nms_threshold = nms_threshold
        self.score_threshold = score_threshold

        self.with_p6 = with_p6

        providers = get_providers()
        self.session = onnxruntime.InferenceSession(model_file, providers=providers)
        logger.info("Model loaded")

        self.input_shape = self.session.get_inputs()[0].shape[2:]
        if self.verbose:
            logger.info("model file: {}, input shape: {}, nms threshold: {}, "
                        "score threshold: {}, with p6: {}".format(self.model_file,
                                                                  self.input_shape,
                                                                  self.nms_threshold,
                                                                  self.score_threshold,
                                                                  self.with_p6))
        else:
            logger.info("model file: {}, input shape: {}".format(self.model_file, self.input_shape))

        logger.info("Session warming...")
        self._warmup()
        logger.info("Session warmed")
        logger.info("Detector initialized")

    def detect_from_bgr_ndarray(self,
                                bgr: np.ndarray):
        """
        Detect from bgr ordered ndarray image
        Args:
            bgr: ndarray, bgr image

        Returns: ndarray with shape [N, 6]
            N for N entities, 6 for [x1, y1, x2, y2, score, class_index]

        """
        t0 = time.time()
        img, ratio = self._preprocess(bgr)
        outputs = self._infer(img)
        dets = self._postprocess(outputs[0], ratio)
        if dets is not None:
            logger.info("{} entities detected, all cost {:.4f}ms".format(dets.shape[0], (time.time() - t0) * 1000))
        else:
            logger.info("No entities detected, all cost {:.4f}ms".format((time.time()-t0) * 1000))
        return dets

    def detect_from_img_path(self,
                             img_path: str):
        bgr = cv2.imread(img_path)
        return self.detect_from_bgr_ndarray(bgr)

    def _preprocess(self,
                    img: np.ndarray) -> (np.ndarray, float):
        if self.verbose:
            t0 = time.time()
        res = utils.preprocess(img, self.input_shape)
        if self.verbose:
            logger.info("Preprocess cost {:.4f}ms".format((time.time() - t0) * 1000))
        return res

    def _postprocess(self,
                     data: np.ndarray,
                     ratio: float):
        if self.verbose:
            t0 = time.time()
        res = utils.postprocess(data[0], self.input_shape, ratio,
                                self.nms_threshold, self.score_threshold,
                                True, self.with_p6)
        if self.verbose:
            logger.info("Postprocess cost {:.4f}ms".format((time.time() - t0) * 1000))
        return res

    def _infer(self,
               img: np.ndarray):
        if self.verbose:
            t0 = time.time()
        ort_inputs = {self.session.get_inputs()[0].name: img[None, :, :, :]}
        res = self.session.run(None, ort_inputs)  # NCHW
        if self.verbose:
            logger.info("Inference cost {:.4f}ms".format((time.time() - t0) * 1000))
        return res

    def _warmup(self):
        fake = np.random.uniform(size=(3, self.input_shape[0], self.input_shape[1])).astype(np.float32)
        _ = self._infer(fake)
