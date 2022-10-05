# AUTHOR: raichu
# CONTACT: 1012415660@qq.com
# FILE: utils.py
# DATE: 2022/10/5

"""
Note that utils functions comes from YOLOX official project https://github.com/Megvii-BaseDetection/YOLOX
"""
import json
from typing import List

import numpy as np
import cv2


def multiclass_nms(boxes, scores, nms_thr, score_thr, class_agnostic=True):
    """Multiclass NMS implemented in Numpy"""
    if class_agnostic:
        nms_method = multiclass_nms_class_agnostic
    else:
        nms_method = multiclass_nms_class_aware
    return nms_method(boxes, scores, nms_thr, score_thr)


def multiclass_nms_class_aware(boxes, scores, nms_thr, score_thr):
    """Multiclass NMS implemented in Numpy. Class-aware version."""
    final_dets = []
    num_classes = scores.shape[1]
    for cls_ind in range(num_classes):
        cls_scores = scores[:, cls_ind]
        valid_score_mask = cls_scores > score_thr
        if valid_score_mask.sum() == 0:
            continue
        else:
            valid_scores = cls_scores[valid_score_mask]
            valid_boxes = boxes[valid_score_mask]
            keep = nms(valid_boxes, valid_scores, nms_thr)
            if len(keep) > 0:
                cls_inds = np.ones((len(keep), 1)) * cls_ind
                dets = np.concatenate(
                    [valid_boxes[keep], valid_scores[keep, None], cls_inds], 1
                )
                final_dets.append(dets)
    if len(final_dets) == 0:
        return None
    return np.concatenate(final_dets, 0)


def multiclass_nms_class_agnostic(boxes, scores, nms_thr, score_thr):
    """Multiclass NMS implemented in Numpy. Class-agnostic version."""
    cls_inds = scores.argmax(1)
    cls_scores = scores[np.arange(len(cls_inds)), cls_inds]

    valid_score_mask = cls_scores > score_thr
    if valid_score_mask.sum() == 0:
        return None
    valid_scores = cls_scores[valid_score_mask]
    valid_boxes = boxes[valid_score_mask]
    valid_cls_inds = cls_inds[valid_score_mask]
    keep = nms(valid_boxes, valid_scores, nms_thr)
    if keep:
        dets = np.concatenate(
            [valid_boxes[keep], valid_scores[keep, None], valid_cls_inds[keep, None]], 1
        )
    return dets


def nms(boxes, scores, nms_thr):
    """Single class NMS implemented in Numpy."""
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

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

        inds = np.where(ovr <= nms_thr)[0]
        order = order[inds + 1]

    return keep


def preprocess(img, input_size, swap=(2, 0, 1)):
    """
    Single image preprocess, note that yolox use opencv as image reader, and train with bgr order,
    see yolox.data.data_augment.TrainTransform for detail
    Args:
        img: numpy ndarray, input image read by opencv, either 3 or 1 channel
        input_size: List[int], target input size for model, usually 460 x 460 or 416 x 416
        swap: tuple, hwc to chw

    Returns: tuple, (ndarray, float), processed image and resize scaler from input_size

    """
    if len(img.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
    else:
        padded_img = np.ones(input_size, dtype=np.uint8) * 114

    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.uint8)
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

    padded_img = padded_img.transpose(swap)
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    return padded_img, r


def postprocess(outputs: np.ndarray,
                img_size: List[int],
                ratio: float,
                nms_threshold: float,
                score_threshold: float,
                class_agnostic: bool = True,
                p6=False):
    """
    Single inference output postprocess
    Args:
        outputs: ndarray, inference output, Note that just single inference, e.g. [c, h, w], batch dim excluded
        img_size: List[int], target input size for model, usually 460 x 460 or 416 x 416
        ratio: float, target_size / input_size, get from preprocess function
        nms_threshold: float, nms threshold
        score_threshold: float, score threshold
        class_agnostic: bool, filter max class score as result if true, else filter all classes with scores above score threshold
        p6: bool, whether your model uses p6 in FPN/PAN

    Returns: ndarray with shape [N, 6]
            N for N entities, 6 for [x1, y1, x2, y2, score, class_index]
    """
    grids = []
    expanded_strides = []

    if not p6:
        strides = [8, 16, 32]
    else:
        strides = [8, 16, 32, 64]

    hsizes = [img_size[0] // stride for stride in strides]
    wsizes = [img_size[1] // stride for stride in strides]

    for hsize, wsize, stride in zip(hsizes, wsizes, strides):
        xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
        grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
        grids.append(grid)
        shape = grid.shape[:2]
        expanded_strides.append(np.full((*shape, 1), stride))

    grids = np.concatenate(grids, 1)
    expanded_strides = np.concatenate(expanded_strides, 1)
    outputs[..., :2] = (outputs[..., :2] + grids) * expanded_strides
    outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * expanded_strides

    boxes = outputs[:, :4]
    scores = outputs[:, 4:5] * outputs[:, 5:]
    boxes_xyxy = np.ones_like(boxes)
    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.
    boxes_xyxy /= ratio

    dets = multiclass_nms(boxes_xyxy, scores, nms_threshold, score_threshold, class_agnostic)
    return dets


def draw(img: np.ndarray,
         dets: np.ndarray):
    """
    Draw detection info on original image
    Args:
        img: ndarray, image as canvas
        dets: ndarray with shape [N, 6], N for N entities, 6 for [x1, y1, x2, y2, score, class_index]
    """
    for i in dets:
        bbox = [int(j) for j in i[:4]]
        score = float(i[4])
        class_index = int(i[5])
        class_name = COCO_CLASSES[class_index]
        msg = "{:.3f} {}-{}".format(score, class_index, class_name)
        img = cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color=(0, 255, 0))
        img = cv2.putText(img, msg, (bbox[0], bbox[1]), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)
    return img


def write_dets_into_json(file: str,
                         dets:np.ndarray):
    """
    Write detection results into json file
    Args:
        file: str, json file to dummy
        dets: ndarray, detection results

    Returns:

    """
    d = {}
    for i in dets:
        bbox = [float(j) for j in i[:4]]
        score = float(i[4])
        class_index = int(i[5])
        class_name = COCO_CLASSES[class_index]
        if d.get(class_name) is None:
            d[class_name] = [[bbox[0], bbox[1], bbox[2], bbox[3], score]]
        else:
            d[class_name].append([bbox[0], bbox[1], bbox[2], bbox[3]])
    with open(file, 'w') as f:
        json.dump(d, f)



COCO_CLASSES = (
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
)