# yolox-ort
A light yolox inference library with onnxruntime backend.

# Usage

## Commandline
```shell
$ python yolox_ort.commandline.py -f ${IMAGE_PATH} -d ${OUTPUT_DIR} -m ${MODEL_FILE}
```
Also support video and camera sources, see yolox_ort/commandline.py for more detail

## As library
```python
import yolox_ort
import cv2

model_file = "your/model/file"

img_path = "your/image/path"
bgr = cv2.imread(img_path)

detector = yolox_ort.detector.Detector(model_file, nms_threshold=0.45, 
                                       score_threshold=0.3, verbose=True)
dets = detector.detect_from_bgr_ndarray(bgr)

img = yolox_ort.utils.draw(bgr, dets)
cv2.imwrite("result.jpg", img)

```

## Reference
[YOLOX official implementation](https://github.com/Megvii-BaseDetection/YOLOX)