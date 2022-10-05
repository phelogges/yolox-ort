# yolox-ort
A light yolox inference library with onnxruntime backend.

## Install
```shell
$ git clone https://github.com/phelogges/yolox-ort.git
$ cd yolox-ort
$ pip install . # this will install yolox_ort into your site-package dir
$ # python setup.py develop # not install into your site-package, just a soft link point to this project
```


# Usage
### Commandline
```shell
$ python yolox_ort.commandline.py -f ${IMAGE_PATH} -d ${OUTPUT_DIR} -m ${MODEL_FILE}
```
Also support video and camera sources, see yolox_ort/commandline.py for more detail

### As library
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

## Models
Currently this package included yolox_nano.onnx model in yolox_ort/assets/models/yolox_nano.onnx.

Your can download more models from [YOLOX official project onnxruntime module](https://github.com/Megvii-BaseDetection/YOLOX/tree/main/demo/ONNXRuntime).

## Reference
[YOLOX official project](https://github.com/Megvii-BaseDetection/YOLOX)