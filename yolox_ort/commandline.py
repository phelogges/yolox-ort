# AUTHOR: raichu
# CONTACT: 1012415660@qq.com
# FILE: commandline.py
# DATE: 2022/10/5

import argparse
import os

from yolox_ort.detector import Detector
from yolox_ort import utils

import cv2
from loguru import logger


def make_path(save_dir: str,
              source_file_path: str):
    source_file_name, ext = os.path.splitext(os.path.basename(source_file_path))
    return os.path.join(save_dir, source_file_name + "_result")


def arg_parser():
    parser = argparse.ArgumentParser("YOLOX Onnxruntime inference commandline")
    # Just one input source once run by file, video, camera priority
    parser.add_argument('-f', "--file", type=str, default=None, help="Inference from image file path")
    parser.add_argument('-v', "--video", type=str, default=None, help="Inference from video file path")
    parser.add_argument('-c', "--camera", type=int, default=None, help="Inference from camera, use camera index")

    parser.add_argument('-d', "--output_dir", type=str, default=None , help="Output directory for writing")

    parser.add_argument('-m', "--model_file", type=str, help="Model file path")
    parser.add_argument('-n', "--nms_threshold", type=float, default=0.45, help="NMS threshold")
    parser.add_argument('-s', "--score_threshold", type=float, default=0.3, help="Score threshold")
    parser.add_argument('-v', "--verbose", type=int, default=1, help="0/1, whether to log in verbose mode")
    return parser.parse_args()


if __name__ == "__main__":
    args = arg_parser()

    source = None
    source_type = -1  # 0 for file, 1 for video, 2 for camera
    if args.file is not None:
        source = args.file
        source_type = 0
        logger.info("Got image file {} as source".format(source))
    elif args.video is not None:
        source = args.video
        source_type = 1
        logger.info("Got video file {} as source".format(source))
    elif args.camera is not None:
        source = args.camera
        source_type = 2
        logger.info("Got camera index {} as source".format(source))
    else:
        msg = "Neither file, video, camera sources are offered, set one source first"
        logger.error(msg)
        raise ValueError(msg)

    output_dir = args.output_dir
    if output_dir is None:
        msg = "Output directory for saving results should be set first"
        logger.error(msg)
        raise ValueError(msg)

    model_file = args.model_file
    nms_threshold = args.nms_threshold
    score_threshold = args.score_threshold
    verbose = bool(args.verbose)

    logger.info("output dir: {}, model file: {}, nms threshold: {},"
                "score threshold: {}, verbose: {}".format(output_dir, model_file,
                                                          nms_threshold, score_threshold, verbose))

    detector = Detector(model_file, nms_threshold, score_threshold, verbose)

    if source_type == 0:
        logger.info("Detecting image {}".format(source))
        dets = detector.detect_from_img_path(source)
        saved_path = make_path(output_dir, source) + ".jpg"
        img = utils.draw(cv2.imread(source), dets)
        cv2.imwrite(saved_path, img)
        logger.info("Detection finished, result image saved into {}".format(saved_path))
    elif source == 1:
        logger.info("Detecting video file {}".format(source))
        cap = cv2.VideoCapture(source)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        logger.info("Video file meta infos: {}x{}, {} fps, {} frames".format(width, height, fps, frame_count))

        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        saved_path = make_path(output_dir, source) + ".mp4"
        writer = cv2.VideoWriter(saved_path, fourcc, fps, (width, height))
        logger.info("Video writer prepared")

        counter = 1
        try:
            while True:
                if counter % 50 == 0:
                    logger.info("On frame index {}/{}".format(counter, frame_count))
                ret, bgr = cap.read()
                if ret:
                    dets = detector.detect_from_bgr_ndarray(bgr)
                    img = utils.draw(bgr, dets)
                    writer.write(img)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        logger.info("Received quit input on frame index {}/{}, "
                                    "releasing capture and writer...".format(counter, frame_count))
                        break
                else:
                    logger.info("All video frames processed, releasing capture and writer...")
                    break
            cap.release()
            writer.release()
            logger.info("Capture and writer released, result saved into {}".format(saved_path))
        except KeyboardInterrupt:
            logger.info("Received KeyboardInterrupt on frame index {}/{}, "
                        "releasing capture and writer...".format(counter, frame_count))
            cap.release()
            writer.release()
            logger.info("Capture and writer released, result saved into {}".format(saved_path))

    elif source_type == 3:
        logger.info("Detecting from camera index {}".format(source))
        cap = cv2.VideoCapture(source)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fps = cap.get(cv2.CAP_PROP_FPS)
        logger.info("Camera index {} meta infos: {}x{}, {} fps".format(source, width, height, fps))

        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        saved_path = os.path.join(output_dir, "camera{}.mp4".format(source))
        writer = cv2.VideoWriter(saved_path, fourcc, fps, (width, height))
        logger.info("Video writer prepared")

        counter = 1
        try:
            while True:
                if counter % 50 == 0:
                    logger.info("On frame index {}".format(counter))
                ret, bgr = cap.read()
                if ret:
                    dets = detector.detect_from_bgr_ndarray(bgr)
                    img = utils.draw(bgr, dets)
                    writer.write(img)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        logger.info("Received quit input on frame index {}, "
                                    "releasing capture and writer...".format(counter))
                        break
                else:
                    logger.info("camera error, releasing capture and writer...")
                    break
            cap.release()
            writer.release()
            logger.info("Capture and writer released, result saved into {}".format(saved_path))
        except KeyboardInterrupt:
            logger.info("Received KeyboardInterrupt on frame index {}, "
                        "releasing capture and writer...".format(counter))
            cap.release()
            writer.release()
            logger.info("Capture and writer released, result saved into {}".format(saved_path))



