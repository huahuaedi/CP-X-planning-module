# -*- coding: utf-8 -*-

"""
Since multiple CAV normally use the same ML/DL model,
here we have this class to enable different CAVs share the same model to
 avoid duplicate memory consumption.
"""

# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib

import numpy as np
import os
import shutil
import cv2
import torch


class MLManager(object):
    """
    A class that should contain all the ML models you want to initialize.

    Attributes
    -object_detector : torch_detector
        The YoloV5 detector load from pytorch.

    """

    _shared_object_detector = None

    def __init__(self):
        if MLManager._shared_object_detector is not None:
            self.object_detector = MLManager._shared_object_detector
            return

        repo_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), '..', '..', '..'))
        config_dir = os.environ.setdefault(
            'YOLOV5_CONFIG_DIR',
            os.path.join(os.path.expanduser('~'), '.config', 'Ultralytics'))
        os.makedirs(config_dir, exist_ok=True)
        yolo_font = os.path.join(config_dir, 'Arial.ttf')
        system_font = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
        if not os.path.isfile(yolo_font) and os.path.isfile(system_font):
            shutil.copyfile(system_font, yolo_font)

        weights_path = os.environ.get(
            'OPENCDA_YOLO_WEIGHTS',
            os.path.join(repo_root, 'yolov5m.pt'))
        if not os.path.isfile(weights_path):
            raise FileNotFoundError(
                'YOLOv5 weights not found: %s. Set OPENCDA_YOLO_WEIGHTS '
                'to the yolov5m.pt path.' % weights_path)

        requested_device = os.environ.get(
            'OPENCDA_ML_DEVICE', 'auto').strip().lower()
        if requested_device == 'auto':
            device = 'cpu'
            if torch.cuda.is_available():
                major, minor = torch.cuda.get_device_capability(0)
                current_arch = 'sm_%d%d' % (major, minor)
                if current_arch in torch.cuda.get_arch_list():
                    device = 'cuda:0'
        else:
            device = requested_device
        print('[OpenCDA ML] Loading YOLOv5 on %s.' % device)

        # Pin the hub source to the last YOLOv5 generation supporting the
        # project's Python 3.7 / PyTorch 1.10 runtime.
        self.object_detector = torch.hub.load(
            'ultralytics/yolov5:v6.0',
            'custom',
            path=weights_path,
            device=device,
            force_reload=False)
        MLManager._shared_object_detector = self.object_detector

    def draw_2d_box(self, result, rgb_image, index):
        """
        Draw 2d bounding box based on the yolo detection.

        Args:
            -result (yolo.Result):Detection result from yolo 5.
            -rgb_image (np.ndarray): Camera rgb image.
            -index(int): Indicate the index.

        Returns:
            -rgb_image (np.ndarray): camera image with bbx drawn.
        """
        # torch.Tensor
        bounding_box = result.xyxy[index]
        if bounding_box.is_cuda:
            bounding_box = bounding_box.cpu().detach().numpy()
        else:
            bounding_box = bounding_box.detach().numpy()

        for i in range(bounding_box.shape[0]):
            detection = bounding_box[i]

            # the label has 80 classes, which is the same as coco dataset
            label = int(detection[5])
            label_name = result.names[label]

            if is_vehicle_cococlass(label):
                label_name = 'vehicle'

            x1, y1, x2, y2 = int(
                detection[0]), int(
                detection[1]), int(
                detection[2]), int(
                detection[3])
            cv2.rectangle(rgb_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # draw text on it
            cv2.putText(rgb_image, label_name, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 1)

        return rgb_image


def is_vehicle_cococlass(label):
    """
    Check whether the label belongs to the vehicle class according
    to coco dataset.
    Args:
        -label(int): yolo detection prediction.
    Returns:
        -is_vehicle: bool
            whether this label belongs to the vehicle class
    """
    vehicle_class_array = np.array([1, 2, 3, 5, 7], dtype=int)
    return True if 0 in (label - vehicle_class_array) else False
