# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.

import numpy as np
import os
import xml.etree.ElementTree as ET
from typing import List, Tuple, Union

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from detectron2.utils.file_io import PathManager

__all__ = ["load_voc_instances", "register_pascal_voc"]


# fmt: off
CLASS_NAMES = (
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
)
# fmt: on


def load_voc_instances(dirname: str, split: str, class_names: Union[List[str], Tuple[str, ...]]):
    """
    Load Pascal VOC detection annotations to Detectron2 format.

    Args:
        dirname: Contain "Annotations", "ImageSets", "JPEGImages"
        split (str): one of "train", "test", "val", "trainval"
        class_names: list or tuple of class names
    """
    with PathManager.open(os.path.join(dirname, "ImageSets", "Main", split + ".txt")) as f:
        fileids = np.loadtxt(f, dtype=np.str)

    # Needs to read many small annotation files. Makes sense at local
    annotation_dirname = PathManager.get_local_path(os.path.join(dirname, "Annotations/"))
    dicts = []
    for fileid in fileids:
        anno_file = os.path.join(annotation_dirname, fileid + ".xml")
        jpeg_file = os.path.join(dirname, "JPEGImages", fileid + ".jpg")

        with PathManager.open(anno_file) as f:
            tree = ET.parse(f)

        r = {
            "file_name": jpeg_file,
            "image_id": fileid,
            "height": int(tree.findall("./size/height")[0].text),
            "width": int(tree.findall("./size/width")[0].text),
        }
        instances = []

        for obj in tree.findall("object"):
            cls = obj.find("name").text
            # We include "difficult" samples in training.
            # Based on limited experiments, they don't hurt accuracy.
            # difficult = int(obj.find("difficult").text)
            # if difficult == 1:
            # continue
            bbox = obj.find("bndbox")
            bbox = [float(bbox.find(x).text) for x in ["xmin", "ymin", "xmax", "ymax"]]
            # Original annotations are integers in the range [1, W or H]
            # Assuming they mean 1-based pixel indices (inclusive),
            # a box with annotation (xmin=1, xmax=W) covers the whole image.
            # In coordinate space this is represented by (xmin=0, xmax=W)
            bbox[0] -= 1.0
            bbox[1] -= 1.0
            if cls in class_names:
                instances.append(
                    {"category_id": CLASS_NAMES.index(cls), "bbox": bbox, "bbox_mode": BoxMode.XYXY_ABS}
                )
        r["annotations"] = instances
        
        if len(instances) > 0:
            dicts.append(r)
    return dicts


def register_pascal_voc(name, dirname, split, year, class_names=CLASS_NAMES):
    DatasetCatalog.register(name, lambda: load_voc_instances(dirname, split, class_names))
    MetadataCatalog.get(name).set(thing_classes=list(CLASS_NAMES), dirname=dirname, year=year, split=split)


# # ==== Predefined splits for PASCAL VOC ===========
# def register_all_pascal_voc(root):
#     SPLITS = [
#         ("voc_2007_trainval", "VOC2007", "trainval"),
#         ("voc_2007_train", "VOC2007", "train"),
#         ("voc_2007_val", "VOC2007", "val"),
#         ("voc_2007_test", "VOC2007", "test"),
#         ("voc_2012_trainval", "VOC2012", "trainval"),
#         ("voc_2012_train", "VOC2012", "train"),
#         ("voc_2012_val", "VOC2012", "val"),
#     ]
#     for name, dirname, split in SPLITS:
#         year = 2007 if "2007" in name else 2012
#         register_pascal_voc(name, os.path.join(root, dirname), split, year)
#         MetadataCatalog.get(name).evaluator_type = "pascal_voc"


def register_incremental_pascal_voc(root, cls_num_splits=[10, 10]):
    SPLITS = [
        ("voc_2007_trainval", "VOC2007", "trainval"),
        ("voc_2007_train", "VOC2007", "train"),
        ("voc_2007_val", "VOC2007", "val"),
        ("voc_2007_test", "VOC2007", "test"),
        ("voc_2012_trainval", "VOC2012", "trainval"),
        ("voc_2012_train", "VOC2012", "train"),
        ("voc_2012_val", "VOC2012", "val"),
    ]
    cls_milestones = [0]
    for cls_num in cls_num_splits:
        cls_milestones.append(cls_milestones[-1] + cls_num)
        
    for name, dirname, split in SPLITS:
        year = 2007 if "2007" in name else 2012
        for idx, cls_milestone in enumerate(cls_milestones[0:-1]):
            name = name + "_" + str(cls_milestone) + "_" + str(cls_milestones[idx+1])
            register_pascal_voc(name,
                                os.path.join(root, dirname),
                                split, year,
                                class_names=CLASS_NAMES[cls_milestone:cls_milestones[idx+1]])
            MetadataCatalog.get(name).evaluator_type = "pascal_voc"

root = os.path.expanduser(os.getenv("DETECTRON2_DATASETS", "datasets"))
register_incremental_pascal_voc(root, [10, 10])
register_incremental_pascal_voc(root, [15, 5])
register_incremental_pascal_voc(root, [19, 1])
register_incremental_pascal_voc(root, [20])




