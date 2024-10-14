import itertools

from omegaconf import OmegaConf

import detectron2.data.transforms as T
from detectron2.config import LazyCall as L
from detectron2.data import (
    build_detection_test_loader,
    build_detection_train_loader,
    get_detection_dataset_dicts,
)
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator

from groundingdino.datasets import DetrDatasetMapper, MetadataCatalog, DatasetCatalog

dataloader = OmegaConf.create()

if "Aquarium_odinw_train" not in DatasetCatalog.list():
    register_coco_instances("Aquarium_odinw_train", {},
                            "datasets/odinw/Aquarium/Aquarium Combined.v2-raw-1024.coco/train/fewshot_train_shot5_seed3.json",
                            "datasets/odinw/Aquarium/Aquarium Combined.v2-raw-1024.coco/train")
DatasetCatalog.get("Aquarium_odinw_train")
train_thing_classes = MetadataCatalog.get("Aquarium_odinw_train").thing_classes


if "Aquarium_odinw_test" not in DatasetCatalog.list():
    register_coco_instances("Aquarium_odinw_test", {},
                            "datasets/odinw/Aquarium/Aquarium Combined.v2-raw-1024.coco/test/annotations_without_background.json",
                            "datasets/odinw/Aquarium/Aquarium Combined.v2-raw-1024.coco/test")
DatasetCatalog.get("Aquarium_odinw_test")
test_thing_classes = MetadataCatalog.get("Aquarium_odinw_test").thing_classes


dataloader.train = L(build_detection_train_loader)(
    dataset=L(get_detection_dataset_dicts)(names="Aquarium_odinw_train"),
    mapper=L(DetrDatasetMapper)(
        augmentation=[
            L(T.RandomFlip)(),
            L(T.ResizeShortestEdge)(
                short_edge_length=(480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800),
                max_size=1333,
                sample_style="choice",
            ),
        ],
        augmentation_with_crop=[
            L(T.RandomFlip)(),
            L(T.ResizeShortestEdge)(
                short_edge_length=(400, 500, 600),
                sample_style="choice",
            ),
            L(T.RandomCrop)(
                crop_type="absolute_range",
                crop_size=(384, 600),
            ),
            L(T.ResizeShortestEdge)(
                short_edge_length=(480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800),
                max_size=1333,
                sample_style="choice",
            ),
        ],
        is_train=True,
        mask_on=False,
        img_format="RGB",
        categories_names=train_thing_classes,
    ),
    total_batch_size=16,
    num_workers=4,
)

dataloader.test = L(build_detection_test_loader)(
    dataset=L(get_detection_dataset_dicts)(names="Aquarium_odinw_test", filter_empty=False),
    mapper=L(DetrDatasetMapper)(
        augmentation=[
            L(T.ResizeShortestEdge)(
                short_edge_length=800,
                max_size=1333,
            ),
        ],
        augmentation_with_crop=None,
        is_train=False,
        mask_on=False,
        img_format="RGB",
        categories_names=test_thing_classes,
    ),
    num_workers=4,
)

dataloader.evaluator = L(COCOEvaluator)(
    dataset_name="${..test.dataset.names}",
)
