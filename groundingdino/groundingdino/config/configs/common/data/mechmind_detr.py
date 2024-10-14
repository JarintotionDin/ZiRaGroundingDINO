from omegaconf import OmegaConf

import detectron2.data.transforms as T
from detectron2.config import LazyCall as L
from detectron2.data import (
    build_detection_test_loader,
    build_detection_train_loader,
    get_detection_dataset_dicts,
)
from detectron2.evaluation import COCOEvaluator
from groundingdino.datasets import DetrDatasetMapper

from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances


# register_coco_instances("mechmind_easy", {},
#                         "datasets/mechmind_easy/mechmind_easy.json",
#                         "datasets/mechmind_easy")
# MetadataCatalog.get("mechmind_easy")
# DatasetCatalog.get("mechmind_easy")


# register_coco_instances("mechmind_easy_0_5", {},
#                         "datasets/mechmind_easy/mechmind_easy_0_5.json",
#                         "datasets/mechmind_easy")
# MetadataCatalog.get("mechmind_easy_0_5")
# DatasetCatalog.get("mechmind_easy_0_5")


# register_coco_instances("mechmind_easy_5_10", {},
#                         "datasets/mechmind_easy/mechmind_easy_5_10.json",
#                         "datasets/mechmind_easy")
# MetadataCatalog.get("mechmind_easy_5_10")
# DatasetCatalog.get("mechmind_easy_5_10")


# register_coco_instances("mechmind_easy_10_15", {},
#                         "datasets/mechmind_easy/mechmind_easy_5_10.json",
#                         "datasets/mechmind_easy")
# MetadataCatalog.get("mechmind_easy_10_15")
# DatasetCatalog.get("mechmind_easy_10_15")


# register_coco_instances("mechmind_hard", {},
#                         "datasets/mechmind_hard/mechmind_hard.json",
#                         "datasets/mechmind_hard")
meta_data = MetadataCatalog.get("mechmind_hard")
dataset_data = DatasetCatalog.get("mechmind_hard")
classes = meta_data.thing_classes
classes = [name for name in classes if not (name == "")]

dataloader = OmegaConf.create()

dataloader.train = L(build_detection_train_loader)(
    dataset=L(get_detection_dataset_dicts)(names="mechmind_easy"),
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
        categories_names=classes,
    ),
    total_batch_size=16,
    num_workers=4,
)

dataloader.test = L(build_detection_test_loader)(
    dataset=L(get_detection_dataset_dicts)(names="mechmind_hard"),
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
        categories_names=classes,
    ),
    num_workers=4,
)

dataloader.evaluator = L(COCOEvaluator)(
    dataset_name="${..test.dataset.names}",
)
