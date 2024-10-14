#!/bin/bash
export TRANSFORMERS_OFFLINE=1
python train_multidatasets_old.py --config-file test_odinw13_softfreeze --model-config-file groundingdino/config/GroundingDINO_SwinT_OGC_rep.py --model-checkpoint-path groundingdino_swint_ogc.pth --num-gpus 2 --seed 42 --output-dir output/odinw13 --shuffle-tasks --eval-only
python train_multidatasets.py --config-file test_odinw13_softfreeze --model-config-file groundingdino/config/GroundingDINO_SwinT_OGC_rep.py --model-checkpoint-path output/odinw13/model_final.pth --num-gpus 2 --seed 42 --output-dir output/odinw13 --shuffle-tasks --eval-only