python train_net.py --config-file test_voc/test_voc_0_10.py --model-config-file groundingdino/config/GroundingDINO_SwinT_OGC_dt.py --model-checkpoint-path groundingdino_swint_ogc.pth --num-gpus 1

python train_net.py --config-file test_voc/test_voc_0_10_10_20.py --model-config-file groundingdino/config/GroundingDINO_SwinT_OGC_dt.py --model-checkpoint-path output/voc/voc_text_adapter_0_10_prompt_memory/model_final.pth --num-gpus 1

python train_net.py --config-file test_voc/test_voc2coco.py --model-config-file groundingdino/config/GroundingDINO_SwinT_OGC_dt.py --model-checkpoint-path output/voc/voc_text_adapter_0_10_10_20_prompt_memory/model_final.pth --num-gpus 1 --eval-only
