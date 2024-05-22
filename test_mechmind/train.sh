python train_net.py --config-file test_mechmind/test_mechmind_0_5.py --model-config-file groundingdino/config/GroundingDINO_SwinT_OGC_dt.py --model-checkpoint-path groundingdino_swint_ogc.pth --num-gpus 1

python train_net.py --config-file test_mechmind/test_mechmind_5_10.py --model-config-file groundingdino/config/GroundingDINO_SwinT_OGC_dt.py --model-checkpoint-path output/mechmind/mechmind_0_5_text_adapter_prompt_memory/model_final.pth --num-gpus 1

python train_net.py --config-file test_mechmind/test_mechmind_10_15.py --model-config-file groundingdino/config/GroundingDINO_SwinT_OGC_dt.py --model-checkpoint-path output/mechmind/mechmind_5_10_text_adapter_prompt_memory/model_final.pth --num-gpus 1

python train_net.py --config-file test_mechmind/test_mechmind2coco.py --model-config-file groundingdino/config/GroundingDINO_SwinT_OGC_dt.py --model-checkpoint-path output/mechmind/mechmind_10_15_text_adapter_prompt_memory/model_final.pth --num-gpus 1 --eval-only
