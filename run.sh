python demo/groundingdino_detector.py \
  -c "/home/jarin/projects/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC_dt.py" \
  -p "output/odinw13/model_final.pth" \
  -i ".asset/rgb_0.png"\
  -o "outputs/0" \
  -t "individual objects." \
  --box_threshold 0.2
# -t "a pipe.banana.apple.green_box.black_sponge.gear.angle_aluminum.cup"
# -t "banana.apple.yellow_box.green_box.gear.pipe.pear.box.knob.cup.washer.grape.carambola"
# -p "/home/jarin/projects/GroundingDINO/groundingdino_swint_ogc.pth" \
# -p "output/mechmind/mechmind2coco_language_adapter/model_final.pth" \
# -p "output/mechmind_hard/model_final.pth" \
# -p "output/mechmind/mechmind2coco_language_adapter/model_final.pth" \
# -p "/home/jarin/projects/GroundingDINO/groundingdino_swint_ogc.pth" \
# /home/jarin/projects/interactron-main/data/interactron/test/FloorPlan26_00003/pos=[-1.75,0.90,3.25]_rot=[90deg].jpg
# /home/jarin/projects/interactron-main/data/interactron/test/FloorPlan27_00001/pos=[0.97,0.90,1.38]_rot=[240deg].jpg
# /home/jarin/projects/interactron-main/data/interactron/test/FloorPlan26_00003/pos=[-1.75,0.90,3.25]_rot=[90deg].jpg
# -t "objects."