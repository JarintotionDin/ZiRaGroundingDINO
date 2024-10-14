import torch

# load state dict
model_checkpoint_path = "output/odinw13/model_final.pth"
checkpoint = torch.load(model_checkpoint_path, map_location="cpu")

for key in checkpoint["model"].keys():
    if "adapter" in key:
        print(key)
    

a = {"a": 1, "b": 2, "c": 3}

str_ = "ap result: {}".format(a)
print(str_)
