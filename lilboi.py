import torch

d = torch.load("weights/lite-mono-small-640x192/encoder.pth", map_location="cpu")
print(type(d))
if isinstance(d, dict):
    print(next(iter(d.keys())))
    # If it has 'state_dict', print its keys:
    if "state_dict" in d:
        print("state_dict keys example:", list(d["state_dict"].keys())[:20])
    else:
        print("top-level keys example:", list(d.keys())[:20])
