import torch

from nca_ae.nca import CAModel

CHANS = 16
FIRE_RATE = 0.5
device = "cuda"
HIDDEN_SIZE = 128

if __name__ == "__main__":
    model = CAModel(CHANS, FIRE_RATE, device=device, hidden_size=HIDDEN_SIZE)

    # Sanity check
    inp = torch.rand(10, 32, 32, 16).to(device)
    out = model(inp, steps=15)
    print("Forward sanity check done")
