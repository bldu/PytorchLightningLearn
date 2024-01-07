import torch
from official_train_test import LitAutoEncoder, encoder, decoder

model=LitAutoEncoder.load_from_checkpoint("./lightning_logs/version_9/checkpoints/epoch=1-step=400.ckpt", encoder=encoder, decoder=decoder)
model.eval()
inp = torch.randn(1,1,28,28)
with torch.no_grad():
    out = model(inp)
print(out.shape)
