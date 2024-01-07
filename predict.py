import torch
from official_train_test import LitAutoEncoder, encoder, decoder

device = "cuda" if torch.cuda.is_available() else "cpu"
auto_encoder=LitAutoEncoder.load_from_checkpoint("./lightning_logs/version_9/checkpoints/epoch=1-step=400.ckpt", encoder=encoder, decoder=decoder)
auto_encoder.eval()
inp = torch.randn(1,1,28,28).to(device)
with torch.no_grad():
    #out = auto_encoder.decoder(auto_encoder.encoder(inp.view(inp.size(0), -1)))
    out = auto_encoder(inp.view(inp.size(0), -1))
print(out.shape)
