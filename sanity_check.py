# sanity_check.py
import torch
from gan.models import Generator, Discriminator, LATENT_DIM

def check_generator():
    G = Generator()
    z = torch.randn(4, LATENT_DIM)               # (B, 100)
    out = G(z)                                   # expect (B,1,28,28)
    assert out.shape == (4, 1, 28, 28), f"G out shape {out.shape} != (4,1,28,28)"
    # range check after Tanh
    assert torch.all(out <= 1.0) and torch.all(out >= -1.0), "G output not in [-1,1]"
    print("[PASS] Generator: (B,100) -> (B,1,28,28) with Tanh")

def check_discriminator():
    D = Discriminator()
    x = torch.randn(4, 1, 28, 28)                # (B,1,28,28)
    logit = D(x)                                 # expect (B,)
    assert logit.shape == (4,), f"D out shape {logit.shape} != (4,)"
    print("[PASS] Discriminator: (B,1,28,28) -> (B,) logit")

if __name__ == "__main__":
    check_generator()
    check_discriminator()
    print("All architecture checks passed.")
