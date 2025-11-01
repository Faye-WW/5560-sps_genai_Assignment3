import io, base64, torch
from fastapi import APIRouter
from torchvision.utils import save_image
from gan.models import Generator, LATENT_DIM

router = APIRouter(prefix="/gan", tags=["gan"])
DEVICE = torch.device("mps" if torch.backends.mps.is_available()
                      else ("cuda" if torch.cuda.is_available() else "cpu"))

G = Generator().to(DEVICE)
try:
    G.load_state_dict(torch.load("artifacts/generator.pt", map_location=DEVICE))
    G.eval()
except Exception:
    G.eval()  

@router.get("/health")
def health():
    return {"status": "ok"}

@router.get("/sample")
def sample(n: int = 8):
    n = max(1, min(n, 16))
    z = torch.randn(n, LATENT_DIM, device=DEVICE)
    with torch.no_grad():
        imgs = (G(z).cpu() + 1) / 2
    buf = io.BytesIO()
    save_image(imgs, buf, nrow=n, format="PNG")
    return {"image_base64": base64.b64encode(buf.getvalue()).decode("utf-8"), "count": n}
