# cnnmacet.py
import os, cv2, torch, torch.nn as nn, torch.nn.functional as F
from torchvision import transforms
from PIL import Image

MODEL_PATH = os.environ.get("CONGESTION_MODEL", "best_congestion_classifier.pt")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

_model = None
def _load_model():
    global _model
    if _model is not None:
        return _model
    # 1) Coba TorchScript
    try:
        m = torch.jit.load(MODEL_PATH, map_location=DEVICE).to(DEVICE).eval()
        _model = m
        return _model
    except Exception:
        pass
    # 2) Fallback: checkpoint biasa -> perlu arsitektur
    from efficientnet_pytorch import EfficientNet
    m = EfficientNet.from_name("efficientnet-b0")
    m._fc = nn.Linear(m._fc.in_features, 1)
    ckpt = torch.load(MODEL_PATH, map_location="cpu")
    state_dict = ckpt.get("model_state_dict", ckpt)  # dukung dua format
    m.load_state_dict(state_dict, strict=False)
    _model = m.to(DEVICE).eval()
    return _model

@torch.no_grad()
def check_macet_cnn(frame_bgr):
    if frame_bgr is None or frame_bgr.size == 0:
        return False
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    x = _transform(pil).unsqueeze(0).to(DEVICE)
    out = _load_model()(x)
    if isinstance(out, (tuple, list)): out = out[0]
    if out.ndim == 2 and out.shape[1] == 1:
        prob = torch.sigmoid(out[:, 0])
    elif out.ndim == 2 and out.shape[1] == 2:
        prob = F.softmax(out, dim=1)[:, 1]
    else:
        prob = torch.sigmoid(out.reshape(1))
    return float(prob.item()) >= 0.5