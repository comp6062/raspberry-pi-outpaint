
import os
import io
import logging
from logging.handlers import RotatingFileHandler
from typing import Optional, Tuple

from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from jinja2 import Environment, FileSystemLoader

from PIL import Image
import numpy as np
import uvicorn

# Optional dependencies
HAVE_LAMA = False
try:
    from lama_cleaner.model_manager import ModelManager
    from lama_cleaner.schema import HDStrategy, SDSampler, Config
    HAVE_LAMA = True
except Exception:
    HAVE_LAMA = False

HAVE_ONNX = False
try:
    import onnxruntime as ort  # noqa: F401
    HAVE_ONNX = True
except Exception:
    HAVE_ONNX = False

APP_DIR = os.environ.get("OP_APP_DIR", os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
STATIC_DIR = os.path.join(APP_DIR, "app", "static")
TEMPLATES_DIR = os.path.join(APP_DIR, "app", "templates")
LOG_DIR = os.path.join(APP_DIR, "logs")
TMP_DIR = os.path.join(APP_DIR, "tmp")
MODEL_DIR = os.path.join(APP_DIR, "models")

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(TMP_DIR, exist_ok=True)

# Logging
logger = logging.getLogger("op_app")
logger.setLevel(logging.INFO)
handler = RotatingFileHandler(os.path.join(LOG_DIR, "server.log"), maxBytes=5*1024*1024, backupCount=3)
handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(handler)

app = FastAPI(title="Raspberry Pi Outpainting", docs_url=None, redoc_url=None)

# Static + templates
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
env = Environment(loader=FileSystemLoader(TEMPLATES_DIR))

def clamp_size(img: Image.Image, max_dim: int = 1536) -> Image.Image:
    w, h = img.size
    scale = min(1.0, float(max_dim) / float(max(w, h)))
    if scale < 1.0:
        img = img.resize((int(w*scale), int(h*scale)), Image.LANCZOS)
    return img

def make_border_mask(orig: Image.Image, expanded: Image.Image) -> Image.Image:
    ow, oh = orig.size
    ew, eh = expanded.size
    mask = Image.new("L", (ew, eh), color=255)
    mask.paste(Image.new("L", (ow, oh), color=0), ((ew-ow)//2, (eh-oh)//2))
    return mask

def expand_canvas(img: Image.Image, expand_px: Tuple[int,int,int,int]) -> Image.Image:
    left, right, top, bottom = expand_px
    w, h = img.size
    new_w = w + left + right
    new_h = h + top + bottom
    canvas = Image.new(img.mode, (new_w, new_h))
    canvas.paste(img, (left, top))
    return canvas

def codeformer_face_enhance(img: Image.Image) -> Image.Image:
    # Simplified no-op unless ONNX + model present
    cf_path = os.path.join(MODEL_DIR, "codeformer.onnx")
    if not (HAVE_ONNX and os.path.isfile(cf_path)):
        return img
    # A full face enhancement pipeline is heavy; keep as no-op placeholder for CPU practicality.
    return img

def run_lama_inpaint(img: Image.Image, mask: Image.Image, steps:int, cfg:float, seed:int,
                     sampler:str, denoise:float, photoreal_refine:bool) -> Image.Image:
    if not HAVE_LAMA:
        raise RuntimeError("lama-cleaner not installed. Re-run setup_op.")
    global _MM
    try:
        _MM
    except NameError:
        _MM = ModelManager(name="lama", device="cpu", model_dir=MODEL_DIR)

    hd_strategy = HDStrategy.ORIGINAL
    try:
        sd_sampler = SDSampler[sampler.upper()]
    except Exception:
        sd_sampler = SDSampler.DPMPP_2S_ANCESTRAL

    cfg = float(max(1.0, min(15.0, cfg)))
    denoise = float(max(0.0, min(1.0, denoise)))

    cfg_dict = Config(
        prompt="best quality, realistic, coherent background",
        negative_prompt="lowres, artifacts, blurry, deformed, extra limbs",
        ldm_steps=int(steps),
        ldm_sampler=sd_sampler,
        sd_match_histograms=False,
        hd_strategy=hd_strategy,
        hd_strategy_crop_margin=64,
        hd_strategy_crop_trigger_size=512,
        hd_strategy_resize_limit=1536,
        sd_scale=cfg,
        seed=int(seed),
        cv2_flag="INPAINT_NS",
        sd_mask_blur=4,
        sd_strength=denoise,
        enable_prompt=False,
    )

    np_img = np.array(img.convert("RGB"))
    np_mask = np.array(mask.convert("L"))
    res = _MM.infer(np_img, np_mask, cfg_dict)
    return Image.fromarray(res)

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    template = env.get_template("index.html")
    return HTMLResponse(template.render())

@app.post("/api/outpaint")
async def api_outpaint(
    file: UploadFile = File(...),
    expand_percent: Optional[int] = Form(20),
    expand_px: Optional[str] = Form("0,0,0,0"),
    steps: Optional[int] = Form(20),
    cfg: Optional[float] = Form(4.5),
    seed: Optional[int] = Form(42),
    sampler: Optional[str] = Form("dpmpp_2s_ancestral"),
    denoise: Optional[float] = Form(0.45),
    photoreal_refine: Optional[bool] = Form(False),
    face_preserve: Optional[bool] = Form(False)
):
    try:
        content = await file.read()
        if len(content) > 20 * 1024 * 1024:
            return JSONResponse({"error": "File too large (max 20MB)."}, status_code=400)

        img = Image.open(io.BytesIO(content)).convert("RGB")
        img = clamp_size(img, 1536)

        if expand_px and isinstance(expand_px, str) and expand_px.strip():
            try:
                l, r, t, b = [int(x.strip()) for x in expand_px.split(",")]
                expand = (max(0,l), max(0,r), max(0,t), max(0,b))
            except Exception:
                expand = (0,0,0,0)
        else:
            expand = (0,0,0,0)

        if expand == (0,0,0,0) and expand_percent and int(expand_percent) > 0:
            perc = int(expand_percent) / 100.0
            w, h = img.size
            expand = (int(w*perc/2), int(w*perc/2), int(h*perc/2), int(h*perc/2))

        expanded = expand_canvas(img, expand)
        mask = make_border_mask(img, expanded)

        out = run_lama_inpaint(expanded, mask, steps=int(steps), cfg=float(cfg), seed=int(seed),
                               sampler=str(sampler), denoise=float(denoise), photoreal_refine=bool(photoreal_refine))

        if face_preserve:
            out = codeformer_face_enhance(out)

        bio = io.BytesIO()
        out.save(bio, format="PNG")
        bio.seek(0)
        return StreamingResponse(bio, media_type="image/png")
    except Exception as e:
        logger.exception("Outpaint failed")
        return JSONResponse({"error": str(e)}, status_code=500)

if __name__ == "__main__":
    host = os.environ.get("OP_HOST", "127.0.0.1")
    port = int(os.environ.get("OP_PORT", "8080"))
    uvicorn.run(app, host=host, port=port, log_level="info")
