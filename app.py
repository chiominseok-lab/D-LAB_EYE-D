"""
안저 이미지 분류 앱 (EfficientNet-B0 + Grad-CAM)
사용법: python app.py
EXE 빌드: pyinstaller --onefile --windowed app.py
"""

import sys, os, io, base64
import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.ops import StochasticDepth
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as mpl_cm
import matplotlib.font_manager as fm
import webview

# ── 한국어 폰트 ───────────────────────────────────────────────────────────────
def _set_korean_font():
    candidates = ["Malgun Gothic", "Apple SD Gothic Neo", "NanumGothic", "Nanum Gothic", "DejaVu Sans"]
    available = {f.name for f in fm.fontManager.ttflist}
    for font in candidates:
        if font in available:
            matplotlib.rc("font", family=font)
            break
    matplotlib.rcParams["axes.unicode_minus"] = False

_set_korean_font()

# ──────────────────────────────────────────────────────────────────────────────
# 모델 구조 (app_1.py 원본 유지)
# ──────────────────────────────────────────────────────────────────────────────

class SEBlock(nn.Module):
    def __init__(self, in_channels, reduced_dim):
        super().__init__()
        self.avgpool          = nn.AdaptiveAvgPool2d(1)
        self.fc1              = nn.Conv2d(in_channels, reduced_dim, kernel_size=1)
        self.activation       = nn.SiLU()
        self.fc2              = nn.Conv2d(reduced_dim, in_channels, kernel_size=1)
        self.scale_activation = nn.Sigmoid()

    def forward(self, x):
        scale = self.avgpool(x)
        scale = self.fc1(scale)
        scale = self.activation(scale)
        scale = self.fc2(scale)
        scale = self.scale_activation(scale)
        return x * scale


class MBConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride, expand_ratio, se_ratio=0.25):
        super().__init__()
        self.use_residual = (in_channels == out_channels) and (stride == 1)
        expanded_ch  = in_channels * expand_ratio
        reduced_dim  = max(1, int(in_channels * se_ratio))

        block_layers = []
        if expand_ratio != 1:
            block_layers.append(nn.Sequential(
                nn.Conv2d(in_channels, expanded_ch, 1, bias=False),
                nn.BatchNorm2d(expanded_ch),
                nn.SiLU()
            ))
            block_layers.append(nn.Sequential(
                nn.Conv2d(expanded_ch, expanded_ch, kernel_size,
                          stride=stride, padding=kernel_size // 2,
                          groups=expanded_ch, bias=False),
                nn.BatchNorm2d(expanded_ch),
                nn.SiLU()
            ))
            block_layers.append(SEBlock(expanded_ch, reduced_dim))
            block_layers.append(nn.Sequential(
                nn.Conv2d(expanded_ch, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels)
            ))
        else:
            block_layers.append(nn.Sequential(
                nn.Conv2d(expanded_ch, expanded_ch, kernel_size,
                          stride=stride, padding=kernel_size // 2,
                          groups=expanded_ch, bias=False),
                nn.BatchNorm2d(expanded_ch),
                nn.SiLU()
            ))
            block_layers.append(SEBlock(expanded_ch, reduced_dim))
            block_layers.append(nn.Sequential(
                nn.Conv2d(expanded_ch, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels)
            ))

        self.block            = nn.Sequential(*block_layers)
        self.stochastic_depth = StochasticDepth(p=0.2, mode='row')

    def forward(self, x):
        out = self.block(x)
        if self.use_residual:
            out = self.stochastic_depth(out)
            out = out + x
        return out


class EfficientNetB0(nn.Module):
    MB_CONFIG = [
        (1,  16, 1, 1, 3),
        (6,  24, 2, 2, 3),
        (6,  40, 2, 2, 5),
        (6,  80, 3, 2, 3),
        (6, 112, 3, 1, 5),
        (6, 192, 4, 2, 5),
        (6, 320, 1, 1, 3),
    ]

    def __init__(self, num_classes=3):
        super().__init__()
        stages = [
            nn.Sequential(
                nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.SiLU()
            )
        ]
        in_ch = 32
        for expand_ratio, out_ch, repeats, stride, kernel_size in self.MB_CONFIG:
            stage = []
            for i in range(repeats):
                stage.append(MBConv(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=kernel_size,
                    stride=stride if i == 0 else 1,
                    expand_ratio=expand_ratio
                ))
                in_ch = out_ch
            stages.append(nn.Sequential(*stage))

        stages.append(nn.Sequential(
            nn.Conv2d(320, 1280, 1, bias=False),
            nn.BatchNorm2d(1280),
            nn.SiLU()
        ))
        self.features   = nn.Sequential(*stages)
        self.avgpool    = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.6),
            nn.Linear(1280, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x


# ──────────────────────────────────────────────────────────────────────────────
# Grad-CAM
# ──────────────────────────────────────────────────────────────────────────────

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self._fwd_hook)
        target_layer.register_full_backward_hook(self._bwd_hook)

    def _fwd_hook(self, module, inp, out):
        self.activations = out.detach()

    def _bwd_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def generate(self, input_tensor, target_class):
        self.model.zero_grad()
        output = self.model(input_tensor)
        score = output[0, target_class]
        score.backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam).squeeze().cpu().numpy()
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()
        return cam, output.detach()


# ──────────────────────────────────────────────────────────────────────────────
# 전처리 (app_1.py 원본 유지)
# ──────────────────────────────────────────────────────────────────────────────

TARGET_SIZE = 224
MEAN = np.array([0.485, 0.456, 0.406])
STD  = np.array([0.229, 0.224, 0.225])

val_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=MEAN.tolist(), std=STD.tolist())
])


def apply_clahe(img_rgb_uint8):
    bgr = cv2.cvtColor(img_rgb_uint8, cv2.COLOR_RGB2BGR)
    b, g, r = cv2.split(bgr)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    r_clahe = clahe.apply(r)
    result = cv2.merge([r_clahe, r_clahe, r_clahe])
    return cv2.cvtColor(result, cv2.COLOR_BGR2RGB)


def preprocess_image(pil_image):
    img = pil_image.convert("RGB")
    img_np = np.array(img)
    h, w = img_np.shape[:2]

    mid = w // 2
    left_eye  = img_np[:, :mid, :]
    right_eye = img_np[:, mid:, :]

    left_r  = cv2.resize(left_eye,  (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_LANCZOS4)
    right_r = cv2.resize(right_eye, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_LANCZOS4)

    merged = np.zeros((TARGET_SIZE, TARGET_SIZE * 2, 3), dtype=np.uint8)
    merged[:, :TARGET_SIZE, :]               = left_r
    merged[:, TARGET_SIZE:TARGET_SIZE * 2, :] = right_r

    clahe_img = apply_clahe(merged)

    left_tensor  = val_transform(clahe_img[:, :TARGET_SIZE, :]).unsqueeze(0)
    right_tensor = val_transform(clahe_img[:, TARGET_SIZE:, :]).unsqueeze(0)

    return left_tensor, right_tensor, merged


# ──────────────────────────────────────────────────────────────────────────────
# 추론 + Grad-CAM (app_1.py 원본 유지)
# ──────────────────────────────────────────────────────────────────────────────

CLASS_NAMES  = ["Normal", "NAION", "ON"]
CLASS_COLORS = ["#22c55e", "#f87171", "#fbbf24"]


def run_inference(models, left_tensor, right_tensor, device):
    all_probs = []
    for model in models:
        model.eval()
        with torch.no_grad():
            l_prob = torch.softmax(model(left_tensor.to(device)),  dim=1)
            r_prob = torch.softmax(model(right_tensor.to(device)), dim=1)
            avg_out = (l_prob + r_prob) / 2.0
        all_probs.append(avg_out.cpu())

    final_probs = torch.stack(all_probs, dim=0).mean(dim=0)
    probs = final_probs.squeeze().numpy()
    pred_class = int(np.argmax(probs))
    return probs, pred_class


def make_gradcam_figure(models, left_tensor, right_tensor, merged_img, pred_class, probs, device):
    all_cams_left  = []
    all_cams_right = []

    for model in models:
        model.eval()
        target_layer = model.features[-1]
        gc = GradCAM(model, target_layer)

        lt = left_tensor.clone().to(device)
        lt.requires_grad_(True)
        cam_l, _ = gc.generate(lt, pred_class)
        all_cams_left.append(cam_l)

        rt = right_tensor.clone().to(device)
        rt.requires_grad_(True)
        cam_r, _ = gc.generate(rt, pred_class)
        all_cams_right.append(cam_r)

    avg_cam_l = np.mean(all_cams_left, axis=0)
    avg_cam_r = np.mean(all_cams_right, axis=0)

    def overlay(eye_img, cam):
        h, w = eye_img.shape[:2]
        cam_up = np.array(Image.fromarray((cam * 255).astype(np.uint8))
                          .resize((w, h), Image.BILINEAR)) / 255.0
        heatmap = mpl_cm.jet(cam_up)[:, :, :3]
        result  = 0.5 * eye_img / 255.0 + 0.5 * heatmap
        return np.clip(result, 0, 1)

    left_img  = merged_img[:, :TARGET_SIZE, :]
    right_img = merged_img[:, TARGET_SIZE:, :]

    overlay_l = overlay(left_img,  avg_cam_l)
    overlay_r = overlay(right_img, avg_cam_r)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.patch.set_facecolor("#0f1117")

    titles = ["원본 (좌안 + 우안)", "좌안 Grad-CAM", "우안 Grad-CAM"]
    images = [merged_img / 255.0, overlay_l, overlay_r]

    for ax, title, im in zip(axes, titles, images):
        ax.imshow(im)
        ax.set_title(title, color="white", fontsize=11, pad=6)
        ax.axis("off")
        ax.set_facecolor("#0f1117")

    fig.suptitle(
        "  |  ".join(f"{CLASS_NAMES[i]}: {probs[i]*100:.1f}%" for i in range(3)),
        color="white", fontsize=11, y=0.02
    )
    plt.tight_layout(rect=[0, 0.06, 1, 1])

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight", facecolor="#0f1117")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def get_exe_dir():
    if getattr(sys, "frozen", False):
        return sys._MEIPASS  # PyInstaller 번들 내부 경로 (가중치 파일 위치)
    return os.path.dirname(os.path.abspath(__file__))


# ──────────────────────────────────────────────────────────────────────────────
# API (pywebview ↔ Python 브릿지)
# ──────────────────────────────────────────────────────────────────────────────

class API:
    def __init__(self):
        self.models   = []
        self.device   = torch.device("cpu")
        self.img_path = None
        self._lt = self._rt = self._merged = None
        self._probs = self._pred = None

    def auto_load(self):
        exe = get_exe_dir()
        for d in [exe,
                  os.path.join(exe, "model"),
                  os.path.dirname(exe),
                  os.path.join(os.path.dirname(exe), "model")]:
            if not os.path.isdir(d):
                continue
            pths = sorted(f for f in os.listdir(d) if f.endswith(".pth"))
            if pths:
                return self.load_from_dir(d)
        return {"ok": False, "msg": "자동 탐색 실패 — 폴더를 수동 선택하세요"}

    def select_folder(self):
        dirs = webview.windows[0].create_file_dialog(webview.FOLDER_DIALOG)
        if not dirs:
            return {"ok": False}
        return self.load_from_dir(dirs[0])

    def load_from_dir(self, d):
        pths = sorted(f for f in os.listdir(d) if f.endswith(".pth"))
        if not pths:
            return {"ok": False, "msg": f"{d} 에 .pth 없음"}
        self.models = []
        try:
            for f in pths:
                model = EfficientNetB0(num_classes=3).to(self.device)
                state = torch.load(os.path.join(d, f), map_location=self.device)
                model.load_state_dict(state)
                model.eval()
                self.models.append(model)
            return {"ok": True, "count": len(self.models)}
        except Exception as e:
            return {"ok": False, "msg": str(e)}

    def open_image(self):
        try:
            files = webview.windows[0].create_file_dialog(
                webview.OPEN_DIALOG,
                file_types=("Image Files (*.jpg;*.jpeg;*.png;*.bmp;*.tif;*.tiff)", "All files (*.*)"))
        except Exception as e:
            return {"ok": False, "msg": str(e)}
        if not files:
            return {"ok": False}

        try:
            self.img_path = files[0]
            pil = Image.open(self.img_path)
            pil.thumbnail((600, 200))
            buf = io.BytesIO()
            pil.save(buf, format="PNG")
            preview = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
            w, h = Image.open(self.img_path).size
            return {"ok": True, "name": os.path.basename(self.img_path),
                    "meta": f"{w} × {h} px", "preview": preview}
        except Exception as e:
            return {"ok": False, "msg": str(e)}

    def run(self, doctor):
        if not self.models:
            return {"ok": False, "msg": "모델 미로드"}
        if not self.img_path:
            return {"ok": False, "msg": "이미지 미선택"}
        try:
            pil = Image.open(self.img_path)
            self._lt, self._rt, self._merged = preprocess_image(pil)
            probs, pred = run_inference(self.models, self._lt, self._rt, self.device)
            self._probs, self._pred = probs, pred

            # fold별 개별 예측
            per_fold = []
            for model in self.models:
                model.eval()
                with torch.no_grad():
                    l_p = torch.softmax(model(self._lt.to(self.device)), dim=1)
                    r_p = torch.softmax(model(self._rt.to(self.device)), dim=1)
                    p   = ((l_p + r_p) / 2.0).squeeze().cpu().numpy()
                per_fold.append(int(np.argmax(p)))

            n_agree = sum(1 for f in per_fold if f == pred)
            max_p   = float(probs[pred])

            if max_p >= 0.75 and n_agree >= len(self.models):
                conf, conf_level = "신뢰도: 높음", "high"
            elif max_p >= 0.55 and n_agree >= len(self.models) - 1:
                conf, conf_level = "신뢰도: 중간", "mid"
            else:
                conf, conf_level = "신뢰도: 낮음 — 추가 검사 권장", "low"

            second = sorted(range(3), key=lambda i: probs[i], reverse=True)[1]
            diff   = float(probs[pred] - probs[second])
            warn   = ""
            if diff < 0.15 or n_agree < len(self.models) - 1:
                warn = (f"판독 불확실: {CLASS_NAMES[pred]}과 {CLASS_NAMES[second]}의 "
                        f"확률 차이가 {diff*100:.1f}%로 근소합니다. 임상 소견과 함께 판단하십시오.")

            fname = os.path.basename(self.img_path)
            copy_text = "\n".join([
                "[AI 보조 판독]",
                f"파일: {fname}",
                f"담당의: {doctor or '(미입력)'}",
                "",
                f"AI 예측: {CLASS_NAMES[pred]}",
                f"  Normal {probs[0]*100:.1f}%  /  NAION {probs[1]*100:.1f}%  /  ON {probs[2]*100:.1f}%",
                "",
                "※ 최종 판단은 담당의에 따름",
            ])

            return {
                "ok": True,
                "pred": pred,
                "pred_name": CLASS_NAMES[pred],
                "pred_color": CLASS_COLORS[pred],
                "probs": [float(p) for p in probs],
                "conf": conf,
                "conf_level": conf_level,
                "warn": warn,
                "copy_text": copy_text,
            }
        except Exception as e:
            return {"ok": False, "msg": str(e)}

    def run_gradcam(self):
        if self._probs is None:
            return {"ok": False}
        try:
            img_b64 = make_gradcam_figure(
                self.models, self._lt, self._rt, self._merged,
                self._pred, self._probs, self.device)
            return {"ok": True, "img": img_b64}
        except Exception as e:
            return {"ok": False, "msg": str(e)}


# ──────────────────────────────────────────────────────────────────────────────
# HTML UI (app.py 원본 유지)
# ──────────────────────────────────────────────────────────────────────────────

HTML = """<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>안저 이미지 분류기</title>
<style>
  *{box-sizing:border-box;margin:0;padding:0}
  :root{
    --bg:#0f1117;--panel:#161b27;--card:#1e2536;--border:#2a3352;
    --text:#e2e8f0;--muted:#8892a4;--dim:#4a5568;
    --blue:#3b82f6;--blue-d:#1d4ed8;--green:#22c55e;
    --coral:#f87171;--amber:#fbbf24;--teal:#2dd4bf;
    --radius:10px;--radius-sm:6px;
  }
  body{background:var(--bg);color:var(--text);font-family:'Malgun Gothic','Apple SD Gothic Neo',sans-serif;font-size:13px;height:100vh;display:flex;flex-direction:column;overflow:hidden}

  /* topbar */
  .topbar{background:#0d1120;padding:10px 20px;display:flex;align-items:center;gap:12px;border-bottom:1px solid var(--border);flex-shrink:0}
  .topbar-title{font-size:16px;font-weight:700;color:#fff}
  .topbar-sub{font-size:11px;color:var(--muted)}
  .topbar-spacer{flex:1}
  .topbar input{background:var(--card);border:1px solid var(--border);color:var(--text);padding:5px 10px;border-radius:var(--radius-sm);font-size:12px;font-family:inherit;width:320px}
  .topbar input:focus{outline:none;border-color:var(--blue)}
  .btn{padding:6px 14px;border-radius:var(--radius-sm);border:none;font-family:inherit;font-size:12px;font-weight:600;cursor:pointer;transition:opacity .15s}
  .btn:hover{opacity:.85}
  .btn-purple{background:#7c3aed;color:#fff}
  .btn-dim{background:var(--border);color:var(--text)}
  .model-status{font-size:11px;color:var(--dim);display:flex;align-items:center;gap:5px}
  .dot{width:8px;height:8px;border-radius:50%;background:var(--dim)}
  .dot.ok{background:var(--green)}

  /* main */
  .main{display:grid;grid-template-columns:320px 1fr;gap:10px;padding:10px;flex:1;overflow:hidden;min-height:0}
  .col{display:flex;flex-direction:column;gap:10px;overflow-y:auto;min-height:0}
  .col::-webkit-scrollbar{width:4px}
  .col::-webkit-scrollbar-thumb{background:var(--border);border-radius:2px}

  /* card */
  .card{background:var(--card);border:1px solid var(--border);border-radius:var(--radius);padding:14px;flex-shrink:0}
  .card-grow{flex:1;min-height:0;display:flex;flex-direction:column}
  .sec-label{font-size:10px;font-weight:700;color:var(--muted);letter-spacing:.06em;margin-bottom:10px;text-transform:uppercase}

  /* image upload */
  .drop-zone{border:1.5px dashed var(--border);border-radius:var(--radius-sm);background:var(--bg);padding:16px;text-align:center;cursor:pointer;transition:border-color .15s;margin-bottom:8px}
  .drop-zone:hover{border-color:var(--blue)}
  .drop-zone .icon{font-size:28px;margin-bottom:6px}
  .drop-zone .hint{font-size:11px;color:var(--muted)}
  .file-info{background:var(--bg);border:1px solid var(--border);border-radius:var(--radius-sm);padding:8px 10px;display:flex;align-items:center;gap:8px;margin-bottom:8px;display:none}
  .file-info .fname{font-size:12px;font-weight:600;flex:1;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
  .file-info .fmeta{font-size:10px;color:var(--muted)}
  .preview-img{width:100%;border-radius:var(--radius-sm);margin-top:6px;display:none}

  /* run btn */
  .btn-run{width:100%;padding:12px;background:#166534;color:#fff;border:none;border-radius:var(--radius-sm);font-family:inherit;font-size:13px;font-weight:700;cursor:pointer;transition:background .15s;display:flex;align-items:center;justify-content:center;gap:8px}
  .btn-run:hover:not(:disabled){background:#15803d}
  .btn-run:disabled{background:var(--dim);cursor:not-allowed;opacity:.6}

  /* result */
  .result-header{display:flex;align-items:center;gap:12px;margin-bottom:10px}
  .result-name{font-size:28px;font-weight:700;color:var(--muted)}
  .conf-badge{font-size:11px;padding:3px 8px;border-radius:var(--radius-sm)}
  .conf-high{background:#052e16;color:#86efac}
  .conf-mid{background:#2d1a00;color:#fde68a}
  .conf-low{background:#2d0a0a;color:#fca5a5}

  /* warn */
  .warn-box{background:#2d1a00;border:1px solid #92400e;border-radius:var(--radius-sm);padding:8px 10px;font-size:12px;color:#fde68a;margin-bottom:10px;display:none;line-height:1.5}

  /* prob bars */
  .prob-row{display:flex;align-items:center;gap:8px;margin-bottom:6px}
  .prob-label{font-size:12px;color:var(--muted);width:52px;flex-shrink:0;font-family:monospace}
  .bar-bg{flex:1;height:10px;background:var(--bg);border-radius:5px;overflow:hidden}
  .bar-fill{height:100%;border-radius:5px;transition:width .4s ease}
  .prob-pct{font-size:12px;font-weight:600;width:40px;text-align:right;font-family:monospace}

  /* divider */
  .divider{border:none;border-top:1px solid var(--border);margin:10px 0}

  /* gradcam */
  .gradcam-wrap{flex:1;min-height:0;display:flex;align-items:center;justify-content:center}
  .gradcam-wrap img{max-width:100%;max-height:100%;border-radius:var(--radius-sm)}
  .gradcam-placeholder{color:var(--dim);font-size:12px;text-align:center}
  .cam-note{font-size:11px;color:var(--teal);margin-top:8px;flex-shrink:0}

  /* copy text */
  .copy-text{background:var(--bg);border:1px solid var(--border);border-radius:var(--radius-sm);padding:10px;font-size:11px;color:var(--muted);font-family:monospace;line-height:1.8;margin-bottom:8px;min-height:120px;white-space:pre-wrap}
  .btn-copy{width:100%;padding:7px;background:var(--border);color:var(--text);border:none;border-radius:var(--radius-sm);font-family:inherit;font-size:12px;font-weight:600;cursor:pointer;margin-bottom:5px}
  .btn-copy:hover{background:var(--dim)}
  .btn-save{width:100%;padding:7px;background:var(--blue-d);color:#fff;border:none;border-radius:var(--radius-sm);font-family:inherit;font-size:12px;font-weight:600;cursor:pointer}
  .btn-save:hover:not(:disabled){background:var(--blue)}
  .btn-save:disabled{opacity:.4;cursor:not-allowed}

  /* disclaimer */
  .disclaimer{font-size:11px;color:var(--dim);line-height:1.7}
  .disclaimer-title{font-size:12px;font-weight:700;color:var(--amber);margin-bottom:5px}

  /* status */
  .statusbar{background:#0d1120;padding:4px 16px;font-size:11px;color:var(--muted);border-top:1px solid var(--border);flex-shrink:0}

  /* spinner */
  .spinner{display:none;width:16px;height:16px;border:2px solid rgba(255,255,255,.3);border-top-color:#fff;border-radius:50%;animation:spin .7s linear infinite}
  @keyframes spin{to{transform:rotate(360deg)}}

  /* toast */
  .toast{position:fixed;bottom:40px;left:50%;transform:translateX(-50%);background:#1d4ed8;color:#fff;padding:8px 20px;border-radius:20px;font-size:12px;font-weight:600;opacity:0;transition:opacity .3s;pointer-events:none}
  .toast.show{opacity:1}
</style>
</head>
<body>

<div class="topbar">
  <span class="topbar-title">안저 이미지 분류기</span>
  <span class="topbar-sub">EfficientNet-B0 · 3-fold ensemble</span>
  <div class="topbar-spacer"></div>
  <input type="text" id="weight-path" placeholder="fold1~fold3_best.pth 폴더를 선택하세요" readonly>
  <button class="btn btn-dim" onclick="selectFolder()">폴더 선택</button>
  <button class="btn btn-purple" onclick="loadModels()">모델 로드</button>
  <div class="model-status">
    <div class="dot" id="model-dot"></div>
    <span id="model-status-text">모델 미로드</span>
  </div>
  <span style="color:var(--dim);margin:0 8px">|</span>
  <span style="font-size:11px;color:var(--muted)">담당의</span>
  <input type="text" id="doctor" placeholder="성명" style="width:90px">
</div>

<div class="main">
  <!-- 왼쪽 -->
  <div class="col">
    <div class="card">
      <div class="sec-label">이미지 업로드</div>
      <div class="drop-zone" onclick="openImage()">
        <div class="icon">🖼</div>
        <div style="font-size:12px;color:var(--muted);margin-bottom:3px">이미지를 클릭하여 선택</div>
        <div class="hint">jpg / png · 좌안+우안 합성 이미지</div>
      </div>
      <div class="file-info" id="file-info">
        <span style="color:var(--green);font-size:16px">✓</span>
        <div style="flex:1;min-width:0">
          <div class="fname" id="file-name"></div>
          <div class="fmeta" id="file-meta"></div>
        </div>
      </div>
      <img class="preview-img" id="preview-img">
    </div>

    <div class="card">
      <button class="btn-run" id="run-btn" disabled onclick="runInference()">
        <span id="run-spinner" class="spinner"></span>
        <span id="run-text">🔍 분류 + Grad-CAM 실행</span>
      </button>
    </div>

    <div class="card" style="flex:1">
      <div class="sec-label">결과 텍스트 복사</div>
      <div class="copy-text" id="copy-text">분류 실행 후 결과가 표시됩니다.</div>
      <button class="btn-copy" onclick="copyText()">클립보드 복사</button>
      <button class="btn-save" id="save-btn" disabled onclick="saveResult()">PNG 결과 저장</button>
    </div>

    <div class="card">
      <div class="disclaimer-title">⚠ 면책 고지</div>
      <div class="disclaimer">
        본 결과는 AI 보조 분석 결과이며 의사의 최종 임상 판단을 대체하지 않습니다.
        확진을 위해 OCT, 시야 검사 등 추가 검사가 필요할 수 있습니다.
      </div>
    </div>
  </div>

  <!-- 오른쪽 -->
  <div class="col">
    <div class="card">
      <div class="sec-label">분류 결과</div>
      <div class="result-header">
        <div class="result-name" id="result-name">—</div>
        <div class="conf-badge conf-mid" id="conf-badge" style="display:none"></div>
      </div>
      <div class="warn-box" id="warn-box"></div>
      <div class="prob-row">
        <span class="prob-label">Normal</span>
        <div class="bar-bg"><div class="bar-fill" id="bar-0" style="width:0%;background:#22c55e"></div></div>
        <span class="prob-pct" id="pct-0">—</span>
      </div>
      <div class="prob-row">
        <span class="prob-label">NAION</span>
        <div class="bar-bg"><div class="bar-fill" id="bar-1" style="width:0%;background:#f87171"></div></div>
        <span class="prob-pct" id="pct-1">—</span>
      </div>
      <div class="prob-row">
        <span class="prob-label">ON</span>
        <div class="bar-bg"><div class="bar-fill" id="bar-2" style="width:0%;background:#fbbf24"></div></div>
        <span class="prob-pct" id="pct-2">—</span>
      </div>
    </div>

    <div class="card card-grow" style="flex:1">
      <div class="sec-label">Grad-CAM 히트맵</div>
      <div class="gradcam-wrap" id="gradcam-wrap">
        <div class="gradcam-placeholder">분류 실행 후 히트맵이 표시됩니다</div>
      </div>
      <div class="cam-note" id="cam-note" style="display:none">
        ● 히트맵 집중 영역이 시신경 유두 주변일수록 판독 근거가 명확합니다.
      </div>
    </div>
  </div>
</div>

<div class="statusbar" id="statusbar">준비</div>
<div class="toast" id="toast">클립보드에 복사되었습니다</div>

<script>
let camDataUrl = null;

function status(msg){ document.getElementById('statusbar').textContent = msg; }

function setRunning(v){
  document.getElementById('run-spinner').style.display = v ? 'block' : 'none';
  document.getElementById('run-text').textContent = v ? '분석 중…' : '🔍 분류 + Grad-CAM 실행';
  document.getElementById('run-btn').disabled = v;
}

async function selectFolder(){
  status('폴더 선택 중…');
  const r = await pywebview.api.select_folder();
  if(r.ok){
    document.getElementById('weight-path').value = '폴더 선택 완료';
    document.getElementById('model-dot').classList.add('ok');
    document.getElementById('model-status-text').textContent = `✅ ${r.count}개 모델 로드 완료`;
    status(`${r.count}개 가중치 로드 완료`);
    updateRunBtn();
  } else {
    status('폴더 선택 실패: ' + (r.msg||''));
  }
}

async function loadModels(){
  status('모델 로드 중…');
  const r = await pywebview.api.select_folder();
  if(r.ok){
    document.getElementById('weight-path').value = '폴더 선택 완료';
    document.getElementById('model-dot').classList.add('ok');
    document.getElementById('model-status-text').textContent = `✅ ${r.count}개 모델 로드 완료`;
    status(`${r.count}개 가중치 로드 완료`);
    updateRunBtn();
  }
}

async function openImage(){
  try{
    const r = await pywebview.api.open_image();
    if(!r.ok){
      if(r.msg) status('이미지 로드 실패: ' + r.msg);
      return;
    }
    document.getElementById('file-info').style.display = 'flex';
    document.getElementById('file-name').textContent = r.name;
    document.getElementById('file-meta').textContent = r.meta || '';
    const img = document.getElementById('preview-img');
    img.src = r.preview; img.style.display = 'block';
    status('이미지 로드 완료: ' + r.name);
    updateRunBtn();
  }catch(e){
    status('이미지 선택 중 오류: ' + e);
  }
}

function updateRunBtn(){
  const modelOk = document.getElementById('model-dot').classList.contains('ok');
  const imgOk   = document.getElementById('file-info').style.display === 'flex';
  document.getElementById('run-btn').disabled = !(modelOk && imgOk);
}

async function runInference(){
  setRunning(true);
  status('추론 중…');
  camDataUrl = null;
  document.getElementById('save-btn').disabled = true;

  const doctor = document.getElementById('doctor').value;
  const r = await pywebview.api.run(doctor);
  if(!r.ok){ status('오류: '+(r.msg||'')); setRunning(false); return; }

  // 결과명
  const rname = document.getElementById('result-name');
  rname.textContent = r.pred_name;
  rname.style.color = r.pred_color;

  // 신뢰도 배지
  const cb = document.getElementById('conf-badge');
  cb.textContent = r.conf;
  cb.className = 'conf-badge conf-' + r.conf_level;
  cb.style.display = 'inline-block';

  // 경고
  const wb = document.getElementById('warn-box');
  if(r.warn){ wb.textContent = '⚠ ' + r.warn; wb.style.display = 'block'; }
  else { wb.style.display = 'none'; }

  // 확률 바
  r.probs.forEach((p,i)=>{
    document.getElementById('bar-'+i).style.width = (p*100)+'%';
    document.getElementById('pct-'+i).textContent = (p*100).toFixed(1)+'%';
  });

  // 복사 텍스트
  document.getElementById('copy-text').textContent = r.copy_text;

  status('Grad-CAM 생성 중…');
  const cr = await pywebview.api.run_gradcam();
  if(cr.ok){
    camDataUrl = cr.img;
    document.getElementById('gradcam-wrap').innerHTML = `<img src="${cr.img}" alt="Grad-CAM">`;
    document.getElementById('cam-note').style.display = 'block';
    document.getElementById('save-btn').disabled = false;
    status('완료');
  } else {
    status('Grad-CAM 오류: '+(cr.msg||''));
  }
  setRunning(false);
}

function copyText(){
  const txt = document.getElementById('copy-text').textContent;
  navigator.clipboard.writeText(txt).then(()=>{
    const t = document.getElementById('toast');
    t.classList.add('show');
    setTimeout(()=>t.classList.remove('show'), 2000);
  });
}

function saveResult(){
  if(!camDataUrl) return;
  const a = document.createElement('a');
  a.href = camDataUrl;
  a.download = 'gradcam_result.png';
  a.click();
}

// 자동 모델 로드
window.addEventListener('pywebviewready', async ()=>{
  status('모델 자동 탐색 중…');
  const r = await pywebview.api.auto_load();
  if(r && r.ok){
    document.getElementById('weight-path').value = '자동 로드 완료';
    document.getElementById('model-dot').classList.add('ok');
    document.getElementById('model-status-text').textContent = `✅ ${r.count}개 모델 로드 완료`;
    status(`${r.count}개 가중치 자동 로드 완료`);
    updateRunBtn();
  } else {
    status(r ? (r.msg||'모델 자동 탐색 실패') : '모델 자동 탐색 실패');
  }
});
</script>
</body>
</html>"""

# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import ctypes
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except:
        pass

    api = API()
    window = webview.create_window(
        "안저 이미지 분류기  ·  EfficientNet-B0",
        html=HTML,
        js_api=api,
        width=1320,
        height=880,
        min_size=(1100, 700),
        background_color="#0f1117",
    )
    webview.start(debug=False)
