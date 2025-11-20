# Raspberry Pi Outpainting App

Local-only outpainting for Raspberry Pi 5 (ARM64, Bookworm 64-bit). Upload a photo, expand its canvas, and auto-fill the new area realistically. Includes **Photoreal Refine** and optional **Face-Preserve** enhancement — all on CPU.

> **Security**: Binds to `127.0.0.1:8080` so it is not reachable from the network by default.

---

## ✨ Features

- Local-only FastAPI web app (127.0.0.1:8080)
- CPU-friendly inpainting using **LaMa** (ONNXRuntime)
- Automatic border mask for outpainting + optional manual mask upload
- Controls: canvas expansion (%, px), steps, CFG, seed, sampler, denoise
- Toggles: **Photoreal Refine** (RealisticVision 1.4 inpainting), **Face-Preserve** (CodeFormer ONNX)
- Live preview + downloadable PNG
- REST endpoint: `POST /api/outpaint`
- Installer verifies Python version, free disk, model checksums
- Optional `systemd` unit (disabled by default)
- Rotating logs with detailed error traces

---

## 🧰 Installation

### Option A — Manual (recommended)

```bash
bash ~/setup_op
```

This installs to `~/op_app/`, creates `.venv/`, installs dependencies, and downloads models into `~/op_app/models/`.

### Option B — One-liner from GitHub

Replace `<username>` with your GitHub username (repo name must be **raspberry-pi-outpaint**):

```bash
curl -sL https://raw.githubusercontent.com/comp6062/raspberry-pi-outpaint/main/setup_op | bash
```

> The script prefers piwheels.org for ARM wheels.

---

## ▶️ Usage

```bash
~/run_op
```

Then open **http://127.0.0.1:8080** in your Pi’s browser.

---

## 🧩 Systemd (optional, disabled by default)

Copy the unit into your user systemd folder, then enable/start it:

```bash
mkdir -p ~/.config/systemd/user
cp ~/op_app/systemd/op.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable op.service
systemctl --user start op.service
# Later:
systemctl --user stop op.service
systemctl --user disable op.service
```

---

## 🔧 Troubleshooting (Raspberry Pi OS Bookworm, ARM64)

- **Illegal instruction / missing wheels**: Ensure you’re on Pi 5 (ARMv8.2-A) and 64-bit Bookworm. The installer uses piwheels; if pip tries to build from source, check swap size (2–4 GB suggested) and run again.
- **Port 8080 in use**: Stop other services using 8080 or modify `run_op` (`HOST`/`PORT` env).
- **Slow inference**: Increase swap, close other apps, reduce output size, or lower steps/denoise. CPU-only will be slower than GPU.
- **Models cache/reset**: Delete `~/op_app/models/*` and rerun `~/setup_op` to re-download.
- **First run downloads**: All models are downloaded by the installer; app startup should not re-download.

---

## 📁 Folder Layout

```
raspberry-pi-outpaint/
 ├─ app/                 # FastAPI backend + HTML/JS frontend
 ├─ models/              # LaMa + CodeFormer + RealisticVision-inpainting
 ├─ logs/
 ├─ tmp/
 ├─ systemd/op.service
 ├─ .venv/               # created by setup_op
 ├─ setup_op
 ├─ run_op
 ├─ LICENSE
 ├─ .gitignore
 └─ README.md
```

---

## 📜 License & Credits

**MIT License** (see `LICENSE`).

- LaMa: https://github.com/saic-mdal/lama
- LaMa Cleaner: https://github.com/Sanster/lama-cleaner
- CodeFormer: https://github.com/sczhou/CodeFormer
- Realistic Vision 1.4: https://huggingface.co/SG161222/Realistic_Vision_V1.4
