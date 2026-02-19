# Running DIAMOND CSGO on a Headless GCE VM from a Mac

## Overview

The CSGO world model uses Pygame for display, which requires a screen. Since the GCE VM has no physical display, we use a virtual framebuffer (Xvfb) + VNC server, then connect from the Mac via an SSH tunnel.

```
Mac (VNC client) <--SSH tunnel--> GCE VM (Xvfb :99 + x11vnc + Pygame)
```

## One-Time VM Setup

### 1. Install display packages

```bash
sudo apt-get update
sudo apt-get install -y xvfb x11vnc openbox scrot
```

### 2. Start the virtual display

```bash
Xvfb :99 -screen 0 1280x720x24 &
DISPLAY=:99 openbox &
x11vnc -display :99 -passwd diamond -listen 0.0.0.0 -forever -shared &
```

This creates a 1280x720 virtual screen on display `:99`, runs a lightweight window manager, and starts a VNC server with password `diamond`.

### 3. Install tmux (for persistent sessions)

```bash
sudo apt-get install -y tmux
```

### 4. Fix Pygame fullscreen for VNC

In `src/game/game.py`, change:

```python
screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
```

to:

```python
screen = pygame.display.set_mode((1280, 720))
```

`pygame.FULLSCREEN` doesn't render properly on virtual framebuffers.

## Connecting from Mac

### 1. Open an SSH tunnel

```bash
ssh -L 5900:localhost:5900 <VM_IP>
```

This forwards local port 5900 to the VM's VNC server. No firewall rules needed.

### 2. Connect with VNC

Open **Screen Sharing** (built into macOS) or any VNC client and connect to:

```
vnc://localhost:5900
```

Password: `diamond`

## Running Inference

### Interactive (Pygame)

SSH into the VM (or attach to tmux) and run:

```bash
conda activate diamond
cd ~/diamond

# Pretrained model (from HuggingFace)
DISPLAY=:99 python src/play.py

# Local checkpoint
DISPLAY=:99 python src/play.py \
  --local-ckpt outputs/<date>/<time>/checkpoints/agent_versions/agent_epoch_XXXXX.pt \
  --spawn-dir spawn
```

Press Enter in the terminal when prompted, then switch to the VNC window.

Controls:
- WASD: move
- Arrow keys: camera
- Space: jump
- M: switch human/replay mode
- Enter: reset environment
- Esc: quit

### Headless (save frames to disk)

```bash
conda activate diamond
cd ~/diamond
python src/inference_test.py
```

Frames are saved to `inference_frames/`. Edit the script to change `CKPT`, `SPAWN_DIR`, or `NUM_STEPS`.

## Creating Spawn Data

The interactive mode needs spawn data (initial frames + actions to bootstrap the model). To generate it from your training HDF5 files:

```python
import h5py, torch, numpy as np
from PIL import Image

f = h5py.File("src/processed_data_omgrab/full_res/omgrab/omgrab_0.hdf5", "r")
sl = 4  # must match num_steps_conditioning in config/agent/csgo.yaml

os.makedirs("spawn/000", exist_ok=True)
full_res, low_res, acts = [], [], []
for i in range(sl):
    frame = torch.tensor(f[f"frame_{i}_x"][:]).flip(2).permute(2, 0, 1)  # BGR->RGB, HWC->CHW
    full_res.append(frame)
    img = Image.fromarray(frame.permute(1, 2, 0).numpy())
    low_res.append(torch.tensor(np.array(img.resize((56, 30), Image.BICUBIC))).permute(2, 0, 1))
    acts.append(torch.tensor(f[f"frame_{i}_y"][:]))

np.save("spawn/000/full_res.npy", torch.stack(full_res).numpy())
np.save("spawn/000/low_res.npy", torch.stack(low_res).numpy())
np.save("spawn/000/act.npy", torch.stack(acts).numpy())
np.save("spawn/000/next_act.npy", torch.tensor(f[f"frame_{sl}_y"][:]).unsqueeze(0).numpy())
```

Key details:
- `low_res.npy`: shape `(4, 3, 30, 56)` uint8
- `full_res.npy`: shape `(4, 3, 150, 280)` uint8
- `act.npy`: shape `(4, 51)` float
- `next_act.npy`: shape `(1, 51)` float
- The number of frames must equal `num_steps_conditioning` (4), not 5.

## Troubleshooting

| Problem | Fix |
|---------|-----|
| VNC shows black screen | Check `pygame.FULLSCREEN` is replaced with windowed mode |
| VNC connection hangs | Ensure x11vnc was started with `-passwd` flag |
| `GLX` / `BadValue` errors | Use `Xvfb` + VNC instead of `ssh -X` (X11 forwarding) |
| `libGL.so.1 not found` | `sudo apt-get install libgl1 libglib2.0-0` |
| Tensor size mismatch on inference | Spawn data has wrong number of frames (must be 4, not 5) |
| Process dies on SSH disconnect | Run inside `tmux new -s play` and detach with `Ctrl-B D` |
