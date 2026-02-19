# Processed Data Format

This directory contains CS:GO gameplay data processed by `process_csgo_tar_files.py` from the
[CS:GO Behavioural Cloning dataset](https://github.com/TeaPearce/Counter-Strike_Behavioural_Cloning/).

The same format is used for custom data produced by
`Counter-Strike_Behavioural_Cloning/raw_data/scripts/main.py` (written to `processed_data_omgrab/`).

## Directory Structure

```
processed_data/
├── full_res/                              # High-resolution HDF5 files
│   └── <subdir>/                          # e.g. 1201-1400 (CSGO) or omgrab (custom)
│       └── <name>_<N>.hdf5                # One file per episode, exactly 1000 frames
│
└── low_res/                               # Downscaled PyTorch episodes
    ├── train/
    │   ├── info.pt                        # Dataset metadata
    │   └── <episode_dirs>/                # Hierarchical episode directories
    │       └── <id>.pt                    # Single episode file
    └── test/
        ├── info.pt
        └── <episode_dirs>/
            └── <id>.pt
```

## full_res

Each `.hdf5` file contains **exactly 1000 frames**. This is a hard requirement --
`CSGOHdf5Dataset` in `src/data/dataset.py` assumes every episode is 1000 frames.
Clips shorter than 1000 frames must be discarded; clips longer than 1000 frames
must be split into 1000-frame chunks (any leftover tail is discarded).

### HDF5 layout (per frame `i` from 0 to 999)

| Key                    | Shape         | Dtype   | Description                           |
|------------------------|---------------|---------|---------------------------------------|
| `frame_{i}_x`          | (150, 280, 3) | uint8   | RGB screenshot (H x W x C)           |
| `frame_{i}_y`          | (51,)         | float64 | Action vector (see below)             |
| `frame_{i}_xaux`       | (54,)         | float64 | Auxiliary game-state features         |
| `frame_{i}_helperarr`  | (2,)          | float64 | Helper values (e.g. reward signals)   |

### Filename convention

Filenames must end with `_<integer>.hdf5` so `CSGOHdf5Dataset` can sort them
(it extracts the integer via `stem.split("_")[-1]`).

- CSGO originals: `hdf5_dm_july2021_1201.hdf5` (integer = 1201)
- Custom data: `omgrab_0.hdf5` (integer = 0)

Files are placed in a subdirectory (e.g. `1201-1400/` or `omgrab/`).
The dataset key for each episode is `"<subdir>/<filename>"` -- this must match
the `original_file_id` stored in the corresponding low_res `.pt` episode.

### Action vector (51 dimensions)

```
[0:11]   keys:    w=0, a=1, s=2, d=3, space=4, ctrl=5, shift=6, 1=7, 2=8, 3=9, r=10
[11]     left click
[12]     right click
[13:36]  mouse_x: 23-bin one-hot (zero-movement at bin index 11)
[36:51]  mouse_y: 15-bin one-hot (zero-movement at bin index 7)
```

## low_res

PyTorch-serialized episodes created by resizing `full_res` frames from 150x280
down to **30x56** using bicubic interpolation. This is the data used for DIAMOND
training.

### Episode files (`<id>.pt`)

Each `.pt` file is a dict saved with `torch.save` containing:

| Key     | Shape              | Dtype       | Description                                        |
|---------|--------------------|-------------|----------------------------------------------------|
| `obs`   | (1000, 3, 30, 56)  | uint8       | Observations (T x C x H x W), pixel values [0,255] |
| `act`   | (1000, 51)         | float64     | Actions per timestep                                |
| `rew`   | (1000,)            | float32     | Rewards (all zeros for CSGO/custom)                 |
| `end`   | (1000,)            | uint8       | Episode termination flags (all zeros)               |
| `trunc` | (1000,)            | uint8       | Truncation flags (all zeros)                        |
| `info`  | dict               | --          | `{"original_file_id": "<subdir>/<name>.hdf5"}`      |

The `original_file_id` links back to the corresponding full_res HDF5 file.
It must exactly match a key in `CSGOHdf5Dataset._filenames`.

### Episode directory numbering

Episodes are stored in a 3-level hierarchy derived from the episode ID to avoid
large flat directories. The path for episode `N` is:

```
{hundreds_digit}00/{tens_digit}0/{ones_digit}/{N}.pt
```

For example: episode 0 -> `000/00/0/0.pt`, episode 42 -> `000/40/2/42.pt`,
episode 136 -> `100/30/6/136.pt`.

### Metadata (`info.pt`)

Each split's `info.pt` contains dataset-level metadata:

| Key            | Type         | Description                                   |
|----------------|--------------|-----------------------------------------------|
| `num_episodes` | int          | Total number of episodes                      |
| `num_steps`    | int          | Total timesteps (= num_episodes x 1000)       |
| `lengths`      | numpy array  | Length of each episode (always 1000)           |
| `start_idx`    | numpy array  | Cumulative start index per episode             |
| `counter_rew`  | Counter      | Distribution of reward values                  |
| `counter_end`  | Counter      | Distribution of end flags                      |
| `is_static`    | bool         | Whether the dataset is static (False)          |

## Frame rate

The original CSGO data was recorded at **~16 fps**. Custom data from MKV
recordings may be at a different rate (e.g. 25 fps). The model trains on
frame-to-frame transitions regardless of the source rate, but differences
in apparent motion per step may affect generation quality.

## Creating custom data

Use `Counter-Strike_Behavioural_Cloning/raw_data/scripts/main.py`:

```bash
python main.py <out_dir> --split train --chunk-size 1000
```

This scans for `<uuid>.mkv` + `<uuid>_hands.ndjson` pairs in the `raw_data/`
folder, infers actions from hand landmarks (headless), and writes the dataset.
Clips shorter than 1000 frames are discarded. Longer clips are split into
1000-frame chunks with any leftover tail discarded.
