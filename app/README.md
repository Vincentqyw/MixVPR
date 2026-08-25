# PlaceLens — on-device visual place recognition (iOS)

A small SwiftUI app that runs **MixVPR** and **MegaLoc** CoreML models on the live camera feed.
Capture views of places into sessions, then watch the viewfinder recognise them in real time.

## The app

A **session is one capture session**: a named set of images that may come from one place or many.
Retrieval runs **inside the current session by default**; Settings › Retrieval › *Search across all
sessions* widens it to every session. Either way the result is a specific image — shown as
*Session · #index* — and tapping it opens a preview.

| | |
|---|---|
| **Viewfinder** | 4:3 camera; the corner brackets mark the square crop the model sees and double as the match indicator: white → nothing, amber → weak, green → confident. The corner pill reads `model 9.1 ms · camera 24 fps` — model inference time per frame, and the frame rate the camera delivers. |
| **Match card** | Thumbnail of the best-matching stored image, its session and index, and the similarity. Tap to preview it (with a shortcut to switch to that session when it belongs to another one). |
| **Shutter** | Adds the current frame to the current session. |
| **＋ New** (bottom-left) | Starts a new session and makes it current. **Sessions** (bottom-right) lists every session with its image count and, for the sessions being searched, its live best similarity; tap to switch, swipe to delete, long-press to rename. The top-left chip shows the current session and its image count; tap it to rename. |
| **Shelf** | Images of the current session, newest first, each with a live similarity bar and score; the best match is ringed. Tap to preview, long-press to delete. |
| **Smoothing** | Scores are EMA-filtered (α = 0.35), the winning image only changes after a challenger leads by 0.03 for 4 frames, and the UI is refreshed at 4 Hz, so names and numbers do not flicker. |
| **Settings** (top-right) | Model (MixVPR/MegaLoc × FP16/INT8), compute unit (ANE/GPU/CPU), retrieval scope, per-family match threshold, benchmark. Sessions carry descriptors for *both* model families — switching model re-indexes any image that lacks one (from the stored crop). |
| **Benchmark** | Every bundled model × compute unit; MegaLoc uses 8 iterations, MixVPR 50. **Stop** cancels at the next iteration boundary. |

Sessions persist as binary plists in Documents/sessions/. Icon: `Assets.xcassets/AppIcon.appiconset/AppIcon.png`
(generated procedurally — viewfinder brackets + pin).

## Measured on iPhone 15 Pro Max (A17 Pro, iOS 26.5)

Inference only, median of the built-in benchmark after warm-up (`--bench`):

| Model | Input | ANE | GPU | CPU | cos vs PyTorch fp32 (best unit) |
|-------|------:|----:|----:|----:|------:|
| MixVPR FP16 (21 MB) | 320² | **2.5 ms · 375 fps** | 13.6 ms | 10.4 ms | 0.99999 (ANE) |
| MixVPR INT8 (11 MB) | 320² | **2.5 ms · 370 fps** | 14.9 ms | 10.9 ms | 0.9983 (ANE) |
| MegaLoc FP16 (437 MB) | 322² | 128 ms ⚠ | **154 ms · 6.5 fps** | 158 ms | 0.999996 (GPU) |
| MegaLoc INT8 (219 MB) | 322² | 186 ms ⚠ | 119 ms · 8.4 fps | **96 ms · 10 fps** | 0.9985 (GPU) |
| MegaLoc FP16 | 518² (paper res.) | 633 ms ⚠ | 585 ms · 1.7 fps | 658 ms | 0.99999 (GPU) |
| MegaLoc INT8 | 518² | 698 ms ⚠ | 615 ms | 746 ms | 0.9988 (GPU) |

⚠ MegaLoc on the iPhone's ANE is inaccurate (cos 0.94–0.95 vs PyTorch) — see below — and no faster,
so the app defaults MegaLoc to the GPU and MixVPR to the ANE. Preprocessing costs 0.2–0.4 ms.

### Why the live view shows ~9 ms for MixVPR when the benchmark says 2.6 ms

The benchmark runs inferences back-to-back, so the CPU and Neural Engine sit at their top
clock. The live view is a sparse 24–30 Hz workload and iOS keeps the clocks low for that duty
cycle. `--log` mode runs five inferences per frame and they get faster one after another as
the clocks ramp (`7.1 → 5.5 → 4.9 → 4.3 → 3.8 ms`). Not camera contention (the benchmark with
the camera running still gives 2.8 ms), not thread QoS. Peak throughput ≈ 350 fps; steady-state
cost at camera rate ≈ 11 ms/frame — a third of the 33 ms budget at a fraction of the power.

### MegaLoc on the Neural Engine

Plain fp16 MegaLoc is accurate on the GPU (cos 0.9998) but collapses on the ANE (cos 0.94):
DINOv2's activation outliers overflow the ANE's fp16 `layer_norm`/`softmax`. On the M4 Pro,
keeping every op except `linear/matmul/conv/gelu` in fp32 (`megaloc/export_ios.py`, variant
`ane`) restores cos 0.9999 on the ANE — but the **A17 Pro's ANE still returns cos 0.95** with
the same package, and its ANE latency is no better than the GPU because 529–1369-token
attention does not map well onto it. Conclusion: on iPhone, MegaLoc = GPU (or INT8 on CPU).

At the paper resolution (518²) MegaLoc runs at 1.7 fps on the phone; the bundled export uses
322² (2.7× faster, cos 0.99999 to PyTorch at the same resolution). Same-image (90 % crop)
similarity drops from ~0.86 to ~0.81 while neighbouring-frame similarity stays ~0.73–0.76, so
the default MegaLoc threshold is 0.80 (MixVPR: 0.70). `MEGALOC_SIZE=518 ./prepare_models.sh`
bundles the 518² export instead — change `VPRModel.inputSize` in `Models.swift` to match.

### Do iPhone and PC produce the same descriptor?

Not bit-for-bit, but numerically equivalent on the recommended unit. `--embed` runs the bundled
`test_320.png` / `test_322.png` through the exact live pipeline; compared with the PyTorch fp32
checkpoints on the Mac (same pixels, no resize):

| iPhone descriptor | cos vs PyTorch fp32 | max \|Δ\| |
|---|---:|---:|
| MixVPR FP16 · ANE / GPU / CPU | 0.999991 / 0.999995 / 0.999883 | 2.6e-4 / 1.8e-4 / 9.4e-4 |
| MixVPR INT8 · ANE | 0.998273 | 3.4e-3 |
| MegaLoc FP16 · GPU / CPU / ANE | 0.999996 / 0.998739 / **0.954** | 1.5e-4 / 2.2e-3 / 1.3e-2 |
| MegaLoc INT8 · GPU / CPU / ANE | 0.998532 / 0.997217 / 0.998497 | 2.1e-3 / 3.0e-3 / 2.2e-3 |

In real use the **resize** is the larger source of divergence (vImage Lanczos in the app vs
PIL bicubic/Lanczos on the PC); descriptors that share a database should share a resampler.

## Build & run

```bash
brew install xcodegen                     # once
cd app
./prepare_models.sh                       # compiles the .mlpackages → VPR_iOS/Models/*.mlmodelc, generates the .xcodeproj
open VPR_iOS.xcodeproj                 # set your team, run on a device (ANE needs real hardware)
```

Model sources: MixVPR from `../coreml_models/` ([HuggingFace](https://huggingface.co/Realcat/image_retrieval_checkpoints/tree/main/mixvpr/coreml)),
MegaLoc from `../../megaloc/coreml_ios_322/` (produced by `megaloc/export_ios.py --size 322 --variants ane`).
Missing MegaLoc packages are skipped; the app only lists models that are bundled.

Command line (device paired in Xcode, Developer Mode on):

```bash
xcodebuild -project VPR_iOS.xcodeproj -scheme VPR_iOS -configuration Release \
  -destination 'generic/platform=iOS' -derivedDataPath build -allowProvisioningUpdates build
xcrun devicectl device install app --device <UDID> build/Build/Products/Release-iphoneos/VPR_iOS.app
xcrun devicectl device process launch --console --device <UDID> com.anureka.mixvprdemo --bench   # BENCH lines
xcrun devicectl device process launch --console --device <UDID> com.anureka.mixvprdemo --embed   # EMBED lines
xcrun devicectl device process launch --console --device <UDID> com.anureka.mixvprdemo --log     # LIVE lines
```

## Code map

| File | Role |
|------|------|
| `Models.swift` | `VPRModel` (family, precision, resource, input size), `ComputeChoice`; only bundled models are listed. |
| `VPREngine.swift` | Loads a `.mlmodelc`, preprocesses a `CVPixelBuffer` (centre-crop → N² → ImageNet-normalised fp16 NCHW), runs prediction, L2-normalises. Cancellable benchmark. |
| `CameraManager.swift` | `AVCaptureSession` (back camera, 640×480 BGRA, portrait) + SwiftUI preview layer. |
| `Sessions.swift` | `Session → PlaceImage` model and the plist store. |
| `VPRWorker.swift` | Serial worker: per-frame preprocess → infer → cosine against the searched sessions' images, smoothing + hysteresis on the best image; session/image CRUD; re-indexing; benchmark with stop. |
| `AppState.swift` | `@MainActor` view state, persisted preferences, capture/rename/benchmark actions. |
| `ContentView.swift` / `Viewfinder.swift` / `Shelf.swift` / `SessionsSheet.swift` / `SettingsSheet.swift` | UI. |

Both models expect ImageNet-normalised RGB at exactly their input size; normalisation is **not** baked
into the CoreML packages.
