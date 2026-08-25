# MixVPR iOS demo

A minimal SwiftUI app that runs the CoreML MixVPR model on the live camera feed:
register a few places, then watch the app recognise them in real time.
It also ships a built-in benchmark over every model × compute-unit combination.

<p align="center">
  <b>Live view</b> · 320×320 centre crop · 4096-d descriptor per frame · cosine match against registered places
</p>

## Measured on iPhone 15 Pro Max (A17 Pro, iOS 26.5)

Inference-only latency of the compiled `.mlmodelc`, 50 iterations after warm-up (`--bench` mode):

| Model | Compute | Median | p95 | Throughput | First load |
|-------|---------|-------:|----:|-----------:|-----------:|
| FP16 | **ANE** | **2.6 ms** | 3.6 ms | **~360 FPS** | 40–50 ms |
| INT8 | **ANE** | **2.4 ms** | 3.1 ms | **~400 FPS** | 35 ms (1.7 s on first-ever launch) |
| FP16 | GPU | 14.4 ms | 15.5 ms | ~69 FPS | 70 ms–2 s |
| INT8 | GPU | 14.9 ms | 15.7 ms | ~68 FPS | 1–13 s |
| FP16 | CPU | 10.9 ms | 11.2 ms | ~93 FPS | 60 ms |
| INT8 | CPU | 11.4 ms | 11.6 ms | ~89 FPS | 60 ms |

Preprocessing (centre crop → vImage scale → ImageNet-normalised fp16 NCHW) costs **~0.2 ms**.
With the Neural Engine the whole pipeline is ~3 ms/frame, so the live demo is
capped by the camera at 30 FPS while using <10 % of a core. The 21 MB FP16 model is
the recommended default; INT8 halves the size for no measurable speed gain on ANE.

### Why the live view shows ~9 ms when the benchmark says 2.6 ms

The benchmark runs inferences back-to-back, so the CPU and Neural Engine sit at their
top clock. The live view is a sparse 24–30 Hz workload (a ~10 ms burst, then ~25 ms idle),
and iOS power management keeps the clocks low for that duty cycle. `--log` mode proves it by
running five inferences per frame — they get faster one after another as the clocks ramp:

```
LIVE FP16 ANE fps=24.8 pre=1.68ms infer#1=7.07ms then 5.51 → 4.85 → 4.31 → 3.78 ms
```

The same ramp shows up in the preprocessing loop (0.2 ms hot vs ~2 ms in the live view).
This is not camera contention (running the benchmark with the camera active still gives
2.8 ms) and not thread QoS (`.userInteractive` changes nothing). Practical reading: peak
throughput is ~350 FPS, steady-state cost at camera rate is ~11 ms/frame including
preprocessing, which is still a third of the 33 ms frame budget at a fraction of the power.

### Do iPhone and PC produce the same descriptor?

Not bit-for-bit, but numerically equivalent. `--embed` runs the bundled 320×320
`test_image.png` through the exact live pipeline; compared with the PyTorch fp32
checkpoint on the Mac (same image, no resize involved):

| Descriptor source | cos vs PyTorch fp32 | max |Δ| (unit vectors) | identical bits |
|-------------------|--------------------:|-------------------------:|---------------:|
| iPhone CoreML FP16 · ANE | 0.999991 | 2.4e-4 | 0 % |
| iPhone CoreML FP16 · GPU | 0.999994 | 2.2e-4 | 0 % |
| iPhone CoreML FP16 · CPU | 0.999889 | 8.6e-4 | 0 % |
| iPhone CoreML INT8 · ANE | 0.998187 | 3.5e-3 | 0 % |
| Mac (M4 Pro) CoreML FP16 · ANE | 0.999991 | 2.4e-4 | 0 % |

iPhone vs Mac CoreML with the same model and compute unit: cos = 1.000000
(max |Δ| 6e-5 on ANE, 7e-9 on CPU). The fp16 arithmetic, different kernels and
accumulation order make bit-exactness impossible; a cosine of 0.99999 is
far inside the noise of any retrieval threshold. In real use the **resize** is the
larger source of divergence (vImage Lanczos in the app vs PIL bicubic on the PC), so
if descriptors from both sides go into one database, use the same resampling filter.

## Build & run

```bash
brew install xcodegen                     # once
cd ios_demo
./prepare_models.sh                       # compiles ../coreml_models/*.mlpackage → MixVPRDemo/Models/*.mlmodelc, generates the .xcodeproj
open MixVPRDemo.xcodeproj                 # set your team, run on a device (the ANE path needs real hardware)
```

Command-line alternative (device paired in Xcode, Developer Mode on):

```bash
xcodebuild -project MixVPRDemo.xcodeproj -scheme MixVPRDemo -configuration Release \
  -destination 'generic/platform=iOS' -derivedDataPath build -allowProvisioningUpdates build
xcrun devicectl device install app --device <UDID> build/Build/Products/Release-iphoneos/MixVPRDemo.app
# headless benchmark, results are printed as BENCH lines
xcrun devicectl device process launch --console --device <UDID> com.anureka.mixvprdemo --bench
```

The `.mlpackage` files come from the [HuggingFace release](https://huggingface.co/Realcat/image_retrieval_checkpoints/tree/main/mixvpr/coreml)
(`mixvpr/coreml/mixvpr_fp16.mlpackage`, `mixvpr_int8.mlpackage`); place them in `../coreml_models/`.

## How it works

Launch flags (`xcrun devicectl … launch … com.anureka.mixvprdemo <flag>`): `--bench` prints
`BENCH` lines, `--log` prints per-second live timings with a 5-inference burst, `--embed`
prints the descriptor of `test_image.png` (base64 float32) for cross-platform comparison.

| File | Role |
|------|------|
| `MixVPREngine.swift` | Loads a `.mlmodelc` with the chosen `MLComputeUnits`, preprocesses a `CVPixelBuffer` into the fp16 `[1,3,320,320]` input, runs prediction, L2-normalises the 4096-d output. Also hosts the benchmark. |
| `CameraManager.swift` | `AVCaptureSession` (back camera, 640×480 BGRA, portrait) + SwiftUI preview layer. |
| `VPRWorker.swift` | Serial worker queue: per-frame preprocess → infer → cosine scores against registered places; drops frames while busy. |
| `AppState.swift` / `ContentView.swift` | UI: FPS / latency readout, model & compute pickers, place strip, register / clear / bench buttons. |

Match thresholds shown in the UI: ≥ 0.70 confident (green), 0.55–0.70 maybe (yellow), otherwise unknown.
The model's input normalisation is **not** baked into the CoreML package, so any other client must
apply `(x/255 − mean)/std` with ImageNet statistics in RGB order and resize to exactly 320×320.
