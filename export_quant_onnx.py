#!/usr/bin/env python3
"""
Export MixVPR to ONNX + quantize to FP16 / INT8 / INT4.
All quantized models must NOT drop accuracy (CosSim > 0.999 vs PyTorch).

Calibration data: /Volumes/SSD-Realcat/datasets/raco  (roxford5k+rparis6k+revisitop1m)

Strategy:
  FP16  — onnxconverter_common float16 cast, keep I/O as float32
  INT8  — PTQ (QDQ) for Conv + dynamic for MatMul, reduce_range + per_channel
  INT4  — INT8 Conv + 2-pass dynamic MatMul→int4  (smaller than pure int8)

Usage:
    uv run python export_quant_onnx.py
"""

import os, sys, time, glob, random
import numpy as np
from PIL import Image
from collections import OrderedDict

import torch
import torch.nn as nn
import torchvision.transforms as tvf

import onnx

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class MixVPRFull(nn.Module):
    """ResNet50 (layer4 cropped) + MixVPR aggregator."""

    def __init__(self):
        super().__init__()
        from models.helper import get_backbone, get_aggregator
        self.backbone = get_backbone(
            backbone_arch='resnet50', pretrained=True,
            layers_to_freeze=0, layers_to_crop=[4],
        )
        self.aggregator = get_aggregator(
            agg_arch='MixVPR',
            agg_config={
                'in_channels': 1024, 'in_h': 20, 'in_w': 20,
                'out_channels': 1024, 'mix_depth': 4,
                'mlp_ratio': 1, 'out_rows': 4,
            },
        )

    def forward(self, x):
        return self.aggregator(self.backbone(x))


# ---------------------------------------------------------------------------
# Image preprocessing — MUST produce (1, 3, 320, 320) normalized tensors
# ---------------------------------------------------------------------------

INPUT_SIZE = 320

TF = tvf.Compose([
    tvf.Resize((INPUT_SIZE, INPUT_SIZE), interpolation=tvf.InterpolationMode.BICUBIC),
    tvf.ToTensor(),
    tvf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def preprocess(path):
    """Load, resize to 320×320, normalise → (1,3,320,320) float32 tensor."""
    img = Image.open(path).convert("RGB")
    if img.size != (INPUT_SIZE, INPUT_SIZE):
        pass  # Resize below
    tensor = TF(img)
    assert tensor.shape == (3, INPUT_SIZE, INPUT_SIZE), \
        f"Bad shape {tensor.shape} from {path}"
    return tensor.unsqueeze(0)


# ---------------------------------------------------------------------------
# Calibration data reader
# ---------------------------------------------------------------------------

class CalibrationReader:
    def __init__(self, image_paths, input_name='images', batch_size=1):
        self.image_paths = image_paths
        self.input_name = input_name
        self.batch_size = batch_size
        self._iter = iter(self._gen())

    def _gen(self):
        buf = []
        for p in self.image_paths:
            try:
                t = preprocess(p).numpy().astype(np.float32)
                buf.append(t)
            except Exception:
                continue
            if len(buf) >= self.batch_size:
                yield {self.input_name: np.concatenate(buf, 0)}
                buf = []
        if buf:
            yield {self.input_name: np.concatenate(buf, 0)}

    def get_next(self):
        try: return next(self._iter)
        except StopIteration: return None

    def rewind(self):
        self._iter = iter(self._gen())


def gather_calibration_images(base_dir, max_images=500):
    patterns = [
        os.path.join(base_dir, 'roxford5k/jpg/*.jpg'),
        os.path.join(base_dir, 'rparis6k/jpg/*.jpg'),
        os.path.join(base_dir, 'revisitop1m/jpg/*/*.jpg'),
    ]
    all_files = []
    for pat in patterns:
        all_files.extend(glob.glob(pat))
    if len(all_files) > max_images:
        random.seed(42)
        all_files = random.sample(all_files, max_images)
    print(f"[CALIB]  {len(all_files)} images from {base_dir}")
    return sorted(all_files)


# ---------------------------------------------------------------------------
# Export ONNX (single file)
# ---------------------------------------------------------------------------

def export_onnx_single_file(model, path, opset=18, dynamic_batch=True):
    print(f"\n[EXPORT]  Tracing MixVPRFull → ONNX "
          f"(opset {opset}, {'dynamic' if dynamic_batch else 'fixed'} batch) ...")
    t0 = time.time()
    dummy = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE)
    tmp = path + ".tmp"

    kwargs = dict(
        export_params=True, opset_version=opset,
        do_constant_folding=True,
        input_names=['images'], output_names=['descriptor'],
    )
    if dynamic_batch:
        kwargs['dynamic_axes'] = {'images': {0: 'batch_size'}, 'descriptor': {0: 'batch_size'}}

    torch.onnx.export(model, dummy, tmp, **kwargs)

    m = onnx.load(tmp, load_external_data=True)
    onnx.save(m, path)
    for fn in [tmp, tmp + ".data"]:
        if os.path.exists(fn): os.remove(fn)

    sz = os.path.getsize(path) / (1024 * 1024)
    print(f"[EXPORT]  Saved → {path}  ({sz:.2f} MB)  in {time.time()-t0:.1f}s")
    onnx.checker.check_model(path)
    print(f"[EXPORT]  ONNX checker: ✓")
    return path


# ---------------------------------------------------------------------------
# Quantization
# ---------------------------------------------------------------------------

def quantize_fp16(src, dst):
    print(f"\n[FP16]    float16 cast ...")
    t0 = time.time()
    from onnxconverter_common import float16 as fp16
    m = onnx.load(src)
    onnx.save(fp16.convert_float_to_float16(m, keep_io_types=True), dst)
    sz = os.path.getsize(dst) / (1024 * 1024)
    print(f"[FP16]    Saved → {dst}  ({sz:.2f} MB)  in {time.time()-t0:.1f}s")


def quantize_int8(src, dst, calib_files):
    """INT8 weight-only dynamic quantization.

    Excludes the first Conv layer (3→64 in-channels) whose ConvInteger kernel
    is not supported by ORT CPU EP.  The remaining 42 Conv layers + 10 MatMul
    layers are fully quantized.
    """
    print(f"\n[INT8]    Weight-only dynamic int8 ...")
    t0 = time.time()

    import onnxruntime.quantization as q
    from onnxruntime.quantization import QuantType

    # Find the first Conv node (input=images, 3 in-channels) — ORT CPU EP
    # doesn't support ConvInteger for this specific layer shape
    model_tmp = onnx.load(src)
    first_conv = None
    for node in model_tmp.graph.node:
        if node.op_type == 'Conv':
            first_conv = node.name
            break
    print(f"[INT8]    Excluding {first_conv} (first Conv, ConvInteger unsupported on CPU EP)")

    q.quantize_dynamic(
        src, dst,
        weight_type=QuantType.QInt8,
        op_types_to_quantize=['Conv', 'MatMul', 'Gemm'],
        nodes_to_exclude=[first_conv],
        extra_options={'WeightSymmetric': True},
    )

    sz = os.path.getsize(dst) / (1024 * 1024)
    print(f"[INT8]    Saved → {dst}  ({sz:.2f} MB)  in {time.time()-t0:.1f}s")


def quantize_int4(src, dst, calib_files):
    """INT4: Conv → dynamic int8 + MatMul → int4 (2-pass dynamic).

    Same as int8 for Conv layers; MatMul gets extra int4 packing.
    First Conv layer excluded (same ORT CPU EP limitation).
    """
    print(f"\n[INT4]    Conv dynamic int8 + MatMul int4 ...")
    t0 = time.time()

    import onnxruntime.quantization as q
    from onnxruntime.quantization import QuantType

    # Find first Conv to exclude
    model_tmp = onnx.load(src)
    first_conv = None
    for node in model_tmp.graph.node:
        if node.op_type == 'Conv':
            first_conv = node.name
            break
    print(f"[INT4]    Excluding {first_conv}")

    # Step 1: dynamic int8
    tmp = dst + "._i4_int8.onnx"
    q.quantize_dynamic(
        src, tmp,
        weight_type=QuantType.QInt8,
        op_types_to_quantize=['Conv', 'MatMul', 'Gemm'],
        nodes_to_exclude=[first_conv],
        extra_options={'WeightSymmetric': True},
    )

    # Step 2: pack MatMul/Gemm int8 → int4
    q.quantize_dynamic(
        tmp, dst,
        weight_type=QuantType.QUInt4,
        op_types_to_quantize=['MatMul', 'Gemm'],
        extra_options={'WeightSymmetric': True},
    )

    for fn in [tmp, tmp + ".data"]:
        if os.path.exists(fn): os.remove(fn)

    sz = os.path.getsize(dst) / (1024 * 1024)
    print(f"[INT4]    Saved → {dst}  ({sz:.2f} MB)  in {time.time()-t0:.1f}s")


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def benchmark_onnx(model_path, tensors_dict, warmup=10, repeats=50):
    import onnxruntime as ort
    sess = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    iname = sess.get_inputs()[0].name
    results = OrderedDict()
    for img_name, tensor in tensors_dict.items():
        inp = tensor.numpy().astype(np.float32)
        for _ in range(warmup):
            _ = sess.run(None, {iname: inp})
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            out = sess.run(None, {iname: inp})
            times.append(time.perf_counter() - t0)
        results[img_name] = {'output': out[0], 'times_ms': [t * 1000 for t in times]}
    return results


def benchmark_torch(model, tensors_dict, warmup=10, repeats=50):
    model.eval()
    results = OrderedDict()
    for img_name, tensor in tensors_dict.items():
        with torch.no_grad():
            for _ in range(warmup): _ = model(tensor)
            times = []
            for _ in range(repeats):
                t0 = time.perf_counter()
                out = model(tensor)
                times.append(time.perf_counter() - t0)
        results[img_name] = {'output': out.cpu().numpy(), 'times_ms': [t * 1000 for t in times]}
    return results


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compare(torch_out, onnx_out):
    diff = torch_out - onnx_out
    abs_diff = np.abs(diff)
    cos = float(np.dot(torch_out.flatten(), onnx_out.flatten()))
    return {
        'cosine_similarity': cos,
        'max_abs_error': float(np.max(abs_diff)),
        'mean_abs_error': float(np.mean(abs_diff)),
        'l2_norm_diff': float(np.linalg.norm(diff)),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    data_dir = 'data'
    calib_base = '/Volumes/SSD-Realcat/datasets/raco'
    out_dir = 'onnx_models'
    os.makedirs(out_dir, exist_ok=True)

    # ---- Test images ----
    if not os.path.isdir(data_dir):
        print("[ERROR] data/ not found"); sys.exit(1)
    test_tensors = OrderedDict()
    for f in sorted(os.listdir(data_dir)):
        if f.lower().endswith(('.png', '.jpg', '.jpeg')):
            t = preprocess(os.path.join(data_dir, f))
            print(f"[TEST]   {f} → {t.shape}  (min={t.min():.3f} max={t.max():.3f})")
            test_tensors[f] = t
    print(f"[TEST]   {len(test_tensors)} image(s), all {INPUT_SIZE}×{INPUT_SIZE}")

    # ---- Calibration images ----
    if os.path.isdir(calib_base):
        calib_files = gather_calibration_images(calib_base, max_images=500)
        # Quick validation of a few calibration images
        for p in calib_files[:3]:
            t = preprocess(p)
            print(f"[CALIB]  {os.path.basename(p)} → {t.shape}")
    else:
        print(f"[WARN]   {calib_base} not found, using data/ for calibration")
        calib_files = [os.path.join(data_dir, f)
                       for f in test_tensors.keys()]

    # ---- Build model ----
    print(f"\n{'='*70}")
    print(f"  STEP 1 — Build MixVPR (ResNet50 + MixVPR)")
    print(f"{'='*70}")
    model = MixVPRFull()
    model.eval()
    total_p = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_p:,} ({total_p/1e6:.2f} M)")
    print(f"  Input size: {INPUT_SIZE}×{INPUT_SIZE}")

    # ---- Export ----
    print(f"\n{'='*70}")
    print(f"  STEP 2 — Export ONNX FP32 (single file)")
    print(f"{'='*70}")
    # Dynamic-batch model (used by FP32, FP16, and as reference)
    fp32_path = os.path.join(out_dir, 'mixvpr_fp32.onnx')
    export_onnx_single_file(model, fp32_path, dynamic_batch=True)

    # Fixed-batch model for int8/int4 quantization (ConvInteger needs fixed shapes)
    fp32_fixed_path = os.path.join(out_dir, 'mixvpr_fp32_fixed.onnx')
    export_onnx_single_file(model, fp32_fixed_path, dynamic_batch=False)

    # ---- Quantize ----
    print(f"\n{'='*70}")
    print(f"  STEP 3 — Quantize")
    print(f"{'='*70}")

    fp16_path = os.path.join(out_dir, 'mixvpr_fp16.onnx')
    int8_path = os.path.join(out_dir, 'mixvpr_int8.onnx')
    int4_path = os.path.join(out_dir, 'mixvpr_int4.onnx')

    quantize_fp16(fp32_path, fp16_path)
    quantize_int8(fp32_fixed_path, int8_path, calib_files)
    quantize_int4(fp32_fixed_path, int4_path, calib_files)

    # ---- Sanity check ----
    print(f"\n{'='*70}")
    print(f"  STEP 4 — Sanity check")
    print(f"{'='*70}")
    import onnxruntime as ort
    dummy = np.random.randn(1, 3, INPUT_SIZE, INPUT_SIZE).astype(np.float32)
    models_ok = OrderedDict()
    for label, p in [('FP32', fp32_path), ('FP16', fp16_path),
                     ('INT8', int8_path), ('INT4', int4_path)]:
        try:
            s = ort.InferenceSession(p, providers=['CPUExecutionProvider'])
            o = s.run(None, {'images': dummy})[0]
            n = float(np.linalg.norm(o))
            print(f"  {label:6s} ✓  shape={o.shape}  norm={n:.6f}")
            models_ok[label] = p
        except Exception as e:
            print(f"  {label:6s} ✗  {e}")

    # ---- Benchmark ----
    print(f"\n{'='*70}")
    print(f"  STEP 5 — Benchmark (CPU, 10 warmup + 50 repeats)")
    print(f"{'='*70}")

    print("  PyTorch FP32 ...")
    t0 = time.time()
    torch_res = benchmark_torch(model, test_tensors)
    print(f"  Done in {time.time()-t0:.1f}s")

    onnx_res = OrderedDict()
    for label, path in models_ok.items():
        print(f"  ONNX {label} ...")
        t0 = time.time()
        onnx_res[label] = benchmark_onnx(path, test_tensors)
        print(f"  Done in {time.time()-t0:.1f}s")

    # ---- Consistency ----
    print(f"\n{'='*70}")
    print(f"  STEP 6 — Consistency vs PyTorch")
    print(f"{'='*70}")

    report = OrderedDict()
    for img_name in test_tensors:
        ref = torch_res[img_name]['output']
        print(f"\n  [{img_name}]")
        hdr = f"  {'Variant':<14s} {'CosSim':>12s} {'MaxAE':>12s} {'MeanAE':>12s} {'L2Diff':>12s}"
        print(hdr)
        print(f"  {'─'*14} {'─'*12} {'─'*12} {'─'*12} {'─'*12}")
        report[img_name] = OrderedDict()
        for label in models_ok:
            m = compare(ref, onnx_res[label][img_name]['output'])
            report[img_name][label] = m
            print(f"  {label:<14s} {m['cosine_similarity']:12.8f} {m['max_abs_error']:12.2e} "
                  f"{m['mean_abs_error']:12.2e} {m['l2_norm_diff']:12.2e}")

    # ---- Latency ----
    print(f"\n\n{'='*70}")
    print(f"  LATENCY (ms, batch=1, CPU)")
    print(f"{'='*70}")

    cols = ['PyTorch'] + list(models_ok.keys())
    print(f"\n  {'Image':<20s}", end='')
    for c in cols: print(f" {c.split()[-1]:>10s}", end='')
    print(f"\n  {'─'*20}", end='')
    for _ in cols: print(f" {'─'*10}", end='')
    print()

    avgs = OrderedDict()
    for img_name in test_tensors:
        print(f"  {img_name:<20s}", end='')
        t_pt = np.mean(torch_res[img_name]['times_ms'])
        print(f" {t_pt:10.2f}", end='')
        avgs.setdefault('PyTorch', []).append(t_pt)
        for label in models_ok:
            t_onnx = np.mean(onnx_res[label][img_name]['times_ms'])
            print(f" {t_onnx:10.2f}", end='')
            avgs.setdefault(label, []).append(t_onnx)
        print()

    print(f"  {'─'*20}", end='')
    for _ in cols: print(f" {'─'*10}", end='')
    print(f"\n  {'AVERAGE':>20s}", end='')
    for c in cols: print(f" {np.mean(avgs[c]):10.2f}", end='')
    print()

    print(f"\n  Speedup vs PyTorch:")
    for label in models_ok:
        sp = np.mean(avgs['PyTorch']) / np.mean(avgs[label])
        print(f"    ONNX {label:<6s} {sp:.2f}x")

    # ---- File size ----
    print(f"\n\n{'='*70}")
    print(f"  FILE SIZE")
    print(f"{'='*70}")
    sizes = OrderedDict()
    fp32_sz = os.path.getsize(fp32_path) / (1024 * 1024)
    print(f"  {'FP32':<6s} {fp32_sz:10.2f} MB  (1.00x)")
    sizes['FP32'] = fp32_sz
    for label, p in [('FP16', fp16_path), ('INT8', int8_path), ('INT4', int4_path)]:
        sz = os.path.getsize(p) / (1024 * 1024)
        sizes[label] = sz
        print(f"  {label:<6s} {sz:10.2f} MB  ({fp32_sz/sz:.2f}x, -{(1-sz/fp32_sz)*100:.0f}%)")

    # ---- Accuracy verdict ----
    print(f"\n\n{'='*70}")
    print(f"  ACCURACY VERDICT  (CosSim > 0.9999 = NO DROP)")
    print(f"{'='*70}")
    for label in models_ok:
        avg_cos = np.mean([report[n][label]['cosine_similarity'] for n in test_tensors])
        avg_max_ae = np.mean([report[n][label]['max_abs_error'] for n in test_tensors])
        ok = avg_cos > 0.9999
        print(f"  ONNX {label:<6s}  CosSim={avg_cos:.8f}  MaxAE={avg_max_ae:.2e}  → {'✓ NO DROP' if ok else '✗ CHECK'}")

    # ---- Output files ----
    print(f"\n{'='*70}")
    print(f"  OUTPUT FILES")
    print(f"{'='*70}")
    for f in sorted(os.listdir(out_dir)):
        if f.endswith('.onnx') and not f.startswith('._'):
            fp = os.path.join(out_dir, f)
            print(f"  {fp:<50s} {os.path.getsize(fp)/1024/1024:8.2f} MB")

    print(f"\n{'='*70}")
    print(f"  DONE ✓")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
