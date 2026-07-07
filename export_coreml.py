#!/usr/bin/env python3
"""
Convert MixVPR PyTorch → CoreML (FP16 / INT8 / 4-bit palettized).
Benchmark latency & accuracy vs PyTorch reference on Apple Silicon.

Usage:
    uv run python export_coreml.py
"""

import os, sys, time
import numpy as np
from PIL import Image
from collections import OrderedDict

import torch
import torch.nn as nn
import torchvision.transforms as tvf

import coremltools as ct
from coremltools.optimize.coreml import (
    OpLinearQuantizerConfig,
    OptimizationConfig,
    linear_quantize_weights,
    OpPalettizerConfig,
    palettize_weights,
)

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class MixVPRFull(nn.Module):
    def __init__(self):
        super().__init__()
        from models.helper import get_backbone, get_aggregator
        self.backbone = get_backbone('resnet50', True, 0, [4])
        self.aggregator = get_aggregator('MixVPR', {
            'in_channels': 1024, 'in_h': 20, 'in_w': 20,
            'out_channels': 1024, 'mix_depth': 4, 'mlp_ratio': 1, 'out_rows': 4,
        })

    def forward(self, x):
        return self.aggregator(self.backbone(x))


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

INPUT_SIZE = 320
TF = tvf.Compose([
    tvf.Resize((INPUT_SIZE, INPUT_SIZE), interpolation=tvf.InterpolationMode.BICUBIC),
    tvf.ToTensor(),
    tvf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def preprocess(path):
    img = Image.open(path).convert("RGB")
    t = TF(img)
    assert t.shape == (3, INPUT_SIZE, INPUT_SIZE)
    return t.unsqueeze(0)


# ---------------------------------------------------------------------------
# Convert
# ---------------------------------------------------------------------------

def convert_fp16(traced_model, out_path):
    """Convert traced PyTorch → CoreML FP16 mlprogram."""
    print(f"\n[CoreML FP16] Converting ...")
    t0 = time.time()

    mlmodel = ct.convert(
        traced_model,
        inputs=[ct.TensorType(shape=(1, 3, INPUT_SIZE, INPUT_SIZE), name='images')],
        outputs=[ct.TensorType(name='descriptor')],
        convert_to='mlprogram',
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.macOS15,
    )
    mlmodel.save(out_path)
    sz = _dir_size(out_path)
    print(f"[CoreML FP16] Saved → {out_path}  ({sz:.2f} MB)  in {time.time()-t0:.1f}s")
    return mlmodel, out_path


def quantize_int8(mlmodel_fp16, out_path):
    """Apply INT8 linear weight quantization (asymmetric mode)."""
    print(f"\n[CoreML INT8] Applying weight quantization (dtype=int8, asymmetric) ...")
    t0 = time.time()

    config = OptimizationConfig(
        global_config=OpLinearQuantizerConfig(
            mode='linear',           # asymmetric — better for non-centered weight distributions
            dtype='int8',
            granularity='per_channel',
            weight_threshold=8192,   # skip small tensors
        )
    )
    mlmodel_int8 = linear_quantize_weights(mlmodel_fp16, config)
    mlmodel_int8.save(out_path)

    sz = _dir_size(out_path)
    print(f"[CoreML INT8] Saved → {out_path}  ({sz:.2f} MB)  in {time.time()-t0:.1f}s")
    return mlmodel_int8


def quantize_4bit(mlmodel_fp16, out_path):
    """Apply 4-bit linear weight quantization."""
    print(f"\n[CoreML 4bit] Applying weight quantization (dtype=int4) ...")
    t0 = time.time()

    config = OptimizationConfig(
        global_config=OpLinearQuantizerConfig(
            mode='linear',
            dtype='int4',
            granularity='per_channel',
            weight_threshold=8192,
        )
    )
    mlmodel_4bit = linear_quantize_weights(mlmodel_fp16, config)
    mlmodel_4bit.save(out_path)

    sz = _dir_size(out_path)
    print(f"[CoreML 4bit] Saved → {out_path}  ({sz:.2f} MB)  in {time.time()-t0:.1f}s")
    return mlmodel_4bit


def _dir_size(path):
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            total += os.path.getsize(os.path.join(dirpath, f))
    return total / (1024 * 1024)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def coreml_inference(mlmodel, image_tensor, warmup=5, repeats=30):
    """Run CoreML inference, measure time."""
    inp_np = image_tensor.numpy().astype(np.float32)

    # Warmup
    for _ in range(warmup):
        _ = mlmodel.predict({'images': inp_np})

    # Timed runs
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = mlmodel.predict({'images': inp_np})
        times.append(time.perf_counter() - t0)

    out_np = out['descriptor']
    if isinstance(out_np, dict):
        out_np = list(out_np.values())[0]
    return out_np, times


def torch_inference(model, image_tensor, warmup=5, repeats=30):
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(image_tensor)
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            out = model(image_tensor)
            times.append(time.perf_counter() - t0)
    return out.cpu().numpy(), times


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compare(torch_out, coreml_out):
    diff = torch_out - coreml_out
    abs_diff = np.abs(diff)
    cos = float(np.dot(torch_out.flatten(), coreml_out.flatten()))
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
    out_dir = 'coreml_models'
    os.makedirs(out_dir, exist_ok=True)

    # ---- Test images ----
    test_tensors = OrderedDict()
    for f in sorted(os.listdir(data_dir)):
        if f.lower().endswith(('.png', '.jpg', '.jpeg')):
            test_tensors[f] = preprocess(os.path.join(data_dir, f))
            print(f"[TEST]   {f} → {test_tensors[f].shape}")
    print(f"[TEST]   {len(test_tensors)} images")

    # ---- Build & trace model ----
    print(f"\n{'='*70}")
    print(f"  STEP 1 — Trace MixVPR model")
    print(f"{'='*70}")
    model = MixVPRFull()
    model.eval()
    dummy = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE)
    traced = torch.jit.trace(model, dummy)
    out = traced(dummy)
    print(f"  Traced OK — output: {out.shape}, total params: {sum(p.numel() for p in model.parameters()):,}")

    # ---- Convert ----
    print(f"\n{'='*70}")
    print(f"  STEP 2 — Convert to CoreML")
    print(f"{'='*70}")

    ml_fp16, fp16_path = convert_fp16(traced, os.path.join(out_dir, 'mixvpr_fp16.mlpackage'))
    ml_int8 = quantize_int8(ml_fp16, os.path.join(out_dir, 'mixvpr_int8.mlpackage'))
    ml_4bit = quantize_4bit(ml_fp16, os.path.join(out_dir, 'mixvpr_4bit.mlpackage'))

    # ---- Sanity check ----
    print(f"\n{'='*70}")
    print(f"  STEP 3 — Sanity check")
    print(f"{'='*70}")
    dummy_np = dummy.numpy().astype(np.float32)
    for label, ml in [('FP16', ml_fp16), ('INT8', ml_int8), ('4bit', ml_4bit)]:
        try:
            out = ml.predict({'images': dummy_np})['descriptor']
            if isinstance(out, dict):
                out = list(out.values())[0]
            print(f"  {label:6s} ✓  shape={out.shape}  norm={np.linalg.norm(out):.6f}")
        except Exception as e:
            print(f"  {label:6s} ✗  {e}")

    # ---- Benchmark ----
    print(f"\n{'='*70}")
    print(f"  STEP 4 — Benchmark  ({5} warmup + {30} repeats)")
    print(f"{'='*70}")

    print("  PyTorch FP32 ...")
    t0 = time.time()
    torch_res = OrderedDict()
    for name, tensor in test_tensors.items():
        torch_res[name] = torch_inference(model, tensor)
    print(f"  Done in {time.time()-t0:.1f}s")

    benchmarks = OrderedDict([
        ('CoreML FP16', ml_fp16),
        ('CoreML INT8', ml_int8),
        ('CoreML 4bit', ml_4bit),
    ])
    cm_res = OrderedDict()
    for label, ml in benchmarks.items():
        print(f"  {label} ...")
        t0 = time.time()
        cm_res[label] = OrderedDict()
        for name, tensor in test_tensors.items():
            cm_res[label][name] = coreml_inference(ml, tensor)
        print(f"  Done in {time.time()-t0:.1f}s")

    # ---- Consistency ----
    print(f"\n{'='*70}")
    print(f"  STEP 5 — Consistency vs PyTorch")
    print(f"{'='*70}")

    for img_name in test_tensors:
        ref_out, _ = torch_res[img_name]
        print(f"\n  [{img_name}]")
        hdr = f"  {'Variant':<16s} {'CosSim':>12s} {'MaxAE':>12s} {'MeanAE':>12s} {'L2Diff':>12s}"
        print(hdr)
        print(f"  {'─'*16} {'─'*12} {'─'*12} {'─'*12} {'─'*12}")
        for label in benchmarks:
            out_np, _ = cm_res[label][img_name]
            m = compare(ref_out, out_np)
            print(f"  {label:<16s} {m['cosine_similarity']:12.8f} {m['max_abs_error']:12.2e} "
                  f"{m['mean_abs_error']:12.2e} {m['l2_norm_diff']:12.2e}")

    # ---- Latency ----
    print(f"\n\n{'='*70}")
    print(f"  LATENCY (ms, batch=1)")
    print(f"{'='*70}")

    cols = ['PyTorch'] + list(benchmarks.keys())
    print(f"\n  {'Image':<20s}", end='')
    for c in cols: print(f" {c.split()[-1]:>12s}", end='')
    print(f"\n  {'─'*20}", end='')
    for _ in cols: print(f" {'─'*12}", end='')
    print()

    for img_name in test_tensors:
        print(f"  {img_name:<20s}", end='')
        _, t_pt = torch_res[img_name]
        print(f" {np.mean(t_pt)*1000:12.2f}", end='')
        for label in benchmarks:
            _, t_cm = cm_res[label][img_name]
            print(f" {np.mean(t_cm)*1000:12.2f}", end='')
        print()

    print(f"  {'─'*20}", end='')
    for _ in cols: print(f" {'─'*12}", end='')
    print(f"\n  {'AVERAGE':>20s}", end='')
    _, t_pt = torch_res[list(test_tensors.keys())[0]]
    pt_avg = np.mean([np.mean(torch_res[n][1]) for n in test_tensors]) * 1000
    print(f" {pt_avg:12.2f}", end='')
    for label in benchmarks:
        cm_avg = np.mean([np.mean(cm_res[label][n][1]) for n in test_tensors]) * 1000
        print(f" {cm_avg:12.2f}", end='')
    print()

    print(f"\n  Speedup vs PyTorch:")
    for label in benchmarks:
        cm_avg = np.mean([np.mean(cm_res[label][n][1]) for n in test_tensors])
        sp = pt_avg / 1000 / cm_avg
        print(f"    {label:<16s} {sp:.2f}x")

    # ---- File size ----
    print(f"\n\n{'='*70}")
    print(f"  FILE SIZE")
    print(f"{'='*70}")

    for label, p in [('FP16', fp16_path), ('INT8', os.path.join(out_dir, 'mixvpr_int8.mlpackage')),
                     ('4bit', os.path.join(out_dir, 'mixvpr_4bit.mlpackage'))]:
        sz = _dir_size(p)
        print(f"  CoreML {label:<6s} {sz:10.2f} MB")

    # ---- Accuracy verdict ----
    print(f"\n{'='*70}")
    print(f"  ACCURACY VERDICT")
    print(f"{'='*70}")
    for label in benchmarks:
        avg_cos = np.mean([compare(torch_res[n][0], cm_res[label][n][0])['cosine_similarity']
                           for n in test_tensors])
        ok = avg_cos > 0.9999
        print(f"  {label:<16s} CosSim={avg_cos:.8f} → {'✓ NO DROP' if ok else '✗ CHECK'}")

    print(f"\n{'='*70}")
    print(f"  DONE ✓")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
