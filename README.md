# MixVPR: Feature Mixing for Visual Place Recognition

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/mixvpr-feature-mixing-for-visual-place/visual-place-recognition-on-mapillary-test)](https://paperswithcode.com/sota/visual-place-recognition-on-mapillary-test?p=mixvpr-feature-mixing-for-visual-place)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/mixvpr-feature-mixing-for-visual-place/visual-place-recognition-on-mapillary-val)](https://paperswithcode.com/sota/visual-place-recognition-on-mapillary-val?p=mixvpr-feature-mixing-for-visual-place)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/mixvpr-feature-mixing-for-visual-place/visual-place-recognition-on-nordland)](https://paperswithcode.com/sota/visual-place-recognition-on-nordland?p=mixvpr-feature-mixing-for-visual-place)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/mixvpr-feature-mixing-for-visual-place/visual-place-recognition-on-pittsburgh-250k)](https://paperswithcode.com/sota/visual-place-recognition-on-pittsburgh-250k?p=mixvpr-feature-mixing-for-visual-place)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/mixvpr-feature-mixing-for-visual-place/visual-place-recognition-on-pittsburgh-30k)](https://paperswithcode.com/sota/visual-place-recognition-on-pittsburgh-30k?p=mixvpr-feature-mixing-for-visual-place)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/mixvpr-feature-mixing-for-visual-place/visual-place-recognition-on-sped)](https://paperswithcode.com/sota/visual-place-recognition-on-sped?p=mixvpr-feature-mixing-for-visual-place)

This is the official repo for WACV 2023 paper "**MixVPR: Feature Mixing for Visual Place Recognition"**

### Summary

This paper introduces MixVPR, a novel all-MLP feature aggregation method that addresses the challenges of large-scale Visual Place Recognition, while remaining practical for real-world scenarios with strict latency requirements. The technique leverages feature maps from pre-trained backbones as a set of global features, and integrates a global relationship between them through a cascade of feature mixing, eliminating the need for local or pyramidal aggregation. MixVPR achieves new state-of-the-art performance on multiple large-scale benchmarks, while being significantly
more efficient in terms of latency and parameter count compared to existing methods.

[[WACV OpenAccess](https://openaccess.thecvf.com/content/WACV2023/html/Ali-bey_MixVPR_Feature_Mixing_for_Visual_Place_Recognition_WACV_2023_paper.html)] [[ArXiv](https://arxiv.org/abs/2303.02190)]

![architecture](image/README/1678217709949.png)

## Trained models

All models have been trained on GSV-Cities dataset (https://github.com/amaralibey/gsv-cities).

![performance](image/README/1678217802436.png)

### Weights

<table>
<thead>
  <tr>
    <th rowspan="2">Backbone</th>
    <th rowspan="2">Output<br>dimension</th>
    <th colspan="3">Pitts250k-test</th>
    <th colspan="3">Pitts30k-test</th>
    <th colspan="3">MSLS-val</th>
    <th rowspan="2">DOWNLOAD<br></th>
  </tr>
  <tr>
    <th>R@1</th>
    <th>R@5</th>
    <th>R@10</th>
    <th>R@1</th>
    <th>R@5</th>
    <th>R@10</th>
    <th>R@1</th>
    <th>R@5</th>
    <th>R@10</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>ResNet50</td>
    <td>4096</td>
    <td>94.3</td>
    <td>98.2</td>
    <td>98.9</td>
    <td>91.6</td>
    <td>95.5</td>
    <td>96.4</td>
    <td>88.2</td>
    <td>93.1</td>
    <td>94.3</td>
    <td><a href="https://drive.google.com/file/d/1vuz3PvnR7vxnDDLQrdHJaOA04SQrtk5L/view?usp=share_link">LINK</a></td>
  </tr>
 <tr>
    <td>ResNet50</td>
    <td>512</td>
    <td>93.2</td>
    <td>97.9</td>
    <td>98.6</td>
    <td>90.7</td>
    <td>95.5</td>
    <td>96.3</td>
    <td>84.1</td>
    <td>91.8</td>
    <td>93.7</td>
    <td><a href="https://drive.google.com/file/d/1khiTUNzZhfV2UUupZoIsPIbsMRBYVDqj/view?usp=share_link">LINK</a></td>
  </tr>
<tr>
    <td>ResNet50</td>
    <td>128</td>
    <td>88.7</td>
    <td>95.8</td>
    <td>97.4</td>
    <td>87.8</td>
    <td>94.3</td>
    <td>95.7</td>
    <td>78.5</td>
    <td>88.2</td>
    <td>90.4</td>
    <td><a href="https://drive.google.com/file/d/1DQnefjk1hVICOEYPwE4-CZAZOvi1NSJz/view?usp=share_link">LINK</a></td>
  </tr>
</tbody>
</table>

Code to load the pretrained weights is as follows:

```
from main import VPRModel

# Note that images must be resized to 320x320
model = VPRModel(backbone_arch='resnet50', 
                 layers_to_crop=[4],
                 agg_arch='MixVPR',
                 agg_config={'in_channels' : 1024,
                             'in_h' : 20,
                             'in_w' : 20,
                             'out_channels' : 1024,
                             'mix_depth' : 4,
                             'mlp_ratio' : 1,
                             'out_rows' : 4},
                )

state_dict = torch.load('./LOGS/resnet50_MixVPR_4096_channels(1024)_rows(4).ckpt')
model.load_state_dict(state_dict)
model.eval()
```

## Model Export & Quantization

Pre-built ONNX and CoreML models with FP16 / INT8 quantization are hosted on
[Hugging Face](https://huggingface.co/Realcat/image_retrieval_checkpoints/tree/main/mixvpr).

### Download

```bash
# Install huggingface_hub
pip install huggingface_hub

# Download all MixVPR models
huggingface-cli download Realcat/image_retrieval_checkpoints mixvpr/ --local-dir .

# Or download specific files
huggingface-cli download Realcat/image_retrieval_checkpoints mixvpr/onnx/mixvpr_fp16.onnx --local-dir .
huggingface-cli download Realcat/image_retrieval_checkpoints mixvpr/coreml/ --local-dir .
```

### Available Models

| Format | HF Path | Size | Latency (M-series) | CosSim | Status |
|---|---|---|---|---|---|
| ONNX | `mixvpr/onnx/mixvpr_fp32.onnx` | 41.7 MB | 32.4 ms | 1.0000 | ✓ |
| ONNX | `mixvpr/onnx/mixvpr_fp16.onnx` | 21.0 MB | 38.2 ms | 0.9999 | ✓ Recommended |
| CoreML | `mixvpr/coreml/mixvpr_fp16.mlpackage/` | 20.8 MB | **3.1 ms** | 0.9999 | ✓ Recommended |
| CoreML | `mixvpr/coreml/mixvpr_int8.mlpackage/` | 10.5 MB | **3.3 ms** | 0.9983 | ✓ |

> **Note:** Images must be resized to 320×320. The model produces a 4096-dim L2-normalized global descriptor.

### Setup

```bash
uv venv --python 3.10
uv pip install torch torchvision onnx onnxruntime onnxconverter-common coremltools Pillow numpy tqdm
```

### Inference

**ONNX:**

```python
import onnxruntime as ort
import numpy as np
from PIL import Image
import torchvision.transforms as tvf

sess = ort.InferenceSession("onnx_models/mixvpr_fp16.onnx",
                            providers=['CPUExecutionProvider'])

preprocess = tvf.Compose([
    tvf.Resize((320, 320), interpolation=tvf.InterpolationMode.BICUBIC),
    tvf.ToTensor(),
    tvf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
img = preprocess(Image.open("image.jpg").convert("RGB")).unsqueeze(0).numpy()

descriptor = sess.run(None, {'images': img.astype(np.float32)})[0]
# shape: (1, 4096), L2-normalized
```

**CoreML (Apple Silicon):**

```python
import coremltools as ct
import numpy as np
from PIL import Image
import torchvision.transforms as tvf

mlmodel = ct.models.MLModel("coreml_models/mixvpr_fp16.mlpackage")

preprocess = tvf.Compose([
    tvf.Resize((320, 320), interpolation=tvf.InterpolationMode.BICUBIC),
    tvf.ToTensor(),
    tvf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
img = preprocess(Image.open("image.jpg").convert("RGB")).unsqueeze(0).numpy()

descriptor = mlmodel.predict({'images': img.astype(np.float32)})['descriptor']
# Runs on ANE + GPU automatically (~15× faster than ONNX CPU)
```

### Re-export

```bash
# ONNX export + quantization (FP16, INT8, INT4)
uv run python export_quant_onnx.py

# CoreML export + quantization (FP16, INT8)
uv run python export_coreml.py
```

Calibration images are read from `/Volumes/SSD-Realcat/datasets/raco` (roxford5k + rparis6k + revisitop1m). Update the `calib_base` variable in the scripts to point to your own dataset.

### Accuracy Notes

- **ONNX FP16**: CosSim > 0.9999 — retrieval results identical to PyTorch.
- **CoreML FP16**: CosSim > 0.9999 — latency ~3 ms (ANE ~15× speedup).
- **CoreML INT8**: CosSim 0.9983 — latency ~3 ms, only 10.5 MB. Negligible retrieval impact.
- **ONNX INT8/INT4**: Quantized models are generated but the ONNX Runtime CPU EP lacks `ConvInteger` kernels for this architecture. Use GPU EP (CUDA/TensorRT) or switch to CoreML for quantized inference.

## Bibtex

```
@inproceedings{ali2023mixvpr,
  title={{MixVPR}: Feature Mixing for Visual Place Recognition},
  author={Ali-bey, Amar and Chaib-draa, Brahim and Gigu{\`e}re, Philippe},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={2998--3007},
  year={2023}
}
```
