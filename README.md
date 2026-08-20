# BEV-LiDAR Segmentation → Quantized TinyUNet → FINN → FPGA

A low-bit neural-network accelerator project that converts KITTI LiDAR point
clouds into Bird's-Eye-View (BEV) representations, performs BEV segmentation
using a quantized TinyUNet model, and uses the FINN compiler to convert the
INT4 network into an FPGA dataflow accelerator (target board: **Zybo Z7-10**,
part `xc7z010clg400-1`).

---

## 1. One-line summary

> We take 3D LiDAR data from a vehicle, turn it into a 2D top-down (BEV)
> image, run a small quantized AI model on it to detect objects, shrink that
> model from 32-bit to 4-bit numbers, and use FINN to turn the 4-bit model
> into custom FPGA hardware instead of running it on a CPU/GPU.

---

## 2. Pipeline overview

```
KITTI LiDAR (.bin)
        │
        ▼
   BEV Generation  (3 × 300 × 300 tensor)
        │
        ▼
 Quantized TinyUNetQ (Brevitas, INT4 weights / INT8 activations)
        │
        ▼
      QONNX / ONNX export
        │
        ▼
       FINN  (Docker, Ubuntu VM)
        │
        ▼
 FINN Dataflow Architecture (HLS / RTL)
        │
        ▼
     Vivado (synthesis + implementation)
        │
        ▼
   Bitstream → Zybo Z7-10 FPGA
```

---

## 3. Dataset — KITTI

```
training/
    image_2/     → camera images
    label_2/     → object annotations
    velodyne/    → LiDAR point clouds (.bin)  ← primary input
testing/
    image_2/
    velodyne/
```

LiDAR points (`x, y, z, intensity`) are filtered and projected onto an XY
grid to produce a **multi-channel BEV tensor**:

- Channel 0 → occupancy / density
- Channel 1 → height-related information
- Channel 2 → additional BEV information

Input tensor shape: `3 × 300 × 300`.

Approximately **7,481 frames** were exported from KITTI into the BEV
dataset. Example sanity-check output from the pipeline:

```
sample 001736
x2 shape: (2, 300, 300)
x[min,max]: 0.0 1.0
y shape: (300, 300)
y sum: 176
```

---

## 4. Model — TinyUNetQ (Brevitas, INT4)

The network is a compact U-Net-style encoder/decoder, quantized end-to-end
with **Brevitas** (quantization-aware training) so it stays small enough for
FINN to synthesize.

Key settings:

| Setting | Value |
|---|---|
| Input channels | 3 |
| Base channels | 16 |
| Weight bit-width | 4 (`Int4WeightPerTensorFloatDecoupled`) |
| Activation bit-width | 4/8 (`Int8ActPerTensorFixedPoint`) |
| Framework | PyTorch + Brevitas |

**FINN-compatibility change:** skip connections (`torch.cat([...])` between
encoder and decoder blocks) were **removed**, and the corresponding decoder
conv input channels were updated accordingly (e.g. `base*4 → base*2` became
`base*2 → base*2`), since FINN's dataflow build does not support the
original concat-based skip connections in this step of the project.

Layers used: `QuantConv2d`, `QuantConvTranspose2d`, `QuantReLU`,
`QuantIdentity`, with `return_quant_tensor=True` (required for FINN tracing).
`nn.MaxPool2d` is currently used (guide recommends `QuantMaxPool2d` — flagged
for review, not yet resolved).

---

## 5. Quantization background

| Precision | Bits | Hardware cost | Typical accuracy |
|---|---|---|---|
| FP32 | 32 | Highest | Baseline |
| INT8 | 8 | Lower | Usually close to FP32 |
| INT4 | 4 | Very low | Needs verification |

Goal: find the **lowest bit-width that still gives acceptable accuracy**
while producing an efficient FPGA implementation. Low-bit values (e.g. 4-bit
× 4-bit multiplications) require much simpler FPGA logic than floating point,
which is why FINN is attractive for this project.

---

## 6. Evaluation results

### Segmentation metrics (earlier checkpoint)

| Metric | Value |
|---|---|
| PR-AUC | 0.7017 |
| ROC-AUC | 0.9914 |
| Best F1 | 0.6527 (threshold ≈ 0.8436) |
| Precision @ best F1 | 0.6460 |
| Recall @ best F1 | 0.6595 |

### Old (with skip connections) vs New (FINN-compatible, no skip connections)

| Metric | Old | New | Change |
|---|---|---|---|
| Params | 105,491 | 86,295 | −18.20% |
| Loss | 0.0169 | 0.0247 | +0.0078 |
| Accuracy | 0.9934 | 0.9904 | −0.0031 |
| IoU | 0.4091 | 0.3253 | −0.0839 |
| Precision | 0.4495 | 0.3469 | −0.1026 |
| Recall | 0.8201 | 0.8390 | +0.0190 |
| F1 | 0.5807 | 0.4909 | −0.0898 |

Removing the skip connections reduced parameter count but also reduced
segmentation quality — a known trade-off accepted for FINN compatibility.

---

## 7. Environment setup (Windows → VirtualBox → Ubuntu → Docker → FINN)

### 7.1 Machine setup
1. Install **VirtualBox 7.2.14** on Windows.
2. Create an Ubuntu VM.
   - Enable **Nested VT-x/AMD-V** in VM settings (Settings → System →
     Acceleration) — this option is greyed out unless virtualization is
     enabled in the **BIOS/UEFI** first (Intel VT-x / AMD-V), and in some
     cases unless the VM is powered off and hardware virtualization is
     confirmed available via `systeminfo` (Windows) showing
     "Virtualization Enabled in Firmware: Yes".
   - Set VM disk size to **150–200 GB** (FINN + Docker images are large).
3. Enable **Shared Clipboard**: VM window → **Devices → Shared Clipboard →
   Bidirectional** (VM must be running; Guest Additions must be installed
   for this option to be selectable).
4. Open a terminal in Ubuntu: **Activities → search "Terminal"**, or
   right-click Desktop → **Open Terminal**, or `Ctrl+Alt+T`.

### 7.2 One-time setup (already completed)
- Ubuntu installation
- Docker installation
- FINN Docker image pulled (~8.7 GB)
- FINN v0.9 cloned (`~/finn_ws/finn`)
- FINN dependencies installed (`qonnx`, `brevitas`, etc.)
- `run-docker-safe.sh` script created

### 7.3 Every time you want to work on FINN

```bash
# 1. Start the Ubuntu VM in VirtualBox
# 2. Open a terminal
cd ~/finn_ws/finn
bash run-docker-safe.sh
```

You'll land inside the FINN container:

```
root@pavitinder-VirtualBox:/workspace#
```

To leave the container (without destroying your work — files inside the
mounted project folder persist):

```bash
exit
```

Then you can safely close the Ubuntu VM / VirtualBox window.

**You do NOT need to repeat:** Ubuntu install, Docker install, FINN image
pull, FINN clone, or `fetch-repos.sh`. Those are one-time steps only.

### 7.4 Vivado note
Vivado is installed on **Windows**, not inside the Ubuntu FINN container.
The FINN guide's early steps (through model conversion / transformation) do
not require Vivado — it is only needed later, for synthesis, implementation,
and bitstream generation. A "Vivado not available" warning inside the FINN
container is expected until that stage is reached and Vivado is mounted /
made accessible to the container.

---

## 8. Project file locations

```
D:\Segmentation Project\
│
├── training\
│   ├── image_2\
│   ├── label_2\
│   └── velodyne\
├── testing\
│   ├── image_2\
│   └── velodyne\
└── bev_outputs\
    └── bev_outputs_4bit\
        ├── tiny_bev_halo_qat_int4.pt
        ├── tiny_bev_halo_qat_int4_finn.pt
        ├── tiny_bev_halo_qat_int4_finn_qonnx.onnx
        └── dataset_3ch_halo\test\

C:\Users\DELL\OpenPCDet\data\kitti\training      (alternate dataset copy)
D:\Data\essentials\Thapar\MTech\Projects\Main proj\   (run/output dir)
```

Inside the Ubuntu FINN Docker container:

```
/workspace/workspace/my_project/
    ├── build.py
    ├── tiny_bev_halo_qat_int4_finn.pt
    ├── tiny_bev_halo_qat_int4_finn_qonnx.onnx
    └── build_log.txt
```

File transfer path: **Windows → VirtualBox Shared Folder
(`/media/sf_shared`) → Ubuntu → copied into the FINN project workspace.**

---

## 9. FINN build configuration

```python
FPGA_PART = "xc7z010clg400-1"   # Zybo Z7-10
CLOCK_NS  = 10.0
FOLDING_CONFIG = None            # default folding for first build
shell_flow_type = ShellFlowType.VIVADO_ZYNQ
```

Note: the guide originally referenced `ShellFlowType.ZYNQ_IPI`, which does
not exist in the installed FINN version. The installed FINN exposes
`VIVADO_ZYNQ` and `VITIS_ALVEO`, so `VIVADO_ZYNQ` was used for the Zynq-based
Zybo Z7-10 target — this is an adaptation to the installed FINN version, not
a deviation from the guide's intent.

---

## 10. Progress checklist

| Stage | Status |
|---|---|
| KITTI dataset prepared | ✅ Done |
| LiDAR → BEV processing | ✅ Done |
| BEV dataset generation (7,481 frames) | ✅ Done |
| Dataset splitting | ✅ Done |
| TinyUNetQ (INT4) model built | ✅ Done |
| Skip connections removed (FINN Step 1) | ✅ Done |
| Quantization compatibility check (Step 1.3) | ✅ Done |
| Model training | ✅ Done |
| Model evaluation (old vs new) | ✅ Done |
| QONNX / ONNX export | ✅ Done |
| Windows → Ubuntu file transfer | ✅ Done |
| FINN Docker environment set up | ✅ Done |
| QONNX model loads in FINN (47 nodes) | ✅ Done |
| PyVerilator / RTL sim import fixed | ✅ Done |
| `build.py` written, Zybo Z7-10 target set | ✅ Done |
| `ShellFlowType.ZYNQ_IPI` → `VIVADO_ZYNQ` fix | ✅ Done |
| FINN model conversion (`step_qonnx_to_finn`) | ❌ **Blocked (current issue)** |
| HLS synthesis | ⏸️ Not reached |
| IP stitching | ⏸️ Not reached |
| Vivado synthesis / implementation | ⏸️ Not reached |
| Bitstream generation | ⏸️ Not reached |
| FPGA testing | ⏸️ Not reached |

---

## 11. Current blocker

**Step:** `step_qonnx_to_finn [1/17]` (the very first FINN transformation
step, before folding, HLS, or Vivado).

**Error:**
```
ValueError: FINN only supports unsigned and non-narrow Quant nodes for Relu activations.
```

**Meaning:** FINN's early transformation stage found a `Quant` node feeding
a `Relu` that is configured as **signed and/or narrow-range**, which FINN
does not support for ReLU activations. FINN expects an **unsigned**
activation quantizer (the guide's reference example uses
`Uint8ActPerTensorFixedPoint`) for these positions.

**Current investigation step:** inspect the exported ONNX graph to find the
exact `Quant`/`Relu` node(s) responsible, before making any changes to the
model or `build.py`:

```bash
python3 - << 'EOF'
from qonnx.core.modelwrapper import ModelWrapper

model = ModelWrapper(
    "/workspace/workspace/my_project/tiny_bev_halo_qat_int4_finn_qonnx.onnx"
)

print("=== QUANTIZATION NODES ===")
for node in model.graph.node:
    if "Quant" in node.op_type or node.op_type in ["Relu", "Quant"]:
        print()
        print("Node name :", node.name)
        print("Op type   :", node.op_type)
        print("Inputs    :", list(node.input))
        print("Outputs   :", list(node.output))
        for attr in node.attribute:
            print("Attribute :", attr.name, "=", attr)
EOF
```

**Do not** run `python3 build.py` again until this is resolved — it will
fail at the same step and re-trigger the same long-running attempt.

---

## 12. Next steps (in order)

1. Inspect the QONNX graph's `Quant`/`Relu` nodes (command above).
2. Identify which activation quantizer(s) are signed/narrow and need to
   become unsigned (likely switching relevant `QuantReLU` activation
   quantizers toward an unsigned fixed-point quantizer, consistent with the
   guide's reference configuration).
3. Re-export QONNX after the fix and re-verify node count/shapes.
4. Re-run `step_qonnx_to_finn` only (not the full multi-hour build) to
   confirm the fix.
5. Continue strictly through the guide's remaining FINN steps in order:
   Streamline → Lowering → Cleanup → Folding → HLS synthesis → IP stitching
   → weight verification → Vivado block design → synthesis/implementation →
   bitstream → deploy to Zybo Z7-10.
6. After a successful first build, tune folding (PE/SIMD) for
   resource/performance trade-offs — not before.

---

## 13. Debugging philosophy for this project

At every stage, confirm the output makes sense **before** moving to the
next stage:

| Stage | What to check |
|---|---|
| BEV generation | shape, min/max, non-zero pixel count, visualization |
| Model training | loss, F1, IoU, precision, recall |
| Quantization | bit-width, accuracy degradation vs FP32 |
| ONNX/QONNX export | graph structure, input/output shapes, operators |
| FINN | graph transformations, datatypes, folding, build logs |
| FPGA | bitstream validity, runtime output, latency, resource usage |

This avoids debugging multiple systems (model, export, FINN, Vivado) at the
same time.

---

## 14. Glossary (quick reference)

- **BEV** — Bird's-Eye View; top-down 2D projection of a 3D LiDAR point cloud.
- **Brevitas** — PyTorch library for quantization-aware training.
- **ONNX** — standardized neural-network graph format.
- **QONNX** — ONNX variant that represents arbitrary-precision quantized
  networks; FINN's intermediate representation.
- **FINN** — Xilinx/AMD framework that compiles quantized neural networks
  into FPGA dataflow accelerators.
- **Dataflow architecture** — hardware pipeline where each layer is a
  dedicated, parallel processing block, rather than sequential CPU
  instructions.
- **LUT / FF / BRAM / DSP** — FPGA hardware resources (logic, registers,
  on-chip memory, arithmetic blocks) tracked when evaluating a build.
