# DispTrack — ACM MM 2026

Core training and inference code for *DispTrack: Disparity-Aware RGB-IR Multi-Object Tracking via Association-Level Fusion*.

## Setup

Install a matching PyTorch/torchvision pair, then `pip install -r requirements.txt`. The historical BoxMOT version is not pinned; this implementation uses its internal tracker APIs.

## Train

```bash
python train.py --data-root data/M3OT --fusion-init weights/fusion_init.pth --ir-init weights/ir_only.pth --output outputs/disptrack
```

`--data-root` contains `1/{rgb,ir}/{train,val,test}` and `Annotations/1/{rgb,ir}/*_cocoformat.json`. The fusion initialization checkpoint should already use the dual-model parameter names; the IR-only checkpoint uses `backbone`, `encoder`, and `decoder` prefixes. The trainer validates required parameter coverage and freezes the complete IR branch. RGB/IR annotations are paired by sequence and frame name rather than image ID, and unmatched frames are rejected. It retains paired augmentation, the dual-branch DEIM criterion, AdamW, flat-cosine scheduling, AMP, EMA, validation AP selection and resume support. It targets a single device.

The training entry is a refactoring of the available local implementation, not a recovered exact final run. Its explicit default `--selection-split test` follows the available historical training configuration; the four-sequence `val` split is used for the paper's tracking comparisons. Choose `--selection-split val` only if intentionally changing this protocol. Neither split should be described as an untouched test set when used for selection. Current annotations also contain a 451/450-frame RGB/IR discrepancy in training sequence 1-07; resolve it before training. Final checkpoint compatibility and paper metrics remain unverified.

## Track

```bash
python scripts/run_tracking.py --checkpoint weights/disptrack.pth --rgb data/M3OT/1/rgb/val/1-08 --ir data/M3OT/1/ir/val/1-08T --output outputs/1-08T
```

Uses the dual model's IR detections and fusion-branch identities, ByteTrack, beta=0.35, min_hits=2, and last-tid. Add `--online-only` to disable offline consolidation. Output is `output/track_results/results.txt` in IR coordinates.

## Dataset splitting

See [splitting rules](data_protocol/README.md) and `tools/split_paired_sequence.py`. Images, annotations and weights are not included.

## Attribution

Built on DEIM, D-FINE, RT-DETR and BoxMOT. Original copyright notices and LICENSE are retained; dependencies retain their own licenses.

## Citation

If you find this work or code useful in your research, please consider citing our paper:

[DispTrack: Disparity-Aware RGB-IR Multi-Object Tracking via Association-Level Fusion](https://dl.acm.org/doi/10.1145/3767308.3835740)

```bibtex
@inproceedings{liu2026disptrack,
  author    = {Liu, Xuyang and Jin, Qunchao and Zhou, Qiliang and Chen, Hang},
  title     = {{DispTrack}: Disparity-Aware {RGB-IR} Multi-Object Tracking via Association-Level Fusion},
  booktitle = {Proceedings of the 34th ACM International Conference on Multimedia},
  series    = {MM '26},
  year      = {2026},
  publisher = {Association for Computing Machinery},
  doi       = {10.1145/3767308.3835740},
  url       = {https://dl.acm.org/doi/10.1145/3767308.3835740}
}
```
