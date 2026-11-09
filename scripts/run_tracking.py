"""Launch the released single-checkpoint ALPF preset; dry-run needs only Python."""
import argparse
from pathlib import Path
import subprocess
import sys

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--rgb', required=True, type=Path)
    parser.add_argument('--ir', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--online-only', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    checkpoint, rgb, ir, output = [p.resolve() for p in (args.checkpoint, args.rgb, args.ir, args.output)]
    command = [sys.executable, str(root / 'tools/inference/track_ir_v7assisted.py'), '--config-v7', str(root / 'configs/release/deim_hgnetv2_m3ot_dual_v12_infer.yml'), '--resume-v7', str(checkpoint), '--aux-name', 'v12', '--aux-model-type', 'dual', '--aux-input', 'rgb', '--main-det-source', 'aux-ir', '--input-rgb', str(rgb), '--input-ir', str(ir), '--output-dir', str(output), '--device', args.device, '--save-txt', '--det-thresh', '0.2', '--track-thresh', '0.5', '--match-thresh', '0.8', '--min-hits', '2', '--v7-tid-bonus', '0.35', '--v7-tid-history-len', '1', '--v7-tid-majority-ratio', '0.5', '--jump-detect', '--recovery']
    if args.online_only:
        command.append('--disable-offline-merge')
    if args.dry_run:
        import json
        print(json.dumps(command, ensure_ascii=False, indent=2))
        return
    for p in (checkpoint, rgb, ir):
        if not p.exists():
            parser.error(f'Required input does not exist: {p}')
    raise SystemExit(subprocess.call(command, cwd=root))
if __name__ == '__main__':
    main()
