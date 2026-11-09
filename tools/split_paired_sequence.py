"""Apply a documented frozen-frame cut to a raw RGB/IR pair, writing a new tree."""
import argparse
import configparser
import json
from pathlib import Path
import shutil

def read_sequence(directory):
    config = configparser.ConfigParser()
    config.read(directory / 'seqinfo.ini', encoding='utf-8-sig')
    info = config['Sequence']
    image_dir = (directory / info['imDir']).resolve()
    if not image_dir.is_relative_to(directory.resolve()):
        raise ValueError('imDir must stay inside the source sequence')
    rows = []
    length = int(info['seqLength'])
    seen = set()
    for line in (directory / 'gt/gt.txt').read_text(encoding='utf-8-sig').splitlines():
        if not line.strip():
            continue
        fields = line.split(',')
        if len(fields) < 6:
            raise ValueError(f'Malformed GT row in {directory}')
        frame, track = (int(fields[0]), int(fields[1]))
        if not 1 <= frame <= length or (frame, track) in seen:
            raise ValueError(f'Invalid or duplicate frame/track in {directory}: {frame}/{track}')
        seen.add((frame, track))
        rows.append((frame, track, fields[2:]))
    for frame in range(1, length + 1):
        if not (image_dir / f"{frame:06d}{info['imExt']}").is_file():
            raise FileNotFoundError(f'Missing source image: {directory}, frame {frame}')
    return (config, image_dir, rows)

def split_pair(rgb_dir, ir_dir, output, rule):
    rgb_dir, ir_dir, output = [p.resolve() for p in (rgb_dir, ir_dir, output)]
    if rgb_dir == ir_dir:
        raise ValueError('RGB and IR inputs must be separate directories')
    for source in (rgb_dir, ir_dir):
        if output.is_relative_to(source) or source.is_relative_to(output):
            raise ValueError('Output and source directory trees must not overlap')
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite output: {output}')
    inputs = {'rgb': read_sequence(rgb_dir), 'ir': read_sequence(ir_dir)}
    expected_length = max((s['source_frames'][1] for s in rule['segments']))
    for modality, (config, _, _) in inputs.items():
        if int(config['Sequence']['seqLength']) != expected_length:
            raise ValueError(f'{modality}: expected raw length {expected_length}; do not use already-cut data')
    output.mkdir(parents=True, exist_ok=False)
    audit = {'source_rgb_sequence': rgb_dir.name, 'source_ir_sequence': ir_dir.name, 'rule': rule, 'track_id_maps': {}}
    for modality, (config, image_dir, rows) in inputs.items():
        for segment in rule['segments']:
            start, end = segment['source_frames']
            name = segment['target_rgb'] + ('T' if modality == 'ir' else '')
            target = output / modality / name
            (target / 'img1').mkdir(parents=True)
            (target / 'gt').mkdir()
            selected = [(f, t, rest) for f, t, rest in rows if start <= f <= end]
            mapping = {t: i for i, t in enumerate(sorted({t for _, t, _ in selected}), 1)} if segment['remap_track_ids'] else {}
            for frame in range(start, end + 1):
                ext = config['Sequence']['imExt']
                shutil.copy2(image_dir / f'{frame:06d}{ext}', target / 'img1' / f'{frame - start + 1:06d}{ext}')
            text = '\n'.join((f"{f - start + 1},{mapping.get(t, t)},{','.join(rest)}" for f, t, rest in sorted(selected, key=lambda row: row[0])))
            (target / 'gt/gt.txt').write_text(text + ('\n' if text else ''), encoding='utf-8')
            labels = (rgb_dir if modality == 'rgb' else ir_dir) / 'gt/labels.txt'
            if labels.is_file():
                shutil.copy2(labels, target / 'gt/labels.txt')
            new_config = configparser.ConfigParser()
            new_config['Sequence'] = dict(config['Sequence'])
            new_config['Sequence'].update({'name': name, 'imDir': 'img1', 'seqLength': str(end - start + 1)})
            with (target / 'seqinfo.ini').open('w', encoding='utf-8') as handle:
                new_config.write(handle)
            audit['track_id_maps'][f'{modality}/{name}'] = mapping
    (output / 'cut_manifest.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return audit

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rgb-seq-dir', required=True, type=Path)
    parser.add_argument('--ir-seq-dir', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path, help='New output directory; must not exist')
    parser.add_argument('--rule', required=True, choices=['2-02', '2-06'])
    args = parser.parse_args()
    protocol = Path(__file__).resolve().parents[1] / 'data_protocol/cleaning_rules.json'
    rules = json.loads(protocol.read_text(encoding='utf-8'))['rules']
    rule = next((r for r in rules if r['source_rgb'] == args.rule))
    split_pair(args.rgb_seq_dir, args.ir_seq_dir, args.output, rule)
    print(f'Wrote paired cut to {args.output}; original sequences preserved')
if __name__ == '__main__':
    main()
