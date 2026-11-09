"""Single-device DispTrack training using the released paired-data and DEIM modules."""
import argparse
import copy
import json
import math
from pathlib import Path
import random
import re
import numpy as np
import torch

def checkpoint_model(path):
    state = torch.load(path, map_location='cpu', weights_only=False)
    if 'ema' in state:
        return state['ema']['module']
    return state.get('model', state)

def initialize(model, fusion_path, ir_path):
    current = model.state_dict()
    fusion = checkpoint_model(fusion_path)
    ir = checkpoint_model(ir_path)
    fusion = {k.replace('decoder.', 'decoder_rgb.', 1) if k.startswith('decoder.') else k: v for k, v in fusion.items()}
    fused_prefixes = ('backbone_rgb.', 'fusion.', 'encoder.', 'decoder_rgb.')
    mapped = {k: v for k, v in fusion.items() if k.startswith(fused_prefixes)}
    for source, target in [('backbone.', 'backbone_ir.'), ('encoder.', 'encoder_ir.'), ('decoder.', 'decoder_ir.')]:
        mapped.update({target + k[len(source):]: v for k, v in ir.items() if k.startswith(source)})
    missing = [k for k, p in model.named_parameters() if k not in mapped or mapped[k].shape != p.shape]
    if missing:
        raise ValueError(f'Initialization is incomplete for {len(missing)} parameters: {missing[:8]}')
    compatible = {k: v for k, v in mapped.items() if k in current and v.shape == current[k].shape}
    missing_buffers = [k for k in current if k not in compatible and (not k.endswith('num_batches_tracked'))]
    if missing_buffers:
        raise ValueError(f'Initialization lacks model state: {missing_buffers[:8]}')
    model.load_state_dict(compatible, strict=False)

def freeze_ir(model):
    for name in ('backbone_ir', 'encoder_ir', 'decoder_ir'):
        module = getattr(model, name)
        module.requires_grad_(False)
        module.eval()

def optimizer_for(model, config):
    config = copy.deepcopy(config)
    config.pop('type')
    patterns = config.pop('params', [])
    groups, assigned = ([], set())
    for pattern in patterns:
        expression = pattern.pop('params')
        selected = []
        for name, parameter in model.named_parameters():
            if parameter.requires_grad and re.search(expression, name):
                if name in assigned:
                    raise ValueError(f'Overlapping optimizer groups: {name}')
                selected.append(parameter)
                assigned.add(name)
        if selected:
            groups.append({'params': selected, **pattern})
    rest = [p for n, p in model.named_parameters() if p.requires_grad and n not in assigned]
    if rest:
        groups.append({'params': rest})
    return torch.optim.AdamW(groups, **config)

@torch.no_grad()
def evaluate(model, postprocessor, loader, device):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    model.eval()
    predictions = []
    for pair, targets in loader:
        outputs = model(pair[0].to(device), pair[1].to(device))
        sizes = torch.stack([target['orig_size'] for target in targets]).to(device)
        processed = postprocessor(outputs, sizes)
        for target, result in zip(targets, processed):
            boxes = result['boxes'].cpu()
            boxes[:, 2:] -= boxes[:, :2]
            for box, label, score in zip(boxes.tolist(), result['labels'].tolist(), result['scores'].tolist()):
                predictions.append({'image_id': int(target['image_id'].item()), 'category_id': int(label), 'bbox': box, 'score': float(score)})
    if not predictions:
        return 0.0
    truth = COCO(loader.dataset.ann_file)
    measured = COCOeval(truth, truth.loadRes(predictions), 'bbox')
    measured.params.imgIds = list(loader.dataset.ids)
    measured.evaluate()
    measured.accumulate()
    measured.summarize()
    return float(measured.stats[0])

def main(args):
    from engine.core import YAMLConfig
    import engine.data
    import engine.deim.matcher
    import engine.deim.deim_criterion
    from engine.optim.ema import ModelEMA
    from engine.optim.schedule import flat_cosine_schedule
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    cfg = YAMLConfig(str(args.config))
    for key, split in [('train_dataloader', 'train'), ('val_dataloader', args.selection_split)]:
        dataset = cfg.yaml_cfg[key]['dataset']
        dataset.update(img_folder_rgb=str(args.data_root / '1/rgb' / split), img_folder_ir=str(args.data_root / '1/ir' / split), ann_file=str(args.data_root / 'Annotations/1/rgb' / f'{split}_cocoformat.json'), ann_file_ir=str(args.data_root / 'Annotations/1/ir' / f'{split}_cocoformat.json'))
    if not args.resume and (not args.fusion_init or not args.ir_init):
        raise ValueError('Provide --fusion-init and --ir-init, or --resume')
    model = cfg.model
    state = None
    if args.resume:
        state = torch.load(args.resume, map_location='cpu', weights_only=False)
        model.load_state_dict(state['model'])
    else:
        initialize(model, args.fusion_init, args.ir_init)
    model.to(device)
    freeze_ir(model)
    criterion = cfg.criterion.to(device)
    optimizer = optimizer_for(model, cfg.yaml_cfg['optimizer'])
    amp = bool(cfg.yaml_cfg.get('use_amp', True) and device.type == 'cuda')
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    ema = ModelEMA(model, **{k: v for k, v in cfg.yaml_cfg.get('ema', {}).items() if k != 'type'})
    start_epoch, best = (0, -math.inf)
    if state:
        optimizer.load_state_dict(state['optimizer'])
        scaler.load_state_dict(state['scaler'])
        ema.load_state_dict(state['ema'])
        start_epoch, best = (state['epoch'] + 1, state['best_ap'])
        random.setstate(state['rng_python'])
        np.random.set_state(state['rng_numpy'])
        torch.set_rng_state(state['rng_torch'])
        if device.type == 'cuda' and state.get('rng_cuda'):
            torch.cuda.set_rng_state_all(state['rng_cuda'])
    train_loader, selection_loader = (cfg.train_dataloader, cfg.val_dataloader)
    if not len(train_loader) or not len(selection_loader):
        raise ValueError('Training and selection loaders must be nonempty')
    args.output.mkdir(parents=True, exist_ok=True)
    settings = cfg.yaml_cfg
    (args.output / 'config.json').write_text(json.dumps(settings, indent=2), encoding='utf-8')
    iterations = len(train_loader)
    total = settings['epoches'] * iterations
    base_lrs = [group['lr'] for group in optimizer.param_groups]
    if state:
        base_lrs = state['base_lrs']
    for epoch in range(start_epoch, settings['epoches']):
        train_loader.set_epoch(epoch)
        model.train()
        freeze_ir(model)
        criterion.train()
        for step, (pair, targets) in enumerate(train_loader):
            optimizer.zero_grad(set_to_none=True)
            targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
            with torch.autocast(device_type=device.type, enabled=amp):
                outputs = model(pair[0].to(device), pair[1].to(device), targets=targets)
            with torch.autocast(device_type=device.type, enabled=False):
                losses = criterion(outputs, targets, epoch=epoch, step=step, global_step=epoch * iterations + step, epoch_step=iterations)
                loss = sum(losses.values())
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite loss at epoch {epoch}, step {step}')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), settings.get('clip_max_norm', 0.1))
            scaler.step(optimizer)
            scaler.update()
            ema.update(model)
            for group, base in zip(optimizer.param_groups, base_lrs):
                group['lr'] = flat_cosine_schedule(total, settings['warmup_iter'], settings['flat_epoch'] * iterations, settings['no_aug_epoch'] * iterations, epoch * iterations + step, base, base * settings['lr_gamma'])
            if step % 20 == 0:
                print(f'epoch={epoch} step={step} loss={loss.item():.6f}', flush=True)
        ap = evaluate(ema.module, cfg.postprocessor.to(device), selection_loader, device)
        improved = ap > best
        best = max(best, ap)
        saved = {'model': model.state_dict(), 'ema': ema.state_dict(), 'optimizer': optimizer.state_dict(), 'scaler': scaler.state_dict(), 'epoch': epoch, 'best_ap': best, 'base_lrs': base_lrs, 'rng_python': random.getstate(), 'rng_numpy': np.random.get_state(), 'rng_torch': torch.get_rng_state(), 'rng_cuda': torch.cuda.get_rng_state_all() if device.type == 'cuda' else None}
        torch.save(saved, args.output / 'last.pth')
        if improved:
            torch.save(saved, args.output / 'best.pth')
        print(f'epoch={epoch} selection_AP={ap:.6f} best_AP={best:.6f}', flush=True)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-c', '--config', type=Path, default=Path(__file__).parent / 'configs/train/disptrack.yml')
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--fusion-init', type=Path)
    parser.add_argument('--ir-init', type=Path)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--selection-split', choices=['val', 'test'], default='test')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', type=int, default=0)
    main(parser.parse_args())
