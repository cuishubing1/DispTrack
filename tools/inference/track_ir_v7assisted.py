"""track_ir_v7assisted.py — ALPF: DualDecoderDEIM 双输出 + TID-assisted tracking"""
import os
import sys
import json
import argparse
import torch
import torch.nn as nn
from collections import defaultdict
import torchvision.transforms as T
from PIL import Image
import numpy as np
import time
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), './')))
from engine.core import YAMLConfig
from engine.logger_module import get_logger
from det_cache_utils import build_results_from_rows, infer_sequence_name, load_detection_cache
logger = get_logger(__name__)

class SingleModel(nn.Module):
    """Single-model wrapper for inference."""

    def __init__(self, cfg):
        super().__init__()
        self.model = cfg.model.deploy()
        self.postprocessor = cfg.postprocessor.deploy()
        self.query_features = None

    def forward(self, images, orig_target_sizes):
        outputs = self.model(images)
        query_feats = outputs.pop('query_features', None)
        if query_feats is not None:
            import torch.nn.functional as _F
            logits = outputs['pred_logits']
            sort_scores = _F.sigmoid(logits).flatten(1)
            _, sort_idx = sort_scores.topk(sort_scores.shape[1], dim=-1)
            query_idx = sort_idx // logits.shape[-1]
            feat_dim = query_feats.shape[-1]
            query_feats = query_feats.gather(1, query_idx.unsqueeze(-1).expand(-1, -1, feat_dim))
        self.query_features = query_feats
        return self.postprocessor(outputs, orig_target_sizes)

class DualModalModel(nn.Module):
    """Dual-modal model wrapper (e.g. v7/v12) for inference."""

    def __init__(self, cfg):
        super().__init__()
        self.model = cfg.model.deploy()
        self.postprocessor = cfg.postprocessor.deploy()
        self.query_features = None
        self.ir_query_features = None

    def forward(self, img_rgb, img_ir, orig_target_sizes, return_branches=False):
        outputs = self.model(img_rgb, img_ir)
        query_feats = outputs.pop('query_features', None)
        ir_query_feats = outputs.pop('ir_query_features', None)
        if query_feats is not None:
            import torch.nn.functional as _F
            logits = outputs['pred_logits']
            sort_scores = _F.sigmoid(logits).flatten(1)
            _, sort_idx = sort_scores.topk(sort_scores.shape[1], dim=-1)
            query_idx = sort_idx // logits.shape[-1]
            feat_dim = query_feats.shape[-1]
            query_feats = query_feats.gather(1, query_idx.unsqueeze(-1).expand(-1, -1, feat_dim))
        self.query_features = query_feats
        if ir_query_feats is not None and 'ir_pred_logits' in outputs:
            import torch.nn.functional as _F
            ir_logits = outputs['ir_pred_logits']
            ir_sort_scores = _F.sigmoid(ir_logits).flatten(1)
            _, ir_sort_idx = ir_sort_scores.topk(ir_sort_scores.shape[1], dim=-1)
            ir_query_idx = ir_sort_idx // ir_logits.shape[-1]
            feat_dim = ir_query_feats.shape[-1]
            ir_query_feats = ir_query_feats.gather(1, ir_query_idx.unsqueeze(-1).expand(-1, -1, feat_dim))
        self.ir_query_features = ir_query_feats
        fusion_results = self.postprocessor(outputs, orig_target_sizes)
        if not return_branches:
            return fusion_results
        branch_outputs = {'fusion': fusion_results}
        if 'ir_pred_logits' in outputs and 'ir_pred_boxes' in outputs:
            ir_outputs = {'pred_logits': outputs['ir_pred_logits'], 'pred_boxes': outputs['ir_pred_boxes']}
            branch_outputs['ir'] = self.postprocessor(ir_outputs, orig_target_sizes)
        return branch_outputs

def load_model(config_path, checkpoint_path, device, model_type='single'):
    """Load a model from config and checkpoint."""
    cfg = YAMLConfig(config_path, resume=checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'ema' in checkpoint:
        state = checkpoint['ema']['module']
    elif 'model' in checkpoint:
        state = checkpoint['model']
    else:
        state = checkpoint
    cfg.model.load_state_dict(state, strict=False)
    if model_type == 'dual':
        model = DualModalModel(cfg).to(device)
    else:
        model = SingleModel(cfg).to(device)
    model.eval()
    return model

def parse_results(outputs):
    """Parse model outputs into (labels, boxes, scores) tuple."""
    if isinstance(outputs, list) and isinstance(outputs[0], dict):
        res = outputs[0]
        labels = res['labels'].unsqueeze(0)
        boxes = res['boxes'].unsqueeze(0)
        scores = res['scores'].unsqueeze(0)
        return (labels, boxes, scores)
    elif isinstance(outputs, tuple):
        return outputs
    else:
        raise ValueError(f'Unknown output format: {type(outputs)}')

def get_sorted_images(directory):
    """Get sorted list of image files from a directory."""
    exts = ('.jpg', '.png', '.jpeg', '.bmp', '.tif', '.tiff')
    img_dir = os.path.join(directory, 'img1')
    if os.path.isdir(img_dir):
        directory = img_dir
    files = []
    for f in os.listdir(directory):
        if f.lower().endswith(exts):
            files.append(os.path.join(directory, f))
    files.sort()
    return files

def _bbox_iou_batch_local(boxes1, boxes2):
    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.zeros((len(boxes1), len(boxes2)), dtype=np.float32)
    boxes1 = np.asarray(boxes1, dtype=np.float32)
    boxes2 = np.asarray(boxes2, dtype=np.float32)
    x1 = np.maximum(boxes1[:, None, 0], boxes2[None, :, 0])
    y1 = np.maximum(boxes1[:, None, 1], boxes2[None, :, 1])
    x2 = np.minimum(boxes1[:, None, 2], boxes2[None, :, 2])
    y2 = np.minimum(boxes1[:, None, 3], boxes2[None, :, 3])
    inter_w = np.maximum(0.0, x2 - x1)
    inter_h = np.maximum(0.0, y2 - y1)
    inter = inter_w * inter_h
    area1 = np.maximum(0.0, boxes1[:, 2] - boxes1[:, 0]) * np.maximum(0.0, boxes1[:, 3] - boxes1[:, 1])
    area2 = np.maximum(0.0, boxes2[:, 2] - boxes2[:, 0]) * np.maximum(0.0, boxes2[:, 3] - boxes2[:, 1])
    union = area1[:, None] + area2[None, :] - inter
    union = np.maximum(union, 1e-06)
    return inter / union

def main(args):
    device = torch.device(args.device)
    ir_cache = load_detection_cache(args.det_cache_ir) if args.det_cache_ir else None
    aux_cache = load_detection_cache(args.det_cache_aux) if args.det_cache_aux else None
    ir_sequence_name = infer_sequence_name(args.input_ir)
    if args.aux_model_type == 'dual' or args.aux_input == 'rgb':
        aux_sequence_name = infer_sequence_name(args.input_rgb)
    else:
        aux_sequence_name = infer_sequence_name(args.input_ir)
    if args.disable_online_bonus:
        effective_tid_bonus = 0.0
    else:
        effective_tid_bonus = args.v7_tid_bonus
    if args.main_det_source == 'aux-ir' and args.aux_model_type != 'dual':
        raise ValueError('--main-det-source aux-ir 仅支持 dual 辅助模型')
    if args.main_det_source == 'ir-model' and (not args.config_ir or not args.resume_ir):
        raise ValueError('--main-det-source ir-model 时必须提供 --config-ir 和 --resume-ir')
    if args.disable_rpmatcher and (not args.disable_tid_propagation):
        raise ValueError('关闭 RPMatcher 时也必须关闭 TID propagation')
    aux_model = None
    if aux_cache is not None:
        if aux_sequence_name not in aux_cache:
            raise KeyError(f'Sequence {aux_sequence_name} not found in auxiliary cache {args.det_cache_aux}')
        print(f'Using auxiliary detection cache: {args.det_cache_aux}')
    else:
        print(f'Loading auxiliary model ({args.aux_name}, type={args.aux_model_type})...')
        aux_model = load_model(args.config_v7, args.resume_v7, device, args.aux_model_type)
    ir_model = None
    if args.main_det_source == 'ir-model':
        if ir_cache is not None:
            if ir_sequence_name not in ir_cache:
                raise KeyError(f'Sequence {ir_sequence_name} not found in IR cache {args.det_cache_ir}')
            print(f'Using IR detection cache: {args.det_cache_ir}')
        else:
            print('Loading IR single-modal model...')
            ir_model = load_model(args.config_ir, args.resume_ir, device, 'single')
    elif ir_cache is not None:
        if ir_sequence_name not in ir_cache:
            raise KeyError(f'Sequence {ir_sequence_name} not found in IR cache {args.det_cache_ir}')
        print(f'Using cached {args.aux_name} IR-branch detections: {args.det_cache_ir}')
    else:
        print(f'Using {args.aux_name} IR branch as main tracker detections')
    from track import Track, ALPFTrack, _bbox_iou_batch
    from relative_position_matcher import RelativePositionMatcher
    from collections import Counter
    aux_tracker = Track(det_thresh=args.det_thresh, track_thresh=args.track_thresh, match_thresh=args.match_thresh, track_buffer=args.track_buffer, frame_rate=args.frame_rate, min_conf=args.min_conf, max_age=args.max_age, min_hits=args.min_hits, iou_threshold=args.iou_threshold, score_ema_alpha=args.score_ema, jump_detect=args.jump_detect, jump_extra_buffer=args.jump_extra_buffer, recovery=False)
    ir_tracker = ALPFTrack(det_thresh=args.det_thresh, track_thresh=args.track_thresh, match_thresh=args.match_thresh, track_buffer=args.track_buffer, frame_rate=args.frame_rate, min_conf=args.min_conf, max_age=args.max_age, min_hits=args.min_hits, iou_threshold=args.iou_threshold, score_ema_alpha=args.score_ema, jump_detect=args.jump_detect, jump_extra_buffer=args.jump_extra_buffer, recovery=args.recovery, recovery_min_score=args.recovery_min_score, recovery_iou_thresh=args.recovery_iou_thresh, recovery_score=args.recovery_score, recovery_max_age=args.recovery_max_age, v7_tid_bonus=effective_tid_bonus, v7_tid_history_len=args.v7_tid_history_len, v7_tid_majority_ratio=args.v7_tid_majority_ratio, adaptive_min_hits=args.adaptive_minhits, amh_confirmed=args.amh_confirmed, amh_unconfirmed=args.amh_unconfirmed, mechanism_debug_jsonl=None, mechanism_near_tie_thresh=args.near_tie_thresh)
    matcher = RelativePositionMatcher(offset_bin_size=args.offset_bin_size, n_neighbors=args.n_neighbors, desc_match_thresh=args.desc_match_thresh, center_dist_thresh=args.center_dist_thresh, min_dets_for_structure=args.min_dets_for_structure, variant=args.matcher_variant)
    main_source_desc = 'ir-model' if args.main_det_source == 'ir-model' else f'{args.aux_name}-ir'
    print(f'[{main_source_desc}+{args.aux_name} Assisted] tid_bonus={effective_tid_bonus}, history_len={args.v7_tid_history_len}, majority_ratio={args.v7_tid_majority_ratio}')
    print(f"[Main tracker] type={'bytetrack'}")
    print(f"[Mechanism] matcher={('off' if args.disable_rpmatcher else 'on')}, tid_propagation={('off' if args.disable_tid_propagation else 'on')}, online_bonus={('off' if args.disable_online_bonus else 'on')}, offline_merge={('off' if args.disable_offline_merge else 'on')}")
    if args.adaptive_minhits:
        print(f'[Adaptive min_hits] ENABLED: v7_confirmed→{args.amh_confirmed}, unconfirmed→{args.amh_unconfirmed}')
        print(f'[Matcher] variant={args.matcher_variant}, offset_bin={args.offset_bin_size}, center_dist_thresh={args.center_dist_thresh}')
    rgb_files = get_sorted_images(args.input_rgb)
    ir_files = get_sorted_images(args.input_ir)
    n_frames = min(len(rgb_files), len(ir_files))
    print(f'Found {len(rgb_files)} RGB, {len(ir_files)} IR images → {n_frames} frames')
    if n_frames == 0:
        print('Error: No images found')
        return
    transforms_rgb = T.Compose([T.Resize((640, 640)), T.ToTensor()])
    transforms_ir = T.Compose([T.Resize((640, 640)), T.ToTensor()])
    os.makedirs(args.output_dir, exist_ok=True)
    txt_file = None
    near_tie_debug_path = None
    if args.save_txt:
        res_dir = os.path.join(args.output_dir, 'track_results')
        os.makedirs(res_dir, exist_ok=True)
        txt_path = os.path.join(res_dir, 'results.txt')
        txt_file = open(txt_path, 'w')
        print(f'Saving tracking results to {txt_path}')
    track_frames = {}
    for frame_id in range(n_frames):
        t0 = time.time()
        rgb_pil = Image.open(rgb_files[frame_id]).convert('RGB')
        ir_pil = Image.open(ir_files[frame_id]).convert('L')
        w, h = ir_pil.size
        orig_sizes = torch.tensor([[w, h]], device=device)
        rgb_tensor = None
        ir_tensor = None
        ir_query_features_np = None
        aux_outputs_shared = None
        if aux_cache is not None:
            aux_results = build_results_from_rows(aux_cache[aux_sequence_name].get(frame_id + 1, []), device)
        else:
            rgb_tensor = transforms_rgb(rgb_pil).unsqueeze(0).to(device)
            ir_tensor = transforms_ir(ir_pil).unsqueeze(0).to(device)
        if ir_cache is not None:
            ir_results = build_results_from_rows(ir_cache[ir_sequence_name].get(frame_id + 1, []), device)
        else:
            if rgb_tensor is None:
                rgb_tensor = transforms_rgb(rgb_pil).unsqueeze(0).to(device)
            if ir_tensor is None:
                ir_tensor = transforms_ir(ir_pil).unsqueeze(0).to(device)
            with torch.no_grad():
                if args.main_det_source == 'aux-ir':
                    aux_outputs_shared = aux_model(rgb_tensor, ir_tensor, orig_sizes, return_branches=True)
                    if 'ir' not in aux_outputs_shared:
                        raise RuntimeError(f'Auxiliary model {args.aux_name} does not expose IR branch outputs')
                    ir_results = parse_results(aux_outputs_shared['ir'])
                    if aux_cache is None:
                        aux_results = parse_results(aux_outputs_shared['fusion'])
                else:
                    ir_outputs = ir_model(ir_tensor, orig_sizes)
                    ir_results = parse_results(ir_outputs)
        if aux_cache is None:
            if aux_outputs_shared is not None:
                pass
            elif rgb_tensor is None:
                rgb_tensor = transforms_rgb(rgb_pil).unsqueeze(0).to(device)
            if aux_outputs_shared is None and ir_tensor is None:
                ir_tensor = transforms_ir(ir_pil).unsqueeze(0).to(device)
            if aux_outputs_shared is None:
                with torch.no_grad():
                    if args.aux_model_type == 'dual':
                        aux_outputs = aux_model(rgb_tensor, ir_tensor, orig_sizes)
                    else:
                        aux_input = rgb_tensor if args.aux_input == 'rgb' else ir_tensor
                        aux_outputs = aux_model(aux_input, orig_sizes)
                    aux_results = parse_results(aux_outputs)
        rgb_np = np.array(rgb_pil)
        aux_img_np = rgb_np if args.aux_input == 'rgb' else np.array(ir_pil.convert('RGB'))
        aux_tracks = aux_tracker(aux_img_np, aux_results)
        aux_mot_raw = aux_tracker.transform_mot(*aux_results)
        aux_mask = aux_mot_raw[:, 4] >= args.det_thresh if len(aux_mot_raw) > 0 else np.array([], dtype=bool)
        aux_filtered = aux_mot_raw[aux_mask] if len(aux_mot_raw) > 0 else aux_mot_raw
        aux_det_to_tid = {}
        if len(aux_tracks) > 0 and len(aux_filtered) > 0:
            trk_boxes = aux_tracks[:, :4].numpy() if isinstance(aux_tracks, torch.Tensor) else aux_tracks[:, :4]
            ious = _bbox_iou_batch(trk_boxes, aux_filtered[:, :4])
            for ti in range(len(trk_boxes)):
                best_di = np.argmax(ious[ti])
                if ious[ti, best_di] > 0.5:
                    aux_det_to_tid[best_di] = int(aux_tracks[ti, 4])
        ir_mot_raw = ir_tracker.transform_mot(*ir_results)
        ir_match_thresh = args.det_thresh
        ir_mask = ir_mot_raw[:, 4] >= ir_match_thresh if len(ir_mot_raw) > 0 else np.array([], dtype=bool)
        ir_filtered = ir_mot_raw[ir_mask] if len(ir_mot_raw) > 0 else ir_mot_raw
        ir_filt_idx = np.where(ir_mask)[0] if len(ir_mot_raw) > 0 else np.array([], dtype=int)
        if args.disable_rpmatcher:
            matched = []
            unmatched_aux = list(range(len(aux_filtered)))
            unmatched_ir = list(range(len(ir_filtered)))
        else:
            matched, unmatched_aux, unmatched_ir = matcher(aux_filtered, ir_filtered)
        aux_det_tids = np.full(len(ir_mot_raw), -1, dtype=int)
        for fi_aux, fi_ir, dist in matched:
            v7_tid = aux_det_to_tid.get(fi_aux, -1)
            orig_ir_idx = ir_filt_idx[fi_ir]
            aux_det_tids[orig_ir_idx] = v7_tid
        propagated_v7_det_tids = None if args.disable_tid_propagation else aux_det_tids
        ir_np = np.array(ir_pil.convert('RGB'))
        track_results = ir_tracker(ir_np, ir_results, v7_det_tids=propagated_v7_det_tids)
        if txt_file and len(track_results) > 0:
            for trk in track_results:
                if len(trk) < 5:
                    continue
                x1, y1, x2, y2 = trk[:4].tolist()
                track_id = int(trk[4])
                conf = trk[5].item() if len(trk) > 5 else 1.0
                cls_id = int(trk[6]) if len(trk) > 6 else -1
                w_box = x2 - x1
                h_box = y2 - y1
                line = f'{frame_id + 1},{track_id},{x1:.2f},{y1:.2f},{w_box:.2f},{h_box:.2f},{conf:.2f},{cls_id},-1,-1\n'
                txt_file.write(line)
        for trk in track_results:
            if len(trk) < 5:
                continue
            ir_tid = int(trk[4])
            if ir_tid not in track_frames:
                track_frames[ir_tid] = set()
            track_frames[ir_tid].add(frame_id)
        t1 = time.time()
        n_trk = len(track_results)
        n_matched = len(matched)
        n_with_aux = 0 if propagated_v7_det_tids is None else int(np.sum(propagated_v7_det_tids >= 0))
        if frame_id % 50 == 0:
            fps_val = 1.0 / (t1 - t0 + 1e-06)
            print(f'  Frame {frame_id:4d}: {n_trk} tracks, {len(ir_filtered)} ir_det, {len(aux_filtered)} {args.aux_name}_det, {n_matched} matched, {n_with_aux} with_aux_tid, {fps_val:.1f} FPS')
    if txt_file:
        txt_file.close()
    tid_mapping = ir_tracker.get_all_tid_mappings()
    tid_history = ir_tracker._tid_history
    merge_map = {}
    if not args.disable_tid_propagation and (not args.disable_offline_merge) and tid_mapping and txt_file:
        strong_mapping = {}
        for ir_tid, v7_tid in tid_mapping.items():
            hist = tid_history.get(ir_tid, [])
            if len(hist) >= 1:
                vote_count = Counter(hist).most_common(1)[0][1]
                confidence = vote_count / len(hist)
                if confidence >= args.v7_tid_majority_ratio:
                    strong_mapping[ir_tid] = v7_tid
        print(f'\n  [Debug] tid_mapping: {len(tid_mapping)} entries')
        print(f'  [Debug] strong_mapping: {len(strong_mapping)} entries')
        hist_lens = {t: len(tid_history.get(t, [])) for t in tid_mapping}
        print(f'  [Debug] tid history lengths: {hist_lens}')
        v7_to_ir = {}
        for ir_tid, v7_tid in strong_mapping.items():
            v7_to_ir.setdefault(v7_tid, []).append(ir_tid)
        for v7_tid, ir_tids in v7_to_ir.items():
            if len(ir_tids) <= 1:
                continue
            ir_tids.sort(key=lambda t: min(track_frames.get(t, {0})))
            base_tid = ir_tids[0]
            for other_tid in ir_tids[1:]:
                base_frames = track_frames.get(base_tid, set())
                other_frames = track_frames.get(other_tid, set())
                overlap = base_frames & other_frames
                if len(overlap) == 0:
                    merge_map[other_tid] = base_tid
                    track_frames.setdefault(base_tid, set()).update(other_frames)
        for tid in list(merge_map.keys()):
            target = merge_map[tid]
            while target in merge_map:
                target = merge_map[target]
            merge_map[tid] = target
    n_merges = 0
    if merge_map and args.save_txt:
        txt_path = os.path.join(args.output_dir, 'track_results', 'results.txt')
        with open(txt_path, 'r') as f:
            lines = f.readlines()
        with open(txt_path, 'w') as f:
            for line in lines:
                parts = line.strip().split(',')
                if len(parts) >= 2:
                    old_tid = int(parts[1])
                    if old_tid in merge_map:
                        parts[1] = str(merge_map[old_tid])
                        n_merges += 1
                f.write(','.join(parts) + '\n')
        print(f'\n  [Post-processing] Merged {len(merge_map)} track IDs, {n_merges} lines rewritten')
        print(f'  Merge map: {merge_map}')
        print(f'  Strong {args.aux_name} mappings: {strong_mapping}')
    elif args.disable_tid_propagation:
        print('\n  [Post-processing] TID propagation disabled')
    elif args.disable_offline_merge:
        print('\n  [Post-processing] Offline merge disabled')
    print(f"\n{'=' * 60}")
    print(f'  {main_source_desc}+{args.aux_name} Assisted Tracking Summary')
    print(f"{'=' * 60}")
    ir_stats = ir_tracker.get_stats()
    v7_stats = aux_tracker.get_stats()
    matcher_stats = matcher.get_stats() if hasattr(matcher, 'get_stats') else matcher._stats
    print(f'  Total frames: {n_frames}')
    print(f'  IR tracker: {ir_stats}')
    print(f'  {args.aux_name} tracker: {v7_stats}')
    print(f'  Matcher: {matcher_stats}')
    print(f'  auxiliary tid mappings: {tid_mapping}')
    if merge_map:
        print(f'  ID merges: {merge_map}')
    print(f"{'=' * 60}")
    print('Done')
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='IR 主体跟踪 + 辅助分支 Track ID 辅助')
    parser.add_argument('--config-ir', help='IR 模型 config')
    parser.add_argument('--resume-ir', help='IR 模型 checkpoint')
    parser.add_argument('--config-v7', required=True, help='辅助模型 config')
    parser.add_argument('--resume-v7', required=True, help='辅助模型 checkpoint')
    parser.add_argument('--aux-name', default='v7', help='辅助模型名称 (用于日志)')
    parser.add_argument('--aux-model-type', choices=['single', 'dual'], default='dual', help='辅助模型类型 (single/dual)')
    parser.add_argument('--aux-input', choices=['rgb', 'ir'], default='rgb', help='single 辅助模型使用哪一路输入')
    parser.add_argument('--main-det-source', choices=['ir-model', 'aux-ir'], default='aux-ir', help='主 tracker 检测来源: 独立 IR 模型 或 dual 辅助模型内部 IR 分支')
    parser.add_argument('--input-rgb', required=True, help='RGB 图像目录')
    parser.add_argument('--input-ir', required=True, help='IR 图像目录')
    parser.add_argument('--output-dir', required=True, help='输出目录')
    parser.add_argument('--save-txt', action='store_true', help='保存 MOT 格式结果')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--det-cache-ir', default=None, help='IR branch detection cache CSV')
    parser.add_argument('--det-cache-aux', default=None, help='auxiliary branch detection cache CSV')
    parser.add_argument('--det-thresh', default=0.2, type=float)
    parser.add_argument('--track-thresh', default=0.5, type=float)
    parser.add_argument('--match-thresh', default=0.8, type=float)
    parser.add_argument('--track-buffer', default=30, type=int)
    parser.add_argument('--frame-rate', default=30, type=int)
    parser.add_argument('--min-conf', default=0.1, type=float)
    parser.add_argument('--max-age', default=30, type=int)
    parser.add_argument('--min-hits', default=2, type=int)
    parser.add_argument('--iou-threshold', default=0.3, type=float)
    parser.add_argument('--score-ema', default=0.0, type=float)
    parser.add_argument('--jump-detect', action='store_true')
    parser.add_argument('--jump-extra-buffer', default=30, type=int)
    parser.add_argument('--recovery', action='store_true')
    parser.add_argument('--recovery-min-score', default=0.4, type=float)
    parser.add_argument('--recovery-iou-thresh', default=0.3, type=float)
    parser.add_argument('--recovery-score', default=0.25, type=float)
    parser.add_argument('--recovery-max-age', default=5, type=int)
    parser.add_argument('--v7-tid-bonus', default=0.35, type=float, help='auxiliary TID consistency bonus (default: 0.35)')
    parser.add_argument('--v7-tid-history-len', default=1, type=int, help='propagated TID window; 1=last-tid (default: 1)')
    parser.add_argument('--v7-tid-majority-ratio', default=0.5, type=float, help='v7 tid 多数投票生效阈值 (default: 0.5)')
    parser.add_argument('--disable-online-bonus', action='store_true', help='关闭 online cost injection，仅保留辅助 tid 统计/后处理')
    parser.add_argument('--disable-offline-merge', action='store_true', help='关闭 offline ID merge，仅保留 online 关联引导')
    parser.add_argument('--disable-rpmatcher', action='store_true', help='关闭 RPMatcher，不进行跨模态 detection 匹配')
    parser.add_argument('--disable-tid-propagation', action='store_true', help='关闭辅助 TID 向主 tracker 的传播')
    parser.add_argument('--adaptive-minhits', action='store_true', help='启用 v7 确认自适应 min_hits')
    parser.add_argument('--amh-confirmed', default=1, type=int, help='v7 确认时的 min_hits (default: 1)')
    parser.add_argument('--amh-unconfirmed', default=3, type=int, help='无 v7 确认时的 min_hits (default: 3)')
    parser.add_argument('--offset-bin-size', default=10.0, type=float)
    parser.add_argument('--n-neighbors', default=4, type=int)
    parser.add_argument('--desc-match-thresh', default=100.0, type=float)
    parser.add_argument('--center-dist-thresh', default=30.0, type=float)
    parser.add_argument('--min-dets-for-structure', default=3, type=int)
    parser.add_argument('--matcher-variant', default='full', type=str, choices=['full', 'no-hist', 'no-shape', 'no-fallback', 'center-only', 'nn'])
    parser.add_argument('--near-tie-thresh', default=0.05, type=float, help='near-tie 判定阈值: 前两候选代价差小于该值')
    args = parser.parse_args()
    main(args)
