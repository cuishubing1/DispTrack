"""MM-ByteTrack: Multi-Modal friendly ByteTrack wrapper."""
import warnings
warnings.filterwarnings('ignore')
import json
from pathlib import Path
import torch
import numpy as np
from boxmot import ByteTrack
import boxmot.trackers.bytetrack.bytetrack as _bt_module
_orig_iou_distance = _bt_module.iou_distance
_orig_fuse_score = _bt_module.fuse_score
_orig_linear_assignment = _bt_module.linear_assignment
_V7TID_CTX = {'active': False, 'det_tids': {}, 'track_tids': {}, 'bonus': 0.15, 'n_bonus': 0, 'frame_id': -1, 'near_tie_thresh': 0.05, 'debug_jsonl': None, 'capture_done': False, 'pending_assoc': None}

def _track_xyxy(track_obj):
    xyxy = getattr(track_obj, 'xyxy', None)
    if xyxy is None:
        return None
    return [float(x) for x in np.asarray(xyxy).tolist()]

def _append_jsonl_event(path, event):
    if not path:
        return
    with open(path, 'a', encoding='utf-8') as handle:
        handle.write(json.dumps(event, ensure_ascii=False) + '\n')

def _append_mechanism_event(event):
    path = _V7TID_CTX.get('debug_jsonl')
    if not path:
        return
    _append_jsonl_event(path, event)

def _v7tid_iou_distance(atracks, btracks):
    """Patched iou_distance: adds v7_tid consistency bonus."""
    pre_cost_matrix = _orig_iou_distance(atracks, btracks)
    cost_matrix = pre_cost_matrix.copy()
    ctx = _V7TID_CTX
    if not ctx['active'] or cost_matrix.size == 0:
        return cost_matrix
    if len(btracks) > 0 and hasattr(btracks[0], 'is_activated') and btracks[0].is_activated:
        return cost_matrix
    bonus = ctx['bonus']
    det_tids = ctx['det_tids']
    track_tids = ctx['track_tids']
    for i, atrk in enumerate(atracks):
        trk_v7 = track_tids.get(getattr(atrk, 'id', -1), -1)
        if trk_v7 < 0:
            continue
        for j, btrk in enumerate(btracks):
            det_v7 = det_tids.get(int(getattr(btrk, 'det_ind', -1)), -1)
            if det_v7 >= 0 and det_v7 == trk_v7:
                if bonus > 0:
                    cost_matrix[i, j] = max(0.0, cost_matrix[i, j] - bonus)
                    ctx['n_bonus'] += 1
    if not ctx.get('capture_done', False):
        ctx['pending_assoc'] = {'atracks': atracks, 'btracks': btracks, 'pre_cost': pre_cost_matrix.copy(), 'post_cost': cost_matrix.copy(), 'fused_pre': None, 'fused_post': None}
    return cost_matrix

def _v7tid_fuse_score(cost_matrix, detections):
    fused_cost = _orig_fuse_score(cost_matrix, detections)
    ctx = _V7TID_CTX
    pending = ctx.get('pending_assoc')
    if ctx.get('active') and pending and (not ctx.get('capture_done', False)):
        if pending['pre_cost'].shape == cost_matrix.shape:
            pending['fused_pre'] = _orig_fuse_score(pending['pre_cost'].copy(), detections)
            pending['fused_post'] = _orig_fuse_score(pending['post_cost'].copy(), detections)
    return fused_cost

def _v7tid_linear_assignment(cost_matrix, thresh):
    matches, unmatched_a, unmatched_b = _orig_linear_assignment(cost_matrix, thresh)
    ctx = _V7TID_CTX
    pending = ctx.get('pending_assoc')
    if not ctx.get('active') or ctx.get('capture_done', False) or (not pending):
        return (matches, unmatched_a, unmatched_b)
    fused_pre = pending.get('fused_pre')
    fused_post = pending.get('fused_post')
    if fused_pre is None or fused_post is None or fused_pre.shape != cost_matrix.shape:
        return (matches, unmatched_a, unmatched_b)
    matches_pre, _, _ = _orig_linear_assignment(fused_pre, thresh)
    matches_post, _, _ = _orig_linear_assignment(fused_post, thresh)
    pre_match_map = {int(track_idx): int(det_idx) for track_idx, det_idx in np.asarray(matches_pre).reshape(-1, 2)} if np.asarray(matches_pre).size else {}
    post_match_map = {int(track_idx): int(det_idx) for track_idx, det_idx in np.asarray(matches_post).reshape(-1, 2)} if np.asarray(matches_post).size else {}
    near_tie_events = []
    for track_idx in range(fused_pre.shape[0]):
        row_pre = fused_pre[track_idx]
        row_post = fused_post[track_idx]
        if row_pre.size < 2:
            continue
        ranked = np.argsort(row_pre)
        best_idx = int(ranked[0])
        second_idx = int(ranked[1])
        gap = float(row_pre[second_idx] - row_pre[best_idx])
        if gap >= ctx['near_tie_thresh']:
            continue
        pre_det_idx = pre_match_map.get(track_idx, -1)
        post_det_idx = post_match_map.get(track_idx, -1)
        if pre_det_idx == post_det_idx:
            continue
        atrk = pending['atracks'][track_idx]
        trk_id = int(getattr(atrk, 'id', -1))
        trk_tid = int(ctx['track_tids'].get(trk_id, -1))
        candidates = []
        for det_idx in [best_idx, second_idx]:
            btrk = pending['btracks'][det_idx]
            det_ind = int(getattr(btrk, 'det_ind', -1))
            candidates.append({'matrix_det_idx': int(det_idx), 'det_ind': det_ind, 'det_tid': int(ctx['det_tids'].get(det_ind, -1)), 'box': _track_xyxy(btrk), 'score': float(getattr(btrk, 'conf', 0.0)), 'pre_cost': float(row_pre[det_idx]), 'post_cost': float(row_post[det_idx])})
        near_tie_events.append({'frame_id': int(ctx['frame_id']), 'track_idx': int(track_idx), 'track_id': trk_id, 'track_tid': trk_tid, 'track_box': _track_xyxy(atrk), 'gap': gap, 'match_thresh': float(thresh), 'pre_match_det_idx': int(pre_det_idx), 'post_match_det_idx': int(post_det_idx), 'candidates': candidates})
    for event in near_tie_events:
        _append_mechanism_event(event)
    ctx['capture_done'] = True
    ctx['pending_assoc'] = None
    return (matches, unmatched_a, unmatched_b)
_bt_module.iou_distance = _v7tid_iou_distance
_bt_module.fuse_score = _v7tid_fuse_score
_bt_module.linear_assignment = _v7tid_linear_assignment

def _bbox_iou_batch(boxes_a, boxes_b):
    """Compute pairwise IoU between two sets of xyxy boxes."""
    x1 = np.maximum(boxes_a[:, 0:1], boxes_b[:, 0:1].T)
    y1 = np.maximum(boxes_a[:, 1:2], boxes_b[:, 1:2].T)
    x2 = np.minimum(boxes_a[:, 2:3], boxes_b[:, 2:3].T)
    y2 = np.minimum(boxes_a[:, 3:4], boxes_b[:, 3:4].T)
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / (union + 1e-06)

def _bbox_center_distance_batch(boxes_a, boxes_b):
    """Compute pairwise center distance between two sets of xyxy boxes."""
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)
    centers_a = np.stack([(boxes_a[:, 0] + boxes_a[:, 2]) * 0.5, (boxes_a[:, 1] + boxes_a[:, 3]) * 0.5], axis=1)
    centers_b = np.stack([(boxes_b[:, 0] + boxes_b[:, 2]) * 0.5, (boxes_b[:, 1] + boxes_b[:, 3]) * 0.5], axis=1)
    diff = centers_a[:, None, :] - centers_b[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=2, dtype=np.float32))

class Track:
    """Multi-Modal friendly ByteTrack wrapper."""

    def __init__(self, det_thresh: float=0.2, track_thresh: float=0.45, match_thresh: float=0.8, track_buffer: int=30, frame_rate: int=30, min_conf: float=0.1, max_age: int=30, min_hits: int=3, iou_threshold: float=0.3, score_ema_alpha: float=0.0, score_ema_iou_thresh: float=0.5, score_ema_min_score: float=0.15, jump_detect: bool=False, jump_lost_ratio: float=0.8, jump_extra_buffer: int=30, recovery: bool=False, recovery_min_score: float=0.4, recovery_iou_thresh: float=0.3, recovery_score: float=0.25, recovery_max_age: int=5):
        self.track = ByteTrack(det_thresh=det_thresh, track_thresh=track_thresh, match_thresh=match_thresh, track_buffer=track_buffer, frame_rate=frame_rate, min_conf=min_conf, max_age=max_age, min_hits=min_hits, iou_threshold=iou_threshold)
        self._real_min_hits = min_hits
        self.score_ema_alpha = score_ema_alpha
        self.score_ema_iou_thresh = score_ema_iou_thresh
        self.score_ema_min_score = score_ema_min_score
        self._prev_dets = None
        self.jump_detect = jump_detect
        self.jump_lost_ratio = jump_lost_ratio
        self.jump_extra_buffer = jump_extra_buffer
        self._prev_n_tracks = 0
        self._jump_cooldown = 0
        self._base_buffer_size = getattr(self.track, 'buffer_size', int(frame_rate / 30.0 * track_buffer))
        self._frame_idx = 0
        self._total_smoothed = 0
        self.recovery = recovery
        self.recovery_min_score = recovery_min_score
        self.recovery_iou_thresh = recovery_iou_thresh
        self.recovery_score = recovery_score
        self.recovery_max_age = recovery_max_age
        self._trusted_buffer = []
        self._total_recovered = 0

    def transform_mot(self, classes, boxes, scores):
        """Convert (labels, boxes, scores) tensors → (N, 6) array"""
        mot_result = []
        for cls, box, sco in zip(classes.cpu().tolist()[0], boxes.cpu().tolist()[0], scores.cpu().tolist()[0]):
            mot_result.append(box + [float(sco), int(cls)])
        return np.array(mot_result) if mot_result else np.empty((0, 6))

    def _smooth_scores(self, dets):
        """Apply EMA smoothing to detection scores based on spatial overlap"""
        if self.score_ema_alpha <= 0:
            self._prev_dets = dets.copy() if len(dets) > 0 else None
            return dets
        if self._prev_dets is None or len(self._prev_dets) == 0 or len(dets) == 0:
            self._prev_dets = dets.copy() if len(dets) > 0 else None
            return dets
        prev_mask = self._prev_dets[:, 4] > self.score_ema_min_score
        prev_valid = self._prev_dets[prev_mask]
        if len(prev_valid) == 0:
            self._prev_dets = dets.copy()
            return dets
        ious = _bbox_iou_batch(dets[:, :4], prev_valid[:, :4])
        alpha = self.score_ema_alpha
        n_smoothed = 0
        for i in range(len(dets)):
            if dets[i, 4] < self.score_ema_min_score:
                continue
            best_j = np.argmax(ious[i])
            if ious[i, best_j] > self.score_ema_iou_thresh:
                prev_score = prev_valid[best_j, 4]
                curr_score = dets[i, 4]
                dets[i, 4] = alpha * prev_score + (1 - alpha) * curr_score
                n_smoothed += 1
        self._total_smoothed += n_smoothed
        self._prev_dets = dets.copy()
        return dets

    def _update_trusted_buffer(self, dets):
        """从当前帧的高置信度检测中更新可信检测 buffer."""
        if not self.recovery or len(dets) == 0:
            for item in self._trusted_buffer:
                item['age'] += 1
            self._trusted_buffer = [item for item in self._trusted_buffer if item['age'] <= self.recovery_max_age]
            return
        high_conf = dets[dets[:, 4] > self.recovery_min_score]
        if len(high_conf) == 0:
            for item in self._trusted_buffer:
                item['age'] += 1
            self._trusted_buffer = [item for item in self._trusted_buffer if item['age'] <= self.recovery_max_age]
            return
        new_buffer = []
        matched_indices = set()
        if len(self._trusted_buffer) > 0:
            buf_boxes = np.array([item['bbox'] for item in self._trusted_buffer])
            ious = _bbox_iou_batch(high_conf[:, :4], buf_boxes)
            for di in range(len(high_conf)):
                best_bi = np.argmax(ious[di])
                if ious[di, best_bi] > 0.3:
                    old_bbox = self._trusted_buffer[best_bi]['bbox']
                    new_bbox = high_conf[di, :4].tolist()
                    old_cx = (old_bbox[0] + old_bbox[2]) / 2
                    old_cy = (old_bbox[1] + old_bbox[3]) / 2
                    new_cx = (new_bbox[0] + new_bbox[2]) / 2
                    new_cy = (new_bbox[1] + new_bbox[3]) / 2
                    vx = new_cx - old_cx
                    vy = new_cy - old_cy
                    self._trusted_buffer[best_bi].update({'bbox': new_bbox, 'velocity': [vx, vy], 'age': 0, 'score': float(high_conf[di, 4]), 'cls': int(high_conf[di, 5])})
                    matched_indices.add(best_bi)
                else:
                    new_buffer.append({'bbox': high_conf[di, :4].tolist(), 'velocity': [0, 0], 'age': 0, 'score': float(high_conf[di, 4]), 'cls': int(high_conf[di, 5])})
        else:
            for di in range(len(high_conf)):
                new_buffer.append({'bbox': high_conf[di, :4].tolist(), 'velocity': [0, 0], 'age': 0, 'score': float(high_conf[di, 4]), 'cls': int(high_conf[di, 5])})
        for bi, item in enumerate(self._trusted_buffer):
            if bi not in matched_indices:
                item['age'] += 1
        self._trusted_buffer = [item for item in self._trusted_buffer if item['age'] <= self.recovery_max_age] + new_buffer

    def _recover_dets(self, dets):
        """关键帧引导补检: 用 trusted buffer 投影框恢复间歇丢失的检测."""
        if not self.recovery or len(self._trusted_buffer) == 0:
            return dets
        if len(dets) > 0:
            current_high = dets[dets[:, 4] > self.recovery_min_score]
        else:
            current_high = np.empty((0, 6))
        n_recovered = 0
        dets_list = list(dets) if len(dets) > 0 else []
        for item in self._trusted_buffer:
            if item['age'] == 0:
                continue
            vx, vy = item['velocity']
            age = item['age']
            proj_bbox = [item['bbox'][0] + vx * age, item['bbox'][1] + vy * age, item['bbox'][2] + vx * age, item['bbox'][3] + vy * age]
            proj_bbox_np = np.array([proj_bbox])
            if len(current_high) > 0:
                ious_high = _bbox_iou_batch(proj_bbox_np, current_high[:, :4])
                if np.max(ious_high) > self.recovery_iou_thresh:
                    continue
            recovered = False
            if len(dets) > 0:
                ious_all = _bbox_iou_batch(proj_bbox_np, dets[:, :4])[0]
                best_idx = np.argmax(ious_all)
                if ious_all[best_idx] > self.recovery_iou_thresh:
                    if dets[best_idx, 4] < self.recovery_min_score:
                        dets[best_idx, 4] = max(dets[best_idx, 4], self.recovery_score)
                        recovered = True
            if not recovered:
                inject_score = self.recovery_score * max(0.5, 1.0 - 0.1 * age)
                inject_det = np.array([proj_bbox[0], proj_bbox[1], proj_bbox[2], proj_bbox[3], inject_score, float(item['cls'])])
                dets_list.append(inject_det)
                recovered = True
            if recovered:
                n_recovered += 1
        self._total_recovered += n_recovered
        if n_recovered > 0 and len(dets_list) > len(dets):
            return np.array(dets_list)
        return dets

    def _apply_min_hits(self, track_results):
        """修正 boxmot ByteTrack 忽略 min_hits 的问题."""
        mh = self._real_min_hits
        if mh == 2:
            return track_results
        if mh == 1:
            existing_ids = set()
            if len(track_results) > 0:
                existing_ids = set(track_results[:, 4].astype(int).tolist())
            extra = []
            for t in self.track.active_tracks:
                if not t.is_activated and t.id not in existing_ids:
                    row = list(t.xyxy) + [t.id, t.conf, t.cls, t.det_ind]
                    extra.append(row)
            if extra:
                extra_arr = np.array(extra)
                if len(track_results) > 0:
                    track_results = np.vstack([track_results, extra_arr])
                else:
                    track_results = extra_arr
            return track_results
        if mh >= 3:
            if len(track_results) == 0:
                return track_results
            id_to_len = {}
            for t in self.track.active_tracks:
                id_to_len[t.id] = t.tracklet_len
            keep_mask = []
            for row in track_results:
                tid = int(row[4])
                tlen = id_to_len.get(tid, 0)
                keep_mask.append(tlen >= mh - 1)
            keep_mask = np.array(keep_mask, dtype=bool)
            return track_results[keep_mask] if keep_mask.any() else np.empty((0, track_results.shape[1]))
        return track_results

    def _handle_jump(self, n_tracks):
        """Detect frame jumps and temporarily increase ByteTrack's buffer."""
        if not self.jump_detect:
            self._prev_n_tracks = n_tracks
            return
        if self._prev_n_tracks >= 2:
            lost_ratio = max(0.0, 1.0 - n_tracks / self._prev_n_tracks)
        else:
            lost_ratio = 0.0
        if lost_ratio >= self.jump_lost_ratio and self._jump_cooldown <= 0:
            self._jump_cooldown = self.jump_extra_buffer
            new_buf = self._base_buffer_size + self.jump_extra_buffer
            if hasattr(self.track, 'buffer_size'):
                self.track.buffer_size = new_buf
            print(f'[MM-ByteTrack] Jump @ frame {self._frame_idx}: {self._prev_n_tracks}->{n_tracks} tracks. Buffer: {self._base_buffer_size}->{new_buf}')
        if self._jump_cooldown > 0:
            self._jump_cooldown -= 1
            if self._jump_cooldown <= 0:
                if hasattr(self.track, 'buffer_size'):
                    self.track.buffer_size = self._base_buffer_size
        self._prev_n_tracks = n_tracks

    def __call__(self, image, results):
        """Run one frame of tracking."""
        mot_results = self.transform_mot(*results)
        if len(mot_results) > 0:
            mot_results = self._smooth_scores(mot_results)
        if self.recovery:
            self._update_trusted_buffer(mot_results)
            mot_results = self._recover_dets(mot_results)
        track_results = self.track.update(mot_results, image)
        track_results = self._apply_min_hits(track_results)
        self._handle_jump(len(track_results))
        self._frame_idx += 1
        return torch.from_numpy(track_results)

    def get_stats(self):
        """Return tracking statistics dict."""
        stats = {'total_frames': self._frame_idx, 'total_smoothed_dets': self._total_smoothed, 'avg_smoothed_per_frame': self._total_smoothed / max(1, self._frame_idx)}
        if self.recovery:
            stats['total_recovered_dets'] = self._total_recovered
            stats['avg_recovered_per_frame'] = self._total_recovered / max(1, self._frame_idx)
            stats['trusted_buffer_size'] = len(self._trusted_buffer)
        return stats

class V7AssistedTrack(Track):
    """IR 主体跟踪器 + auxiliary Track ID 软提示辅助关联."""

    def __init__(self, *args, v7_tid_bonus: float=0.2, v7_tid_history_len: int=10, v7_tid_majority_ratio: float=0.5, v7_tid_min_conf: float=0.3, adaptive_min_hits: bool=False, amh_confirmed: int=1, amh_unconfirmed: int=3, mechanism_debug_jsonl: str | None=None, mechanism_near_tie_thresh: float=0.05, **kwargs):
        super().__init__(*args, **kwargs)
        self._v7_tid_bonus = v7_tid_bonus
        self._v7_tid_history_len = v7_tid_history_len
        self._v7_tid_majority_ratio = v7_tid_majority_ratio
        self._v7_tid_min_conf = v7_tid_min_conf
        self._mechanism_debug_jsonl = mechanism_debug_jsonl
        self._mechanism_near_tie_thresh = mechanism_near_tie_thresh
        self._adaptive_min_hits = adaptive_min_hits
        self._amh_confirmed = amh_confirmed
        self._amh_unconfirmed = amh_unconfirmed
        self._v7_confirmed_det_indices = set()
        self._v7_confirmed_tracks = set()
        self._tid_history = {}
        self._tid_mapping = {}
        self._v7_stats = {'frames_with_v7_hint': 0, 'total_bonus_applied': 0, 'total_tid_updates': 0, 'adaptive_mh_fast_confirmed': 0, 'adaptive_mh_slow_filtered': 0}

    def __call__(self, image, results, v7_det_tids=None):
        """Run one frame with v7 tid hints."""
        mot_results = self.transform_mot(*results)
        if len(mot_results) > 0:
            mot_results = self._smooth_scores(mot_results)
        if self.recovery:
            self._update_trusted_buffer(mot_results)
            mot_results = self._recover_dets(mot_results)
        self._current_v7_tids = v7_det_tids
        self._v7_confirmed_det_indices = set()
        if v7_det_tids is not None:
            for i, tid in enumerate(v7_det_tids):
                if tid >= 0:
                    self._v7_confirmed_det_indices.add(i)
        if v7_det_tids is not None and len(self._tid_mapping) > 0:
            _V7TID_CTX['active'] = True
            _V7TID_CTX['det_tids'] = {i: int(tid) for i, tid in enumerate(v7_det_tids) if tid >= 0}
            _V7TID_CTX['track_tids'] = dict(self._tid_mapping)
            _V7TID_CTX['bonus'] = self._v7_tid_bonus
            _V7TID_CTX['n_bonus'] = 0
            _V7TID_CTX['frame_id'] = int(self._frame_idx + 1)
            _V7TID_CTX['near_tie_thresh'] = float(self._mechanism_near_tie_thresh)
            _V7TID_CTX['debug_jsonl'] = self._mechanism_debug_jsonl
            _V7TID_CTX['capture_done'] = False
            _V7TID_CTX['pending_assoc'] = None
        track_results = self.track.update(mot_results, image)
        if self._adaptive_min_hits and self._v7_confirmed_det_indices:
            for t in self.track.active_tracks:
                if t.det_ind in self._v7_confirmed_det_indices:
                    self._v7_confirmed_tracks.add(t.id)
        track_results = self._apply_min_hits(track_results)
        n_bonus = _V7TID_CTX.get('n_bonus', 0)
        _V7TID_CTX['active'] = False
        if n_bonus > 0:
            self._v7_stats['total_bonus_applied'] += n_bonus
        if v7_det_tids is not None and len(track_results) > 0:
            self._update_tid_mapping(track_results, mot_results, v7_det_tids)
        self._handle_jump(len(track_results))
        self._frame_idx += 1
        return torch.from_numpy(track_results)

    def _update_tid_mapping(self, track_results, dets, v7_det_tids):
        """在 ByteTrack update 之后, 将 v7_tid 映射到 IR track."""
        if v7_det_tids is None or len(dets) == 0 or len(track_results) == 0:
            return
        v7_tids = np.array(v7_det_tids)
        trk_boxes = track_results[:, :4]
        det_boxes = dets[:, :4]
        if isinstance(trk_boxes, torch.Tensor):
            trk_boxes = trk_boxes.numpy()
        ious = _bbox_iou_batch(trk_boxes, det_boxes)
        n_updates = 0
        for ti in range(len(track_results)):
            best_di = np.argmax(ious[ti])
            if ious[ti, best_di] < 0.5:
                continue
            v7_tid = int(v7_tids[best_di]) if best_di < len(v7_tids) else -1
            if v7_tid < 0:
                continue
            ir_tid = int(track_results[ti, 4])
            if ir_tid not in self._tid_history:
                self._tid_history[ir_tid] = []
            history = self._tid_history[ir_tid]
            history.append(v7_tid)
            if len(history) > self._v7_tid_history_len:
                history.pop(0)
            from collections import Counter
            vote, vote_count = Counter(history).most_common(1)[0]
            confidence = vote_count / len(history)
            if confidence >= self._v7_tid_majority_ratio:
                self._tid_mapping[ir_tid] = vote
            else:
                self._tid_mapping.pop(ir_tid, None)
            n_updates += 1
        if n_updates > 0:
            self._v7_stats['total_tid_updates'] += n_updates
            self._v7_stats['frames_with_v7_hint'] += 1

    def get_v7_tid_for_track(self, ir_track_id):
        """查询 IR track 对应的 auxiliary track id."""
        return self._tid_mapping.get(int(ir_track_id), -1)

    def get_aux_tid_for_track(self, ir_track_id):
        """查询 IR track 对应的 auxiliary track id."""
        return self.get_v7_tid_for_track(ir_track_id)

    def get_all_tid_mappings(self):
        """返回当前所有 IR→auxiliary tid 映射."""
        return dict(self._tid_mapping)

    def get_all_aux_tid_mappings(self):
        """返回当前所有 IR→auxiliary tid 映射."""
        return self.get_all_tid_mappings()

    def _apply_min_hits(self, track_results):
        """覆盖父类 _apply_min_hits, 实现 per-track 自适应 min_hits."""
        if not self._adaptive_min_hits:
            return super()._apply_min_hits(track_results)
        mh_conf = self._amh_confirmed
        mh_unconf = self._amh_unconfirmed
        id_to_len = {}
        for t in self.track.active_tracks:
            id_to_len[t.id] = t.tracklet_len
        if len(track_results) > 0 and mh_unconf >= 3:
            keep_mask = []
            for row in track_results:
                tid = int(row[4])
                tlen = id_to_len.get(tid, 0)
                is_confirmed = tid in self._v7_confirmed_tracks
                if is_confirmed:
                    keep_mask.append(True)
                else:
                    keep_mask.append(tlen >= mh_unconf - 1)
                    if not keep_mask[-1]:
                        self._v7_stats['adaptive_mh_slow_filtered'] += 1
            keep_mask = np.array(keep_mask, dtype=bool)
            if not keep_mask.all():
                track_results = track_results[keep_mask] if keep_mask.any() else np.empty((0, track_results.shape[1]))
        if mh_conf <= 1:
            existing_ids = set()
            if len(track_results) > 0:
                existing_ids = set(track_results[:, 4].astype(int).tolist())
            extra = []
            for t in self.track.active_tracks:
                if not t.is_activated and t.id not in existing_ids:
                    if t.det_ind in self._v7_confirmed_det_indices:
                        row = list(t.xyxy) + [t.id, t.conf, t.cls, t.det_ind]
                        extra.append(row)
                        self._v7_confirmed_tracks.add(t.id)
                        self._v7_stats['adaptive_mh_fast_confirmed'] += 1
            if extra:
                extra_arr = np.array(extra)
                if len(track_results) > 0:
                    track_results = np.vstack([track_results, extra_arr])
                else:
                    track_results = extra_arr
        return track_results

    def get_stats(self):
        stats = super().get_stats()
        stats.update(self._v7_stats)
        stats['active_tid_mappings'] = len(self._tid_mapping)
        if self._adaptive_min_hits:
            stats['v7_confirmed_tracks'] = len(self._v7_confirmed_tracks)
        return stats
ALPFTrack = V7AssistedTrack
