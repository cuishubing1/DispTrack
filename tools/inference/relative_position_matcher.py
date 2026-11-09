"""RelativePositionMatcher — 基于相对位置结构的跨模态检测匹配"""
import numpy as np
from scipy.optimize import linear_sum_assignment

class RelativePositionMatcher:
    """跨模态检测匹配器: 基于全局偏移估计 + 邻域形状描述子."""
    SUPPORTED_VARIANTS = {'full', 'no-hist', 'no-shape', 'no-fallback', 'center-only', 'nn'}

    def __init__(self, offset_bin_size: float=10.0, offset_peak_ratio: float=0.3, n_neighbors: int=4, desc_match_thresh: float=100.0, center_dist_thresh: float=30.0, min_dets_for_structure: int=3, fallback_to_nearest: bool=True, variant: str='full'):
        if variant not in self.SUPPORTED_VARIANTS:
            raise ValueError(f'Unsupported matcher variant: {variant}. Expected one of {sorted(self.SUPPORTED_VARIANTS)}')
        self.offset_bin_size = offset_bin_size
        self.offset_peak_ratio = offset_peak_ratio
        self.n_neighbors = n_neighbors
        self.desc_match_thresh = desc_match_thresh
        self.center_dist_thresh = center_dist_thresh
        self.min_dets_for_structure = min_dets_for_structure
        self.variant = variant
        self.use_hist_vote = variant in {'full', 'no-shape', 'no-fallback'}
        self.use_shape_descriptor = variant in {'full', 'no-hist', 'no-fallback'}
        self.force_center_only = variant == 'center-only'
        self.force_greedy_nearest = variant == 'nn'
        self.fallback_to_nearest = fallback_to_nearest and variant not in {'no-fallback', 'center-only', 'nn'}
        self._stats = {'total_calls': 0, 'total_matched': 0, 'total_unmatched_a': 0, 'total_unmatched_b': 0, 'offset_reliable': 0, 'fallback_nearest': 0}

    def __call__(self, dets_a, dets_b):
        """匹配两组检测."""
        self._stats['total_calls'] += 1
        N = len(dets_a)
        M = len(dets_b)
        if N == 0 and M == 0:
            return ([], [], [])
        if N == 0:
            return ([], [], list(range(M)))
        if M == 0:
            return ([], list(range(N)), [])
        centers_a = self._get_centers(dets_a)
        centers_b = self._get_centers(dets_b)
        if self.force_greedy_nearest:
            return self._greedy_nearest_match(centers_a, centers_b)
        if self.force_center_only:
            return self._center_only_match(centers_a, centers_b)
        if N < self.min_dets_for_structure or M < self.min_dets_for_structure:
            if self.fallback_to_nearest:
                self._stats['fallback_nearest'] += 1
                return self._nearest_match(centers_a, centers_b)
            else:
                return ([], list(range(N)), list(range(M)))
        if not self.use_hist_vote:
            matched, unmatched_a, unmatched_b = self._structure_match_no_align(centers_a, centers_b)
            self._stats['total_matched'] += len(matched)
            self._stats['total_unmatched_a'] += len(unmatched_a)
            self._stats['total_unmatched_b'] += len(unmatched_b)
            return (matched, unmatched_a, unmatched_b)
        offset, reliable = self._estimate_global_offset(centers_a, centers_b)
        if reliable:
            self._stats['offset_reliable'] += 1
        if reliable:
            centers_b_aligned = centers_b - offset
            if self.use_shape_descriptor:
                matched, unmatched_a, unmatched_b = self._structure_match(centers_a, centers_b_aligned, centers_b)
            else:
                matched, unmatched_a, unmatched_b = self._center_only_match(centers_a, centers_b_aligned, centers_b)
        elif self.use_shape_descriptor:
            matched, unmatched_a, unmatched_b = self._structure_match_no_align(centers_a, centers_b)
        elif self.fallback_to_nearest:
            self._stats['fallback_nearest'] += 1
            matched, unmatched_a, unmatched_b = self._nearest_match(centers_a, centers_b)
        else:
            matched, unmatched_a, unmatched_b = self._center_only_match(centers_a, centers_b)
        self._stats['total_matched'] += len(matched)
        self._stats['total_unmatched_a'] += len(unmatched_a)
        self._stats['total_unmatched_b'] += len(unmatched_b)
        return (matched, unmatched_a, unmatched_b)

    def _estimate_global_offset(self, centers_a, centers_b):
        """估计 B 相对于 A 的全局偏移向量."""
        N = len(centers_a)
        M = len(centers_b)
        deltas_x = centers_b[:, 0][None, :] - centers_a[:, 0][:, None]
        deltas_y = centers_b[:, 1][None, :] - centers_a[:, 1][:, None]
        deltas_x = deltas_x.flatten()
        deltas_y = deltas_y.flatten()
        bs = self.offset_bin_size
        x_min, x_max = (deltas_x.min() - bs, deltas_x.max() + bs)
        y_min, y_max = (deltas_y.min() - bs, deltas_y.max() + bs)
        n_bins_x = max(1, int((x_max - x_min) / bs) + 1)
        n_bins_y = max(1, int((y_max - y_min) / bs) + 1)
        hist, x_edges, y_edges = np.histogram2d(deltas_x, deltas_y, bins=[n_bins_x, n_bins_y], range=[[x_min, x_max], [y_min, y_max]])
        peak_idx = np.unravel_index(np.argmax(hist), hist.shape)
        peak_votes = hist[peak_idx]
        total_votes = N * M
        peak_x = (x_edges[peak_idx[0]] + x_edges[peak_idx[0] + 1]) / 2
        peak_y = (y_edges[peak_idx[1]] + y_edges[peak_idx[1] + 1]) / 2
        offset = self._refine_peak(hist, x_edges, y_edges, peak_idx)
        min_nm = min(N, M)
        reliable = peak_votes >= min_nm * self.offset_peak_ratio and peak_votes >= 3
        return (offset, reliable)

    def _refine_peak(self, hist, x_edges, y_edges, peak_idx):
        """3×3 邻域加权平均细化峰值位置."""
        pi, pj = peak_idx
        H, W = hist.shape
        total_w = 0.0
        wx = 0.0
        wy = 0.0
        for di in range(-1, 2):
            for dj in range(-1, 2):
                ni, nj = (pi + di, pj + dj)
                if 0 <= ni < H and 0 <= nj < W:
                    w = hist[ni, nj]
                    cx = (x_edges[ni] + x_edges[ni + 1]) / 2
                    cy = (y_edges[nj] + y_edges[nj + 1]) / 2
                    wx += w * cx
                    wy += w * cy
                    total_w += w
        if total_w > 0:
            return np.array([wx / total_w, wy / total_w])
        else:
            cx = (x_edges[pi] + x_edges[pi + 1]) / 2
            cy = (y_edges[pj] + y_edges[pj + 1]) / 2
            return np.array([cx, cy])

    def _structure_match(self, centers_a, centers_b_aligned, centers_b_orig):
        """全局偏移对齐后, 用邻域描述子做 Hungarian 匹配."""
        N = len(centers_a)
        M = len(centers_b_aligned)
        desc_a = self._build_descriptors(centers_a)
        desc_b = self._build_descriptors(centers_b_aligned)
        center_dists = np.sqrt((centers_a[:, 0:1] - centers_b_aligned[:, 0:1].T) ** 2 + (centers_a[:, 1:2] - centers_b_aligned[:, 1:2].T) ** 2)
        desc_dists = np.zeros((N, M))
        for i in range(N):
            for j in range(M):
                desc_dists[i, j] = np.linalg.norm(desc_a[i] - desc_b[j])
        desc_max = desc_dists.max() + 1e-06
        desc_norm = desc_dists / desc_max
        max_center = center_dists.max() + 1e-06
        cost = center_dists + 0.1 * max_center * desc_norm
        row_ind, col_ind = linear_sum_assignment(cost)
        matched = []
        unmatched_a = set(range(N))
        unmatched_b = set(range(M))
        for r, c in zip(row_ind, col_ind):
            d_center = center_dists[r, c]
            if d_center < self.center_dist_thresh:
                orig_dist = np.linalg.norm(centers_a[r] - centers_b_orig[c])
                matched.append((r, c, float(orig_dist)))
                unmatched_a.discard(r)
                unmatched_b.discard(c)
        return (matched, sorted(unmatched_a), sorted(unmatched_b))

    def _structure_match_no_align(self, centers_a, centers_b):
        """偏移估计不可靠时, 纯用邻域描述子匹配 (旋转/平移不变)."""
        N = len(centers_a)
        M = len(centers_b)
        desc_a = self._build_distance_descriptors(centers_a)
        desc_b = self._build_distance_descriptors(centers_b)
        cost = np.zeros((N, M))
        for i in range(N):
            for j in range(M):
                cost[i, j] = np.linalg.norm(desc_a[i] - desc_b[j])
        row_ind, col_ind = linear_sum_assignment(cost)
        thresh = self.desc_match_thresh * 1.5
        matched = []
        unmatched_a = set(range(N))
        unmatched_b = set(range(M))
        for r, c in zip(row_ind, col_ind):
            d = cost[r, c]
            if d < thresh:
                orig_dist = np.linalg.norm(centers_a[r] - centers_b[c])
                matched.append((r, c, float(orig_dist)))
                unmatched_a.discard(r)
                unmatched_b.discard(c)
        return (matched, sorted(unmatched_a), sorted(unmatched_b))

    def _center_only_match(self, centers_a, centers_b, centers_b_orig=None):
        """仅基于中心距离做 Hungarian 匹配."""
        N = len(centers_a)
        M = len(centers_b)
        dist = np.sqrt((centers_a[:, 0:1] - centers_b[:, 0:1].T) ** 2 + (centers_a[:, 1:2] - centers_b[:, 1:2].T) ** 2)
        row_ind, col_ind = linear_sum_assignment(dist)
        matched = []
        unmatched_a = set(range(N))
        unmatched_b = set(range(M))
        centers_b_report = centers_b if centers_b_orig is None else centers_b_orig
        for r, c in zip(row_ind, col_ind):
            if dist[r, c] < self.center_dist_thresh:
                report_dist = np.linalg.norm(centers_a[r] - centers_b_report[c])
                matched.append((r, c, float(report_dist)))
                unmatched_a.discard(r)
                unmatched_b.discard(c)
        return (matched, sorted(unmatched_a), sorted(unmatched_b))

    def _build_descriptors(self, centers):
        """构建邻域形状描述子: 相对位置向量 (平移不变)."""
        N = len(centers)
        K = min(self.n_neighbors, N - 1)
        if K == 0:
            return np.zeros((N, self.n_neighbors * 2))
        descriptors = []
        for i in range(N):
            diffs = centers - centers[i]
            dists = np.linalg.norm(diffs, axis=1)
            dists[i] = np.inf
            knn_idx = np.argpartition(dists, K)[:K]
            rel_vecs = diffs[knn_idx]
            angles = np.arctan2(rel_vecs[:, 1], rel_vecs[:, 0])
            sort_order = np.argsort(angles)
            rel_vecs = rel_vecs[sort_order]
            desc = rel_vecs.flatten()
            if len(desc) < self.n_neighbors * 2:
                desc = np.pad(desc, (0, self.n_neighbors * 2 - len(desc)))
            descriptors.append(desc)
        return np.array(descriptors)

    def _build_distance_descriptors(self, centers):
        """构建距离排序描述子 (旋转+平移不变, 用于无对齐情况)."""
        N = len(centers)
        K = min(self.n_neighbors, N - 1)
        if K == 0:
            return np.zeros((N, self.n_neighbors))
        descriptors = []
        for i in range(N):
            diffs = centers - centers[i]
            dists = np.linalg.norm(diffs, axis=1)
            dists[i] = np.inf
            knn_dists = np.sort(np.partition(dists, K)[:K])
            desc = knn_dists
            if len(desc) < self.n_neighbors:
                desc = np.pad(desc, (0, self.n_neighbors - len(desc)), constant_values=0)
            descriptors.append(desc)
        return np.array(descriptors)

    def _nearest_match(self, centers_a, centers_b):
        """检测数太少时退化为最近邻距离矩阵 + Hungarian 匹配."""
        N = len(centers_a)
        M = len(centers_b)
        dist = np.sqrt((centers_a[:, 0:1] - centers_b[:, 0:1].T) ** 2 + (centers_a[:, 1:2] - centers_b[:, 1:2].T) ** 2)
        row_ind, col_ind = linear_sum_assignment(dist)
        matched = []
        unmatched_a = set(range(N))
        unmatched_b = set(range(M))
        for r, c in zip(row_ind, col_ind):
            if dist[r, c] < self.center_dist_thresh:
                matched.append((r, c, float(dist[r, c])))
                unmatched_a.discard(r)
                unmatched_b.discard(c)
        return (matched, sorted(unmatched_a), sorted(unmatched_b))

    def _greedy_nearest_match(self, centers_a, centers_b):
        """纯最近邻: 按距离从小到大贪心选边, 不做全局 Hungarian."""
        N = len(centers_a)
        M = len(centers_b)
        dist = np.sqrt((centers_a[:, 0:1] - centers_b[:, 0:1].T) ** 2 + (centers_a[:, 1:2] - centers_b[:, 1:2].T) ** 2)
        candidate_pairs = []
        for i in range(N):
            for j in range(M):
                if dist[i, j] < self.center_dist_thresh:
                    candidate_pairs.append((float(dist[i, j]), i, j))
        candidate_pairs.sort(key=lambda item: item[0])
        matched = []
        unmatched_a = set(range(N))
        unmatched_b = set(range(M))
        for pair_dist, i, j in candidate_pairs:
            if i not in unmatched_a or j not in unmatched_b:
                continue
            matched.append((i, j, pair_dist))
            unmatched_a.discard(i)
            unmatched_b.discard(j)
        return (matched, sorted(unmatched_a), sorted(unmatched_b))

    @staticmethod
    def _get_centers(dets):
        """提取检测框中心点. dets: (N, 4+) [x1, y1, x2, y2, ...]"""
        cx = (dets[:, 0] + dets[:, 2]) / 2
        cy = (dets[:, 1] + dets[:, 3]) / 2
        return np.stack([cx, cy], axis=1)

    def get_stats(self):
        return dict(self._stats)

    def reset_stats(self):
        for k in self._stats:
            self._stats[k] = 0
