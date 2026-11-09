import csv
import os
from collections import defaultdict
import torch

def infer_sequence_name(input_path):
    norm_path = os.path.normpath(input_path)
    base = os.path.basename(norm_path)
    if base.lower() == 'img1':
        return os.path.basename(os.path.dirname(norm_path))
    return base

def load_detection_cache(csv_path):
    cache = defaultdict(lambda: defaultdict(list))
    with open(csv_path, 'r', newline='') as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            sequence = row['sequence']
            frame_id = int(row['frame_id'])
            cache[sequence][frame_id].append({'frame_name': row.get('frame_name', ''), 'x1': float(row['x1']), 'y1': float(row['y1']), 'x2': float(row['x2']), 'y2': float(row['y2']), 'score': float(row['score']), 'class_id': int(float(row['class_id']))})
    return cache

def get_frame_rows(cache, sequence, frame_id):
    if cache is None:
        return []
    return cache.get(sequence, {}).get(frame_id, [])

def build_results_from_rows(rows, device):
    if rows:
        labels = torch.tensor([[row['class_id'] for row in rows]], dtype=torch.int64, device=device)
        boxes = torch.tensor([[[row['x1'], row['y1'], row['x2'], row['y2']] for row in rows]], dtype=torch.float32, device=device)
        scores = torch.tensor([[row['score'] for row in rows]], dtype=torch.float32, device=device)
    else:
        labels = torch.empty((1, 0), dtype=torch.int64, device=device)
        boxes = torch.empty((1, 0, 4), dtype=torch.float32, device=device)
        scores = torch.empty((1, 0), dtype=torch.float32, device=device)
    return (labels, boxes, scores)
