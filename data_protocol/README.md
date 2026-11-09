# M3OT splitting rules

RGB and IR are cut synchronously. If one modality remains frozen while the other changes, discard that interval from both, retain the valid prefix under its original name, and write the valid suffix under a new name. New suffix frames start at 1; observed track IDs are independently compacted from 1 in each modality. Preserve box coordinates and other GT columns. IR sequence names append `T`.

| Original pair | Discard (inclusive) | Retained prefix | New suffix |
| --- | --- | --- | --- |
| 2-02 / 2-02T | 169–178 | 1–168 as 2-02 / 2-02T | 179–700 as 2-10 / 2-10T, frames 1–522 |
| 2-06 / 2-06T | 185–194 | 1–184 as 2-06 / 2-06T | 195–600 as 2-11 / 2-11T, frames 1–406 |

Other documented cuts: `1-03 → 1-03 + 1-14`; `1-01 → 1-01 + 1-10 + 1-11`; `1-02 → 1-02 + 1-12`; `1-06 → 1-06 + 1-13`. Their exact original frame intervals are not available in the retained source material.

| Split | RGB sequences (IR appends T) |
| --- | --- |
| train | 1-01, 1-02, 1-04, 1-05, 1-06, 1-07, 1-09, 1-10, 1-11, 1-12, 1-13, 2-04, 2-09, 2-10 |
| val | 1-08, 2-02, 2-06, 2-11 |
| test, current annotations | 1-03, 1-09_test, 2-07 |

The cleaning notes name `1-14` instead of `1-09_test`; equivalence has not been established. `split_manifest.csv` records current annotation names. These rules are a cleaned paired protocol merging sequences from both UAV views.

```bash
python tools/split_paired_sequence.py --rgb-seq-dir raw/rgb/2-02 --ir-seq-dir raw/ir/2-02T --rule 2-02 --output prepared/cut_2-02
python tools/split_paired_sequence.py --rgb-seq-dir raw/rgb/2-06 --ir-seq-dir raw/ir/2-06T --rule 2-06 --output prepared/cut_2-06
```

Use raw uncut sequences. Output must be a new directory; the script preserves inputs and writes `output/{rgb,ir}/SEQUENCE/` with GT, seqinfo and a cut manifest containing ID maps. Move outputs into their assigned splits separately. COCO annotation regeneration is not included.
