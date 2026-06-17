import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl


_MISS = {"n": 0}


def _safe_open(path, preprocess):
    try:
        img = Image.open(path).convert("RGB")
    except (FileNotFoundError, OSError) as e:
        _MISS["n"] += 1
        if _MISS["n"] <= 5:
            print(f"[bench] WARNING missing/corrupt image {path}: {e}; using black.")
        img = Image.new("RGB", (224, 224))
    return preprocess(img)


def _make_loader(ds, collate, batch_size, num_workers):
    # spawn (not fork): the parent has CUDA/NCCL initialized
    return DataLoader(
        ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=True, collate_fn=collate, persistent_workers=False,
        multiprocessing_context="spawn" if num_workers > 0 else None)


class CIRRGallery(Dataset):
    """All val-split gallery images, keyed by CIRR image name."""

    def __init__(self, data_path, preprocess, split="val"):
        self.preprocess = preprocess
        with open(os.path.join(data_path, "image_splits", f"split.rc2.{split}.json")) as f:
            mapper = json.load(f)
        self.names = list(mapper.keys())
        self.paths = [os.path.join(data_path, "img_raw", mapper[n].strip("./")) for n in self.names]

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        return _safe_open(self.paths[i], self.preprocess), self.names[i]


class CIRRQueries(Dataset):
    """Val relative queries: reference image, caption, GT target, subset members."""

    def __init__(self, data_path, preprocess, split="val"):
        self.preprocess = preprocess
        with open(os.path.join(data_path, "image_splits", f"split.rc2.{split}.json")) as f:
            self.mapper = json.load(f)
        with open(os.path.join(data_path, "captions", f"cap.rc2.{split}.json")) as f:
            self.ann = json.load(f)
        self.data_path = data_path

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, i):
        a = self.ann[i]
        ref = a["reference"]
        path = os.path.join(self.data_path, "img_raw", self.mapper[ref].strip("./"))
        return {
            "image": _safe_open(path, self.preprocess),
            "reference": ref,
            "caption": a["caption"],
            "target": a["target_hard"],
            "members": list(a["img_set"]["members"]),
        }


class CIRCOGallery(Dataset):
    """All COCO-2017-unlabeled images, keyed by integer image id."""

    def __init__(self, data_path, preprocess):
        self.preprocess = preprocess
        self.img_dir = os.path.join(data_path, "COCO2017_unlabeled", "data")
        info_path = os.path.join(data_path, "COCO2017_unlabeled", "annotations",
                                 "image_info_unlabeled2017.json")
        with open(info_path) as f:
            info = json.load(f)
        self.ids = [im["id"] for im in info["images"]]
        self.files = [im["file_name"] for im in info["images"]]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        return _safe_open(os.path.join(self.img_dir, self.files[i]), self.preprocess), self.ids[i]


class CIRCOQueries(Dataset):
    """Val relative queries: reference image, caption, single target + all GT ids."""

    _FN = "{:012d}.jpg"

    def __init__(self, data_path, preprocess, split="val"):
        self.preprocess = preprocess
        self.img_dir = os.path.join(data_path, "COCO2017_unlabeled", "data")
        with open(os.path.join(data_path, "annotations", f"{split}.json")) as f:
            self.ann = json.load(f)

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, i):
        a = self.ann[i]
        ref_id = a["reference_img_id"]
        path = os.path.join(self.img_dir, self._FN.format(ref_id))

        caption = f"Find a picture that also {a['shared_concept']}, but {a['relative_caption']}."
        return {
            "image": _safe_open(path, self.preprocess),
            "reference_id": int(ref_id),
            "caption": caption,
            "target_id": int(a["target_img_id"]),
            "gt_ids": [int(x) for x in a["gt_img_ids"]],
        }


def _gallery_collate(batch):
    imgs = torch.stack([b[0] for b in batch])
    names = [b[1] for b in batch]
    return imgs, names


def _query_collate(batch):
    imgs = torch.stack([b["image"] for b in batch])
    rest = {k: [b[k] for b in batch] for k in batch[0] if k != "image"}
    return imgs, rest


@torch.no_grad()
def _encode_gallery(model, loader, device, amp_dtype):
    feats, names = [], []
    for imgs, ns in loader:
        imgs = imgs.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
            f = model.get_image_features(imgs)
        feats.append(F.normalize(f.float(), dim=-1))
        names.extend(ns)
    return torch.cat(feats), names


@torch.no_grad()
def _encode_queries(model, loader, device, amp_dtype):
    feats, meta = [], []
    for imgs, rest in loader:
        imgs = imgs.to(device, non_blocking=True)
        caps = list(rest["caption"])
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
            f = model(imgs, caps)
        feats.append(F.normalize(f.float(), dim=-1))
        meta.append(rest)
    qf = torch.cat(feats)
    merged = {k: [v for m in meta for v in m[k]] for k in meta[0]}
    return qf, merged


@torch.no_grad()
def evaluate_cirr(model, data_path, preprocess, device, amp_dtype,
                  batch_size=256, num_workers=16):
    gds = CIRRGallery(data_path, preprocess)
    qds = CIRRQueries(data_path, preprocess)
    gloader = _make_loader(gds, _gallery_collate, batch_size, num_workers)
    qloader = _make_loader(qds, _query_collate, batch_size, num_workers)
    gfeat, gnames = _encode_gallery(model, gloader, device, amp_dtype)
    qfeat, meta = _encode_queries(model, qloader, device, amp_dtype)

    name_to_idx = {n: i for i, n in enumerate(gnames)}
    gnames_arr = np.array(gnames)
    sims = qfeat @ gfeat.T                                  # (Q, G) on GPU
    # Exclude each query's reference image from the full-gallery ranking.
    ref_idx = torch.tensor([name_to_idx[r] for r in meta["reference"]], device=sims.device)
    sims[torch.arange(sims.size(0), device=sims.device), ref_idx] = float("-inf")

    topk = torch.topk(sims, k=50, dim=1).indices.cpu().numpy()   # (Q, 50)
    ranked_names = gnames_arr[topk]
    targets = np.array(meta["target"])

    recalls = {}
    for k in (1, 5, 10, 50):
        hits = (ranked_names[:, :k] == targets[:, None]).any(axis=1)
        recalls[f"cirr/recall_at{k}"] = float(hits.mean())

    # Subset recall: rank only within the 6 img_set.members (reference excluded).
    sims_cpu = sims.cpu().numpy()
    sub_hits = {1: [], 2: [], 3: []}
    for q in range(len(targets)):
        members = [m for m in meta["members"][q] if m != meta["reference"][q]]
        m_idx = [name_to_idx[m] for m in members]
        order = np.argsort(-sims_cpu[q, m_idx])
        ranked_members = [members[i] for i in order]
        for k in (1, 2, 3):
            sub_hits[k].append(targets[q] in ranked_members[:k])
    for k in (1, 2, 3):
        recalls[f"cirr/recall_subset_at{k}"] = float(np.mean(sub_hits[k]))
    return recalls


@torch.no_grad()
def evaluate_circo(model, data_path, preprocess, device, amp_dtype,
                   batch_size=256, num_workers=16):
    gds = CIRCOGallery(data_path, preprocess)
    qds = CIRCOQueries(data_path, preprocess)
    gloader = _make_loader(gds, _gallery_collate, batch_size, num_workers)
    qloader = _make_loader(qds, _query_collate, batch_size, num_workers)
    gfeat, gids = _encode_gallery(model, gloader, device, amp_dtype)
    qfeat, meta = _encode_queries(model, qloader, device, amp_dtype)

    gids_arr = np.array(gids, dtype=np.int64)
    sims = qfeat @ gfeat.T
    topk = torch.topk(sims, k=50, dim=1).indices.cpu().numpy()   # (Q, 50)
    ranked_ids = gids_arr[topk]                              # (Q, 50)

    ranks = (5, 10, 25, 50)
    aps = {k: [] for k in ranks}
    recs = {k: [] for k in ranks}
    for q in range(ranked_ids.shape[0]):
        preds = ranked_ids[q]                                # 50 unique gallery ids
        gt = np.array(meta["gt_ids"][q], dtype=np.int64)
        tgt = meta["target_id"][q]
        ap_labels = np.isin(preds, gt)
        precisions = np.cumsum(ap_labels) * ap_labels / np.arange(1, len(preds) + 1)
        rec_labels = (preds == tgt)
        for k in ranks:
            aps[k].append(float(np.sum(precisions[:k]) / min(len(gt), k)))
            recs[k].append(float(np.sum(rec_labels[:k])))
    out = {}
    for k in ranks:
        out[f"circo/map_at{k}"] = float(np.mean(aps[k]))
        out[f"circo/recall_at{k}"] = float(np.mean(recs[k]))
    return out


def _unwrap(pl_module):
    """Return the object exposing forward()/get_image_features()."""
    if hasattr(pl_module, "get_image_features"):
        return pl_module
    return getattr(pl_module, "model", pl_module)


class CIRBenchmarkCallback(pl.Callback):
    def __init__(self, cirr_path, circo_path, preprocess, out_dir,
                 every_n_steps=2000, batch_size=256, num_workers=16,
                 monitor="bench/score", run_circo=True, warmup_until=0):
        super().__init__()
        self.cirr_path = cirr_path
        self.circo_path = circo_path
        self.preprocess = preprocess
        self.out_dir = out_dir
        self.every_n_steps = every_n_steps
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.monitor = monitor
        self.run_circo = run_circo
        self.warmup_until = warmup_until
        self.best = float("-inf")
        self._last_eval_step = -1

    def _amp_dtype(self, trainer):
        p = str(trainer.precision)
        if "bf16" in p:
            return torch.bfloat16
        if "16" in p:
            return torch.float16
        return None

    def state_dict(self):
        # Persist best-score bookkeeping so --reload doesn't overwrite best.ckpt with a worse model.
        return {"best": self.best, "_last_eval_step": self._last_eval_step}

    def load_state_dict(self, state):
        self.best = state.get("best", float("-inf"))
        self._last_eval_step = state.get("_last_eval_step", -1)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = trainer.global_step
        if step == 0 or step < self.warmup_until:   # skip eval on the untrained (random-fusion) model
            return
        if step % self.every_n_steps != 0:
            return
        if step == self._last_eval_step:            # guard against grad-accum re-entry at same step
            return
        self._last_eval_step = step
        self._evaluate(trainer, pl_module, step)

    def on_train_end(self, trainer, pl_module):
        if trainer.global_step != self._last_eval_step:
            # Lightning forbids self.log() in on_train_end; just print + checkpoint
            self._evaluate(trainer, pl_module, trainer.global_step, do_log=False)

    def _evaluate(self, trainer, pl_module, step, do_log=True):
        metrics = {}
        if trainer.is_global_zero:
            model = _unwrap(pl_module)
            was_training = model.training
            model.eval()
            device = pl_module.device
            amp = self._amp_dtype(trainer)
            try:
                metrics.update(evaluate_cirr(model, self.cirr_path, self.preprocess,
                                             device, amp, self.batch_size, self.num_workers))
                if self.run_circo:
                    metrics.update(evaluate_circo(model, self.circo_path, self.preprocess,
                                                  device, amp, self.batch_size, self.num_workers))
                metrics["bench/score"] = (
                    0.5 * metrics.get("cirr/recall_at5", 0.0)
                    + 0.5 * metrics.get("circo/map_at10", 0.0)
                )
            except Exception as e:
                import traceback
                print(f"[bench] eval FAILED at step {step}: {e}")
                traceback.print_exc()
                metrics = {}
            finally:
                model.train(was_training)
            if metrics:
                line = " | ".join(f"{k}={v:.4f}" for k, v in sorted(metrics.items()))
                print(f"\n[bench] step {step}: {line}\n", flush=True)

        # Sync to all ranks so logging + checkpoint decisions are consistent.
        metrics = trainer.strategy.broadcast(metrics, src=0)
        if not metrics:
            return
        if do_log:
            for k, v in metrics.items():
                pl_module.log(k, float(v), rank_zero_only=False, sync_dist=False,
                              prog_bar=(k == self.monitor), on_step=True, on_epoch=False)

        score = metrics.get(self.monitor)
        if score is not None and score > self.best:
            self.best = score
            ckpt = os.path.join(self.out_dir, "best.ckpt")
            trainer.save_checkpoint(ckpt)     # collective; safe on all ranks
            if trainer.is_global_zero:
                print(f"[bench] new best {self.monitor}={score:.4f} -> {ckpt}", flush=True)
