import os
import json
import datetime
import argparse

import torch
import yaml
import numpy as np
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from Models import ConText
from Dataset import CIRRDataset, CIRCODataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cfg = yaml.safe_load(open('config.yaml'))
PREP = transforms.Compose([
    transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.481, 0.457, 0.408], std=[0.268, 0.261, 0.275]),
])
SUB_ROOT = cfg.get('submissions', 'submissions')


@torch.no_grad()
def _encode_gallery(model, ds, desc):
    loader = DataLoader(ds, batch_size=256, num_workers=10, shuffle=False, pin_memory=True)
    feats, names = [], []
    for b in tqdm(loader, desc=desc):
        e = model.get_image_features(b['image'].to(device))
        feats.append(F.normalize(e.float(), dim=-1))
        names.extend(b['image_name'])
    assert len(names) == len(set(names)), "duplicate gallery names!"
    return torch.cat(feats).to(device), np.array(names)


@torch.no_grad()
def gen_circo(model, name):
    gal = CIRCODataset(cfg['circo_eval_path'], 'test', 'classic', PREP)
    idx_f, idx_n = _encode_gallery(model, gal, 'circo gallery')
    rel = CIRCODataset(cfg['circo_eval_path'], 'test', 'relative', PREP)
    loader = DataLoader(rel, batch_size=64, num_workers=10, shuffle=False, pin_memory=True)
    qf, qids = [], []
    for b in tqdm(loader, desc='circo queries'):
        e = model(b['reference_image'].to(device), b['relative_caption'])
        qf.append(F.normalize(e.float(), dim=-1)); qids.extend(b['query_id'])
    qf = torch.cat(qf)
    sims = qf @ idx_f.T
    top = torch.topk(sims, k=50, dim=-1).indices.cpu()
    ranked = idx_n[top]
    # CIRCO gallery image_name is the COCO filename ('000000151363.jpg'); the submission format wants
    # the integer COCO id (151363). query_id is the test.json 'id' (0..799) -> str key.
    def _gid(x):
        return int(str(x).split('.')[0])
    def _qk(q):
        return str(int(q)) if not isinstance(q, str) else q
    out = {_qk(qid): [_gid(x) for x in names[:50]] for qid, names in zip(qids, ranked)}
    os.makedirs(f'{SUB_ROOT}/circo', exist_ok=True)
    p = f'{SUB_ROOT}/circo/{name}.json'
    json.dump(out, open(p, 'w'), sort_keys=True)
    # validate vs official example format
    ex = json.load(open(os.path.join(cfg['circo_eval_path'], 'submission_examples', 'submission_test.json')))
    ok = (len(out) == len(ex)) and all(len(v) == 50 for v in out.values()) and \
         all(isinstance(v[0], int) for v in out.values())
    print(f"[circo] wrote {p} | queries={len(out)} (example={len(ex)}) | format_ok={ok}", flush=True)
    return p, len(out), ok


@torch.no_grad()
def gen_cirr(model, name):
    gal = CIRRDataset(cfg['cirr_eval_path'], 'test1', 'classic', PREP)
    idx_f, idx_n = _encode_gallery(model, gal, 'cirr gallery')
    rel = CIRRDataset(cfg['cirr_eval_path'], 'test1', 'relative', PREP)
    loader = DataLoader(rel, batch_size=256, num_workers=10, shuffle=False, pin_memory=True)
    pf, refn, pids, gms = [], [], [], []
    for b in tqdm(loader, desc='cirr queries'):
        gm = np.array(b['group_members']).T.tolist()
        e = model(b['reference_image'].to(device), b['relative_caption'])
        pf.append(F.normalize(e.float(), dim=-1)); refn.extend(b['reference_name'])
        pids.extend(b['pair_id']); gms.extend(gm)
    predicted = torch.cat(pf)
    distances = 1 - predicted @ idx_f.T
    sidx = torch.argsort(distances, dim=-1).cpu()
    snames = idx_n[sidx]
    refmask = snames != np.repeat(np.array(refn), len(idx_n)).reshape(len(snames), -1)
    snames = snames[refmask].reshape(snames.shape[0], snames.shape[1] - 1)
    gm_arr = np.array(gms)
    gmask = (snames[..., None] == gm_arr[:, None, :]).sum(-1).astype(bool)
    sgroup = snames[gmask].reshape(snames.shape[0], -1)
    sub = {'version': 'rc2', 'metric': 'recall'}
    sub.update({str(int(p)): pr[:50].tolist() for p, pr in zip(pids, snames)})
    gsub = {'version': 'rc2', 'metric': 'recall_subset'}
    gsub.update({str(int(p)): pr[:3].tolist() for p, pr in zip(pids, sgroup)})
    os.makedirs(f'{SUB_ROOT}/cirr', exist_ok=True)
    p1 = f'{SUB_ROOT}/cirr/{name}.json'; p2 = f'{SUB_ROOT}/cirr/subset_{name}.json'
    json.dump(sub, open(p1, 'w'), sort_keys=True)
    json.dump(gsub, open(p2, 'w'), sort_keys=True)
    nq = len(sub) - 2
    print(f"[cirr] wrote {p1} + {p2} | queries={nq}", flush=True)
    return p1, nq, True


def main():
    global SUB_ROOT
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', nargs='+', required=True, help='ConText .ckpt path(s)')
    ap.add_argument('--names', nargs='+', default=None, help='output names (default: ckpt basename)')
    ap.add_argument('--benchmarks', nargs='+', default=['circo', 'cirr'], choices=['circo', 'cirr'])
    ap.add_argument('--out_root', default=None, help='output root (default: config submissions)')
    args = ap.parse_args()
    if args.out_root:
        SUB_ROOT = args.out_root
    names = args.names or [os.path.basename(s).replace('.ckpt', '') for s in args.models]
    reg = f'{SUB_ROOT}/REGISTRY.txt'
    os.makedirs(SUB_ROOT, exist_ok=True)
    lines = []
    for spec, name in zip(args.models, names):
        print(f"\n===== {spec} -> {name} =====", flush=True)
        model = ConText.load_from_checkpoint(spec).eval().to(device)
        if 'circo' in args.benchmarks:
            p, n, ok = gen_circo(model, name)
            lines.append(f"CIRCO {p} | model={spec} | {n} queries | format_ok={ok}")
        if 'cirr' in args.benchmarks:
            p, n, ok = gen_cirr(model, name)
            lines.append(f"CIRR  {p} (+subset_) | model={spec} | {n} queries")
        del model; torch.cuda.empty_cache()
    with open(reg, 'a') as f:
        f.write(f"# generated {datetime.date.today().isoformat()}\n" + "\n".join(lines) + "\n")
    print(f"\n[registry] appended {len(lines)} entries to {reg}", flush=True)


if __name__ == '__main__':
    main()
