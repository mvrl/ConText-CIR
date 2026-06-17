import os
import json
from typing import Union, List, Literal, Optional

import torch
from torch.utils.data import Dataset
from PIL import Image


class CIRTripletDataset(Dataset):
    """Generic composed-image-retrieval triplet dataset from a JSON manifest."""

    def __init__(self, manifest_path: Union[str, os.PathLike], image_root: Union[str, os.PathLike],
                 preprocess: callable, parser=None, max_nps: int = 10,
                 np_cache_path: Optional[str] = None):
        """
        manifest_path : JSON list of {"reference","target","caption"} (paths under image_root).
        image_root    : directory the reference/target paths are resolved against.
        preprocess    : image transform shared with eval.
        parser        : CC-on switch. When not None, noun phrases are attached (from the
                        SQLite NP cache + spaCy fallback). NPs are consumed only by the CC loss.
        np_cache_path : prebuilt spaCy NP cache (nps_precompute.py); missing -> spaCy fallback.
        """
        self.image_root = image_root
        self.preprocess = preprocess
        self.parser = parser
        self.max_nps = max_nps
        self.np_cache_path = np_cache_path

        with open(manifest_path, "r") as f:
            triplets = json.load(f)

        self.reference_paths, self.target_paths, self.captions = [], [], []
        for t in triplets:
            self.reference_paths.append(os.path.join(image_root, t["reference"]))
            self.target_paths.append(os.path.join(image_root, t["target"]))
            self.captions.append(t["caption"])

        print(f"CIRTripletDataset initialized with {len(self.captions)} triplets "
              f"from {manifest_path}", flush=True)

        self.noun_phrases = []
        if self.parser is not None:
            self.noun_phrases = self._extract_all_noun_phrases()
        # drop the model-bound parser so the dataset pickles cleanly to spawn workers (NPs cached above)
        self.parser = None

    def _extract_all_noun_phrases(self):
        """caption -> [noun phrases] via the SQLite cache, with one bulk spaCy pass for misses."""
        from nps_precompute import NPCache, _get_nlp, _spacy_nps
        n = len(self.captions)
        out = [None] * n
        cache = None
        if self.np_cache_path and os.path.exists(self.np_cache_path):
            try:
                cache = NPCache(self.np_cache_path)
            except Exception as e:
                print(f"[cir] NP cache open failed at {self.np_cache_path} ({e}); using spaCy")
        miss_idx = []
        if cache is not None:
            for i, cap in enumerate(self.captions):
                hit = cache.get(cap) if cap else []
                if hit is None:
                    miss_idx.append(i)
                else:
                    out[i] = hit
        else:
            miss_idx = list(range(n))
        if miss_idx:
            nlp = _get_nlp()
            miss_caps = [self.captions[i] or "" for i in miss_idx]
            for j, doc in enumerate(nlp.pipe(miss_caps, batch_size=512)):
                out[miss_idx[j]] = _spacy_nps(doc, self.max_nps) if miss_caps[j] else []
        print(f"[cir] noun phrases ready for {n} captions "
              f"(cache hits={n - len(miss_idx)}, spaCy={len(miss_idx)})", flush=True)
        return [o if o is not None else [] for o in out]

    def __getitem__(self, index) -> Optional[dict]:
        try:
            reference_img = self.preprocess(Image.open(self.reference_paths[index]).convert("RGB"))
            target_img = self.preprocess(Image.open(self.target_paths[index]).convert("RGB"))
            return {
                "reference": reference_img,
                "target": target_img,
                "caption": self.captions[index],
                "noun_phrases": self.noun_phrases[index] if self.noun_phrases else [],
            }
        except (FileNotFoundError, OSError) as e:
            print(f"Error loading image at index {index}: {e}")
            return None

    def __len__(self) -> int:
        return len(self.captions)


class CIRRDataset(Dataset):
    """CIRR dataset for classic (gallery) and relative (query) evaluation modes."""

    def __init__(self, data_path: Union[str, os.PathLike], split: str,
                 mode: Literal['classic', 'relative'], preprocess: callable):
        self.preprocess = preprocess
        self.data_path = data_path
        self.split = split
        self.mode = mode

        with open(os.path.join(data_path, 'image_splits', f'split.rc2.{split}.json'), 'r') as f:
            self.mapper = json.load(f)

        if mode == 'classic':
            names, paths = zip(*[
                (name, os.path.join(data_path, 'img_raw', rel_path.strip('./')))
                for name, rel_path in self.mapper.items()
            ]) if self.mapper else ([], [])
            self.image_names: List[str] = list(names)
            self.image_paths: List[str] = list(paths)
        elif mode == 'relative':
            with open(os.path.join(data_path, 'captions', f'cap.rc2.{split}.json'), 'r') as f:
                self.annotations = json.load(f)
        else:
            raise ValueError(f"Mode '{mode}' not supported. Use 'classic' or 'relative'.")

        print(f"CIRR {split} ({mode}) dataset initialized")

    def __getitem__(self, index) -> dict:
        if self.mode == 'classic':
            try:
                img = self.preprocess(Image.open(self.image_paths[index]).convert("RGB"))
                return {"image": img, "image_name": self.image_names[index]}
            except (FileNotFoundError, OSError) as e:
                print(f"Error loading image at index {index}: {e}")
                return None
        else:  # relative
            annotation = self.annotations[index]
            ref_name = annotation['reference']
            ref_path = os.path.join(self.data_path, 'img_raw', self.mapper[ref_name].strip('./'))
            try:
                ref_img = self.preprocess(Image.open(ref_path).convert("RGB"))
            except (FileNotFoundError, OSError) as e:
                print(f"Error loading reference image at index {index}: {e}")
                ref_img = None

            return {
                "reference_image": ref_img,
                "reference_name": ref_name,
                "relative_caption": annotation['caption'],
                # CIRR submission is keyed by the per-query 'pairid' (4148 unique), NOT img_set['id']
                # (the shared image-set id, ~503 unique) -- using the latter collapses the submission.
                "pair_id": annotation['pairid'],
                "group_members": annotation['img_set']['members'],
            }

    def __len__(self) -> int:
        if self.mode == 'classic':
            return len(self.image_names)
        return len(self.annotations)


class CIRCODataset(Dataset):
    """CIRCO dataset for classic (gallery) and relative (query) evaluation modes.

    Images are sourced from COCO 2017 unlabeled, expected at:
        <dataset_path>/COCO2017_unlabeled/data/<000000xxxxxx>.jpg
    Annotations are expected at:
        <dataset_path>/annotations/<split>.json
    """

    _COCO_IMG_FILENAME = '{:012d}.jpg'

    def __init__(self, dataset_path: Union[str, os.PathLike], split: str,
                 mode: Literal['classic', 'relative'], preprocess: callable):
        self.preprocess = preprocess
        self.dataset_path = dataset_path
        self.split = split
        self.mode = mode
        self.img_dir = os.path.join(dataset_path, 'COCO2017_unlabeled', 'data')

        if mode == 'classic':
            img_info_path = os.path.join(
                dataset_path, 'COCO2017_unlabeled', 'annotations',
                'image_info_unlabeled2017.json'
            )
            with open(img_info_path, 'r') as f:
                img_data = json.load(f)
            self.images = img_data['images']  # list of {'id': int, 'file_name': str, ...}
        elif mode == 'relative':
            with open(os.path.join(dataset_path, 'annotations', f'{split}.json'), 'r') as f:
                self.annotations = json.load(f)
        else:
            raise ValueError(f"Mode '{mode}' not supported. Use 'classic' or 'relative'.")

        print(f"CIRCO {split} ({mode}) dataset initialized")

    def __getitem__(self, index) -> dict:
        if self.mode == 'classic':
            img_info = self.images[index]
            img_path = os.path.join(self.img_dir, img_info['file_name'])
            try:
                img = self.preprocess(Image.open(img_path).convert("RGB"))
                return {"image": img, "image_name": img_info['file_name']}
            except (FileNotFoundError, OSError) as e:
                print(f"Error loading image at index {index}: {e}")
                return None
        else:  # relative
            annotation = self.annotations[index]
            ref_img_id = annotation['reference_img_id']
            ref_img_path = os.path.join(self.img_dir, self._COCO_IMG_FILENAME.format(ref_img_id))
            try:
                ref_img = self.preprocess(Image.open(ref_img_path).convert("RGB"))
            except (FileNotFoundError, OSError) as e:
                print(f"Error loading reference image at index {index}: {e}")
                ref_img = None

            # Official CIRCO query template (eval/data/circo_query.jsonl):
            # "Find a picture that also {shared_concept}, but {relative_caption}." Feeding only
            # relative_caption drops the shared-concept clause and ~halves CIRCO mAP@5.
            caption = (f"Find a picture that also {annotation['shared_concept']}, "
                       f"but {annotation['relative_caption']}.")
            return {
                "reference_image": ref_img,
                "relative_caption": caption,
                "query_id": annotation['id'],
            }

    def __len__(self) -> int:
        if self.mode == 'classic':
            return len(self.images)
        return len(self.annotations)


def collate_fn_with_nps(batch):
    """Collate that stacks the image tensors and keeps captions / variable-length NPs as lists.
    Produces the per-batch dict consumed by ConText.training_step."""
    batch = [b for b in batch if b is not None]
    return {
        "reference": torch.stack([b["reference"] for b in batch]),
        "target": torch.stack([b["target"] for b in batch]),
        "caption": [b["caption"] for b in batch],
        "noun_phrases": [b.get("noun_phrases", []) for b in batch],
    }
