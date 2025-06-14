import json
import os
from typing import Union, List, Dict, Literal
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
import argparse

from transformers import CLIPTokenizer, CLIPImageProcessor
from PIL import Image
import PIL
import os
from typing import Union, List, Dict, Literal, Optional
from pathlib import Path
import random
import yaml
import json


class CIRDataset(Dataset):
    """
    CIR dataset for handling different datasets with preprocessing.
    """

    def __init__(self, data_path: Union[str, os.PathLike], split: Literal['train', 'val', 'test'],
                 dataset: Literal['cirr', 'cirr_r', 'hotels'], preprocess: callable):
        """
        Args:
            data_path (Union[str, os.PathLike]): Path to dataset.
            split (Literal): Dataset split ['train', 'test', 'val'].
            dataset (Literal): Dataset name ['cirr', 'cirr_r', 'hotels'].
            preprocess (callable): Image preprocessing function.
        """
        self.split = split
        self.preprocess = preprocess
        self.data_path = data_path

        # Define dataset-specific paths
        dataset_paths = {
            'cirr': os.path.join(data_path, 'captions', f'cap.rc2.{split}.json'),
            'cirr_r': os.path.join(data_path, 'rewritten_captions', f'rewritten_{split}.json'),
            'hotels': os.path.join(data_path, 'captions', f'hotel_{split}_data.json')
        }

        if dataset not in dataset_paths:
            raise ValueError(f"Dataset '{dataset}' not implemented")

        # Load annotations
        with open(dataset_paths[dataset], "r") as f:
            imgs_info = json.load(f)

        self.reference_paths, self.target_paths, self.captions = [], [], []
        self.distractors = []
        self.max_distractor_len = -1

        if dataset == 'cirr' and split == 'test':
            split = 'test1'
            
        # Handle dataset-specific structure
        if dataset in ['cirr', 'cirr_r']:
            with open(os.path.join(data_path, 'image_splits', f'split.rc2.{split}.json'), "r") as f:
                mapper = json.load(f)

            for img_info in imgs_info:
                ref_filename = img_info.get("reference") if dataset == 'cirr' else img_info["query_image_url"].split('/')[-1].strip('.png')
                tgt_filename = None if split in 'test1' and dataset == 'cirr' else img_info.get("target_hard" if dataset == 'cirr' else "retrieved_image_url").split('/')[-1].strip('.png')

                if dataset == 'cirr':
                    self.reference_paths.append(os.path.join(data_path, 'img_raw', mapper[ref_filename].strip('./')))
                    self.target_paths.append(os.path.join(data_path, 'img_raw', mapper[tgt_filename].strip('./')) if tgt_filename else None)
                    self.captions.append(img_info["caption"])
                else:
                    # ref_paths = [os.path.join(data_path, 'img_raw', mapper[ref_filename].strip('./'))] * len(img_info["difference_captions"])
                    # tgt_paths = [os.path.join(data_path, 'img_raw', mapper[tgt_filename].strip('./'))] * len(img_info["difference_captions"])
                    # self.reference_paths.extend(ref_paths)
                    # self.target_paths.extend(tgt_paths)
                    # self.captions.extend(img_info["difference_captions"])
                    self.reference_paths.append(os.path.join(data_path, 'img_raw', mapper[ref_filename].strip('./')))
                    self.target_paths.append(os.path.join(data_path, 'img_raw', mapper[tgt_filename].strip('./')) if tgt_filename else None)
                    self.captions.append(img_info["difference_captions"])
                    

        elif dataset == 'hotels':
            img_dir = 'train' if split == 'train' else 'val_test'

            for img_info in imgs_info:
                self.reference_paths.append(os.path.join(data_path, 'images', img_dir, *img_info["query_image"].split('/')[-3:]))
                self.target_paths.append(os.path.join(data_path, 'images', img_dir, *img_info["result_image"].split('/')[-3:]))
                self.captions.append(img_info["difference"])
                if 'distractors' in img_info:
                    distractors = img_info['distractors']
                    if len(distractors) > self.max_distractor_len:
                        self.max_distractor_len = len(distractors)
                    self.distractors.append([os.path.join(data_path, 'images', img_dir, *distractor.split('/')[-3:]) for distractor in distractors])

        if len(self.distractors) == 0:
            self.distractors = None

        print(f"{split} dataset initialized with {len(self.reference_paths)} samples from {dataset_paths[dataset]}")

    def __getitem__(self, index) -> dict:
        """
        Returns a single dataset sample.
        """
        try:
            reference_img = Image.open(self.reference_paths[index]).convert("RGB")
            target_img = Image.open(self.target_paths[index]).convert("RGB") if self.target_paths[index] else None #.convert("RGB") if self.target_paths[index] else None
            reference_img = self.preprocess(reference_img)
            target_img = self.preprocess(target_img) if target_img else None
            
            caption = self.captions[index]
            #truncate caption to 77 tokens

            distractor_paths = self.distractors[index] if self.distractors else None
            distractors = [Image.open(distractor_path).convert("RGB") for distractor_path in distractor_paths] if distractor_paths else None
            distractors = [self.preprocess(distractor) for distractor in distractors] if distractors else None

            # pad distractor paths and distractors
            # if self.distractors and len(self.distractors[index]) < self.max_distractor_len:
            #     pad_len = self.max_distractor_len - len(self.distractors[index])
            #     distractor_paths.extend([None] * pad_len)
            #     distractors.extend([None] * pad_len)
            

            return {
                "reference": reference_img,
                "reference_path": self.reference_paths[index],
                "target": target_img,
                "target_path": self.target_paths[index],
                "caption": caption,
                # "distractors": distractors,
                # "distractor_paths": distractor_paths
            }

        except (FileNotFoundError, OSError) as e:
            import code; code.interact(local=dict(globals(), **locals()))
            print(f"Error loading image at index {index}: {e}")
            return None
        
    def get_all_distractors(self):
        """
        Returns all distractors in the dataset.
        """
        return self.distractors

    def get_all_images(self, use_distractors):
        """
        Returns all images in the dataset.
        """
        all_paths = []
        for ref, tgt in zip(self.reference_paths, self.target_paths):
            all_paths.append(ref)
            if tgt:
                all_paths.append(tgt)
        
        # if use_distractors and self.distractors:
        #     for distractors in self.distractors:
        #         all_paths.extend(distractors)

        # else:
        #     print("Distractors not used")

        all_paths = [path for path in all_paths if path]
        all_paths = list(set(all_paths))
        return all_paths
        

    def __len__(self) -> int:
        return len(self.captions)

class MixDataset(Dataset):
    """
    Mix dataset for combining a CIRR and CIRR_R dataset in various proportions
    """

    def __init__(self, cirr_set, cirr_r_set, mix_ratio: float, preprocess: callable):

        print(f"Initializing MixDataset with mix ratio {mix_ratio}")
        self.preprocess = preprocess
        cirr_refs = cirr_set.reference_paths
        cirr_tgts = cirr_set.target_paths
        cirr_caps = cirr_set.captions
        # cirr_caps = [caps[0] for caps in cirr_caps]


        cirr_r_refs = cirr_r_set.reference_paths
        cirr_r_tgts = cirr_r_set.target_paths
        cirr_r_caps = cirr_r_set.captions
        cirr_r_caps = [caps[0] for caps in cirr_r_caps]

        #Match the samples in each dataset with identical refs and tgts 
        aligned_reference_paths = []
        aligned_target_paths = []
        aligned_captions = []
        failed = 0
        added = 0

        unmatchable_cirr_refs = []
        unmatchable_cirr_tgts = []
        unmatchable_cirr_caps = []

        # # add unmatchable cirr ref, tar, cap samples into unmatchable pool
        # for i in tqdm(range(len(cirr_refs))):
        #     ref = cirr_refs[i]
        #     tgt = cirr_tgts[i]
        #     cirr_r_idx = -1
        #     for j in range(len(cirr_r_refs)):
        #         if cirr_r_refs[j] == ref and cirr_r_tgts[j] == tgt:
        #             cirr_r_idx = j
        #             break
        #     if cirr_r_idx == -1:
        #         unmatchable_cirr_refs.append(ref)
        #         unmatchable_cirr_tgts.append(tgt)
        #         unmatchable_cirr_caps.append(cirr_caps[i])
        #         added += 1
        # print(f"{added} unmatchable samples")

        failed = 0
        for i in tqdm(range(len(cirr_r_refs))):
            ref = cirr_r_refs[i]
            tgt = cirr_r_tgts[i]
            cirr_idx = -1
            for j in range(len(cirr_refs)):
                if cirr_refs[j] == ref and cirr_tgts[j] == tgt:
                    cirr_idx = j
                    break
            if cirr_idx != -1:
                aligned_reference_paths.append((cirr_refs[cirr_idx], ref))
                aligned_target_paths.append((cirr_tgts[cirr_idx], tgt))
                aligned_captions.append((cirr_caps[cirr_idx], cirr_r_caps[i]))
            else:
                failed += 1
        print(f"Failed to align {failed} samples")

        import code; code.interact(local=dict(globals(), **locals()))

        len_dataset = len(cirr_refs)
        mix_len = int(len_dataset * mix_ratio)

        mask = [1] * mix_len + [0] * (len(aligned_reference_paths) - mix_len)
        random.shuffle(mask)
        
        self.reference_paths = [cirr_r_ref if mask[i] else cirr_ref for i, (cirr_ref, cirr_r_ref) in enumerate(aligned_reference_paths)]
        self.target_paths = [cirr_r_tgt if mask[i] else cirr_tgt for i, (cirr_tgt, cirr_r_tgt) in enumerate(aligned_target_paths)]
        self.captions = [cirr_r_cap if mask[i] else cirr_cap for i, (cirr_cap, cirr_r_cap) in enumerate(aligned_captions)]

        # fill in remaining samples from unmatched
        remaining = len_dataset - len(self.reference_paths)
        idxs = np.random.choice(len(unmatchable_cirr_refs), remaining, replace=False)
        self.reference_paths.extend([unmatchable_cirr_refs[i] for i in idxs])
        self.target_paths.extend([unmatchable_cirr_tgts[i] for i in idxs])
        self.captions.extend([unmatchable_cirr_caps[i] for i in idxs])
    
        print(f"Mix dataset initialized with {len(self.captions)} samples")

    def __getitem__(self, index) -> dict:
        """
        Returns a single dataset sample.
        """
        try:
            reference_img = Image.open(self.reference_paths[index]).convert("RGB")
            target_img = Image.open(self.target_paths[index]).convert("RGB")
            reference_img = self.preprocess(reference_img)
            target_img = self.preprocess(target_img)
            
            caption = self.captions[index]

            return {
                "reference": reference_img,
                "reference_path": self.reference_paths[index],
                "target": target_img,
                "target_path": self.target_paths[index],
                "caption": caption
            }

        except (FileNotFoundError, OSError) as e:
            print(f"Error loading image at index {index}: {e}")
            return None
    
    def __len__(self) -> int:
        return len(self.captions)
                

class GalleryDataset(CIRDataset):
    '''
    Gallery dataset for loading every image referenced in a CIRDataset.
    '''
    
    def __init__(self, data_path: Union[str, os.PathLike], split: Literal['train', 'val', 'test'],
                 dataset: Literal['cirr', 'cirr_r', 'hotels'], preprocess: callable, use_distractors: bool):
        '''
        Initializes the GalleryDataset using the parent class and extracts all images.
        '''
        super().__init__(data_path, split, dataset, preprocess)
        self.image_paths = self.get_all_images(use_distractors=use_distractors)
        
    def __getitem__(self, index) -> dict:
        '''
        Returns a single image and its path from the gallery dataset.
        '''
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert("RGB")
        image = self.preprocess(image)
        
        return {
            "image": image,
            "image_path": image_path
        }
    
    def __len__(self) -> int:
        '''
        Returns the total number of images in the gallery dataset.
        '''
        return len(self.image_paths)


class CIRRDataset(Dataset):
    """
   Copy-paste from https://github.com/miccunifi/SEARLE/blob/main/src/datasets.py
   CIRR dataset class for PyTorch dataloader.
   The dataset can be used in 'relative' or 'classic' mode:
        - In 'classic' mode the dataset yield a dict with keys ['image', 'image_name']
        - In 'relative' mode the dataset yield dict with keys:
            - ['reference_image', 'reference_name', 'target_image', 'target_name', 'relative_caption', 'group_members']
             when split in ['train', 'val']
            - ['reference_image', 'reference_name' 'relative_caption', 'group_members', 'pair_id'] when split == test
    """

    def __init__(self, dataset_path: Union[Path, str], split: Literal['train', 'val', 'test'],
                 mode: Literal['relative', 'classic'], preprocess: callable, no_duplicates: Optional[bool] = False):
        """
        :param dataset_path: path to the CIRR dataset
        :param split: dataset split, should be in ['train', 'val', 'test']
        :param mode: dataset mode, should be in ['relative', 'classic']:
                - In 'classic' mode the dataset yield a dict with keys ['image', 'image_name']
                - In 'relative' mode the dataset yield dict with keys:
                    - ['reference_image', 'reference_name', 'target_image', 'target_name', 'relative_caption',
                    'group_members'] when split in ['train', 'val']
                    - ['reference_image', 'reference_name' 'relative_caption', 'group_members', 'pair_id'] when split == test
        :param preprocess: function which preprocesses the image
        :param no_duplicates: if True, the dataset will not yield duplicate images in relative mode, does not affect classic mode
        """
        dataset_path = Path(dataset_path)
        self.dataset_path = dataset_path
        self.preprocess = preprocess
        self.mode = mode
        self.split = split
        self.no_duplicates = no_duplicates

        if split == "test":
            split = "test1"
            self.split = "test1"

        # Validate inputs
        if split not in ['test1', 'train', 'val']:
            raise ValueError("split should be in ['test1', 'train', 'val']")
        if mode not in ['relative', 'classic']:
            raise ValueError("mode should be in ['relative', 'classic']")

        # get triplets made by (reference_image, target_image, relative caption)
        with open(dataset_path / 'captions' / f'cap.rc2.{split}.json') as f:
            self.triplets = json.load(f)

        # Remove duplicates from triplets
        if self.no_duplicates:
            seen = set()
            new_triplets = []
            for triplet in self.triplets:
                if triplet['reference'] not in seen:
                    seen.add(triplet['reference'])
                    new_triplets.append(triplet)
            self.triplets = new_triplets

        # get a mapping from image name to relative path
        with open(dataset_path / 'image_splits' / f'split.rc2.{split}.json') as f:
            self.name_to_relpath = json.load(f)

        print(f"CIRR {split} dataset in {mode} mode initialized")

    def __getitem__(self, index) -> dict:
        try:
            if self.mode == 'relative':
                group_members = self.triplets[index]['img_set']['members']
                reference_name = self.triplets[index]['reference']
                relative_caption = self.triplets[index]['caption']

                if self.split in ['train', 'val']:
                    reference_image_path = self.dataset_path / self.name_to_relpath[reference_name]
                    reference_image = self.preprocess(Image.open(reference_image_path).convert("RGB"))
                    target_hard_name = self.triplets[index]['target_hard']
                    target_image_path = self.dataset_path / self.name_to_relpath[target_hard_name]
                    target_image = self.preprocess(Image.open(target_image_path).convert("RGB"))

                    return {
                        'reference_image': reference_image,
                        'reference_name': reference_name,
                        'target_image': target_image,
                        'target_name': target_hard_name,
                        'relative_caption': relative_caption,
                        'group_members': group_members
                    }

                elif self.split == 'test1':
                    pair_id = self.triplets[index]['pairid']
                    reference_image_path = self.dataset_path / 'img_raw' /self.name_to_relpath[reference_name]
                    reference_image = self.preprocess(Image.open(reference_image_path).convert("RGB"))
                    return {
                        'reference_image': reference_image,
                        'reference_name': reference_name,
                        'relative_caption': relative_caption,
                        'group_members': group_members,
                        'pair_id': pair_id
                    }

            elif self.mode == 'classic':
                image_name = list(self.name_to_relpath.keys())[index]
                image_path = self.dataset_path / 'img_raw' / self.name_to_relpath[image_name]
                image = self.preprocess(Image.open(image_path).convert("RGB"))

                return {
                    'image': image,
                    'image_name': image_name
                }

            else:
                raise ValueError("mode should be in ['relative', 'classic']")

        except Exception as e:
            print(f"Exception: {e}")

    def __len__(self):
        if self.mode == 'relative':
            return len(self.triplets)
        elif self.mode == 'classic':
            return len(self.name_to_relpath)
        else:
            raise ValueError("mode should be in ['relative', 'classic']")


class CIRCODataset(Dataset):
    """
    CIRCO dataset class for PyTorch.
    The dataset can be used in 'relative' or 'classic' mode:
        - In 'classic' mode the dataset yield a dict with keys ['image', 'image_name']
        - In 'relative' mode the dataset yield dict with keys:
            - ['reference_image', 'reference_name', 'target_image', 'target_name', 'relative_captions', 'shared_concept',
             'gt_img_ids', 'query_id'] when split == 'val'
            - ['reference_image', 'reference_name', 'relative_captions', 'shared_concept', 'query_id'] when split == test
    """

    def __init__(self, dataset_path: Union[str, Path], split: Literal['val', 'test'],
                 mode: Literal['relative', 'classic'], preprocess: callable):
        """
        Args:
            dataset_path (Union[str, Path]): path to CIRCO dataset
            split (str): dataset split, should be in ['test', 'val']
            mode (str): dataset mode, should be in ['relative', 'classic']
            preprocess (callable): function which preprocesses the image
        """

        # Set dataset paths and configurations
        dataset_path = Path(dataset_path)
        self.mode = mode
        self.split = split
        self.preprocess = preprocess
        self.data_path = dataset_path

        # Ensure input arguments are valid
        if mode not in ['relative', 'classic']:
            raise ValueError("mode should be in ['relative', 'classic']")
        if split not in ['test', 'val']:
            raise ValueError("split should be in ['test', 'val']")

        # Load COCO images information
        with open(dataset_path / 'COCO2017_unlabeled' / "annotations" / "image_info_unlabeled2017.json", "r") as f:
            imgs_info = json.load(f)

        self.img_paths = [dataset_path / 'COCO2017_unlabeled' / "unlabeled2017" / img_info["file_name"] for img_info in
                          imgs_info["images"]]
        self.img_ids = [img_info["id"] for img_info in imgs_info["images"]]
        self.img_ids_indexes_map = {str(img_id): i for i, img_id in enumerate(self.img_ids)}

        # get CIRCO annotations
        with open(dataset_path / 'annotations' / f'{split}.json', "r") as f:
            self.annotations: List[dict] = json.load(f)

        # Get maximum number of ground truth images (for padding when loading the images)
        self.max_num_gts = 23  # Maximum number of ground truth images

        print(f"CIRCODataset {split} dataset in {mode} mode initialized")

    def get_target_img_ids(self, index) -> Dict[str, int]:
        """
        Returns the id of the target image and ground truth images for a given query

        Args:
            index (int): id of the query

        Returns:
             Dict[str, int]: dictionary containing target image id and a list of ground truth image ids
        """

        return {
            'target_img_id': self.annotations[index]['target_img_id'],
            'gt_img_ids': self.annotations[index]['gt_img_ids']
        }

    def __getitem__(self, index) -> dict:
        """
        Returns a specific item from the dataset based on the index.

        In 'classic' mode, the dataset yields a dictionary with the following keys: [img, img_id]
        In 'relative' mode, the dataset yields dictionaries with the following keys:
            - [reference_img, reference_img_id, target_img, target_img_id, relative_caption, shared_concept, gt_img_ids,
            query_id]
            if split == val
            - [reference_img, reference_img_id, relative_caption, shared_concept, query_id]  if split == test
        """

        if self.mode == 'relative':
            # Get the query id
            query_id = str(self.annotations[index]['id'])

            # Get relative caption and shared concept
            relative_caption = self.annotations[index]['relative_caption']
            shared_concept = self.annotations[index]['shared_concept']

            # Get the reference image
            reference_img_id = str(self.annotations[index]['reference_img_id'])
            reference_img_path = self.img_paths[self.img_ids_indexes_map[reference_img_id]]
            reference_img = self.preprocess(PIL.Image.open(reference_img_path).convert("RGB"))

            if self.split == 'val':
                # Get the target image and ground truth images
                target_img_id = str(self.annotations[index]['target_img_id'])
                gt_img_ids = [str(x) for x in self.annotations[index]['gt_img_ids']]
                target_img_path = self.img_paths[self.img_ids_indexes_map[target_img_id]]
                target_img = self.preprocess(PIL.Image.open(target_img_path).convert("RGB"))

                # Pad ground truth image IDs with zeros for collate_fn
                gt_img_ids += [''] * (self.max_num_gts - len(gt_img_ids))

                return {
                    'reference_image': reference_img,
                    'reference_name': reference_img_id,
                    'target_image': target_img,
                    'target_name': target_img_id,
                    'relative_caption': relative_caption,
                    'shared_concept': shared_concept,
                    'gt_img_ids': gt_img_ids,
                    'query_id': query_id,
                }

            elif self.split == 'test':
                return {
                    'reference_image': reference_img,
                    'reference_name': reference_img_id,
                    'relative_caption': relative_caption,
                    'shared_concept': shared_concept,
                    'query_id': query_id,
                }

        elif self.mode == 'classic':
            # Get image ID and image path
            img_id = str(self.img_ids[index])
            img_path = self.img_paths[index]

            # Preprocess image and return
            img = self.preprocess(PIL.Image.open(img_path).convert("RGB"))
            return {
                'image': img,
                'image_name': img_id
            }

    def __len__(self):
        """
        Returns the length of the dataset.
        """
        if self.mode == 'relative':
            return len(self.annotations)
        elif self.mode == 'classic':
            return len(self.img_ids)
        else:
            raise ValueError("mode should be in ['relative', 'classic']")
        
def main():
    #test mix dataset
    cfg = yaml.safe_load(open('config.yaml', 'r'))
    preproc = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),  # Resize to 224x224
        transforms.CenterCrop(224),  # Crop to 224x224
        transforms.ToTensor(),  # Convert to tensor (scales to [0, 1])
        transforms.Normalize(mean=[0.481, 0.457, 0.408], std=[0.268, 0.261, 0.275]),  # Normalize
    ])

    cirr_train = CIRDataset(data_path=cfg['cirr_data_path'], split='train', dataset='cirr', preprocess=preproc)
    cirr_r_train = CIRDataset(data_path=cfg['cirr_data_path'], split='train', dataset='cirr_r', preprocess=preproc)

    train_dataset = MixDataset(cirr_train, cirr_r_train, mix_ratio=0.5, preprocess=preproc)

if __name__ == "__main__":
    main()