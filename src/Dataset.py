# ==================== Dataset.py ====================
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import json
import os
from pathlib import Path
from typing import Union, List, Literal, Optional
import random
import yaml


class CIRDataset(Dataset):
    """Enhanced CIR dataset with noun phrase extraction."""
    
    def __init__(self, data_path: Union[str, os.PathLike], split: str,
                 dataset: str, preprocess: callable, parser=None, max_nps: int = 10):
        """Initialize dataset with optional parser for NP extraction."""
        self.split = split
        self.preprocess = preprocess
        self.data_path = data_path
        self.dataset = dataset
        self.parser = parser
        self.max_nps = max_nps
        
        # Load data as in original implementation
        dataset_paths = {
            'cirr': os.path.join(data_path, 'captions', f'cap.rc2.{split}.json'),
            'cirr_r': os.path.join(data_path, 'rewritten_captions', f'rewritten_{split}.json'),
            'hotels': os.path.join(data_path, 'captions', f'hotel_{split}_data.json')
        }
        
        if dataset not in dataset_paths:
            raise ValueError(f"Dataset '{dataset}' not implemented")
        
        with open(dataset_paths[dataset], "r") as f:
            imgs_info = json.load(f)
        
        self.reference_paths, self.target_paths, self.captions = [], [], []
        
        if dataset == 'cirr' and split == 'test':
            split = 'test1'
        
        # Process dataset-specific structure
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
                    self.reference_paths.append(os.path.join(data_path, 'img_raw', mapper[ref_filename].strip('./')))
                    self.target_paths.append(os.path.join(data_path, 'img_raw', mapper[tgt_filename].strip('./')) if tgt_filename else None)
                    self.captions.append(img_info["difference_captions"][0] if isinstance(img_info["difference_captions"], list) else img_info["difference_captions"])
        
        elif dataset == 'hotels':
            img_dir = 'train' if split == 'train' else 'val_test'
            
            for img_info in imgs_info:
                self.reference_paths.append(os.path.join(data_path, 'images', img_dir, *img_info["query_image"].split('/')[-3:]))
                self.target_paths.append(os.path.join(data_path, 'images', img_dir, *img_info["result_image"].split('/')[-3:]))
                self.captions.append(img_info["difference"])
        
        print(f"{split} dataset initialized with {len(self.reference_paths)} samples from {dataset_paths[dataset]}")
        
        # Pre-extract noun phrases if parser is provided
        self.noun_phrases = []
        self.np_spans = []
        
        if self.parser is not None:
            print(f"Pre-extracting noun phrases for {len(self.captions)} captions...")
            for caption in self.captions:
                try:
                    np_info = self.parser.extract_noun_phrases(caption, self.max_nps)
                    self.noun_phrases.append(np_info["nps"])
                    self.np_spans.append(np_info["spans"])
                except Exception as e:
                    print(f"Error extracting NPs: {e}")
                    self.noun_phrases.append([])
                    self.np_spans.append([])
    
    def __getitem__(self, index) -> dict:
        """Returns a sample with noun phrases if available."""
        try:
            reference_img = Image.open(self.reference_paths[index]).convert("RGB")
            target_img = Image.open(self.target_paths[index]).convert("RGB") if self.target_paths[index] else None
            reference_img = self.preprocess(reference_img)
            target_img = self.preprocess(target_img) if target_img else None
            
            caption = self.captions[index]
            
            sample = {
                "reference": reference_img,
                "reference_path": self.reference_paths[index],
                "target": target_img,
                "target_path": self.target_paths[index],
                "caption": caption,
            }
            
            # Add noun phrases if available
            if len(self.noun_phrases) > 0:
                sample["noun_phrases"] = self.noun_phrases[index]
                sample["np_spans"] = self.np_spans[index]
            else:
                sample["noun_phrases"] = []
                sample["np_spans"] = []
            
            return sample
            
        except (FileNotFoundError, OSError) as e:
            print(f"Error loading image at index {index}: {e}")
            return None
    
    def __len__(self) -> int:
        return len(self.captions)


def collate_fn_with_nps(batch):
    """Custom collate function that handles variable-length noun phrases."""
    # Filter out None samples
    batch = [b for b in batch if b is not None]
    
    # Stack tensors
    reference = torch.stack([b["reference"] for b in batch])
    target = torch.stack([b["target"] for b in batch])
    
    # Lists
    captions = [b["caption"] for b in batch]
    reference_paths = [b["reference_path"] for b in batch]
    target_paths = [b["target_path"] for b in batch]
    noun_phrases = [b.get("noun_phrases", []) for b in batch]
    np_spans = [b.get("np_spans", []) for b in batch]
    
    return {
        "reference": reference,
        "target": target,
        "caption": captions,
        "reference_path": reference_paths,
        "target_path": target_paths,
        "noun_phrases": noun_phrases,
        "np_spans": np_spans,
    }


