import json
from pathlib import Path
from typing import Union, List, Dict, Literal

import PIL
import PIL.Image
import torch.utils.data
import torchvision
from torch.utils.data import Dataset


class CIRDataset(Dataset):
    """
    General CIR dataset
    """

    def __init__(self, data_path: Union[str, Path], split: Literal['train', 'val', 'test'],
                 preprocess: callable):
        """
        Args:
            data_path (Union[str, Path]): path to CIRCO dataset
            split (str): dataset split, should be in ['train', 'test', 'val']
            preprocess (callable): function which preprocesses the image
        """

        # Set dataset paths and configurations
        data_path = Path(data_path)
        self.split = split
        self.preprocess = preprocess
        self.data_path = data_path

        # Ensure input arguments are valid
        if split not in ['train', 'test', 'val']:
            raise ValueError("split should be in ['test', 'val']")

        # Load COCO images information
        with open(data_path / 'image_info.json', "r") as f:
            imgs_info = json.load(f)

        self.img_paths = [data_path / 'images' / img_info["file_name"] for img_info in
                          imgs_info["images"]]
        self.img_ids = [img_info["id"] for img_info in imgs_info["images"]]
        self.img_ids_indexes_map = {str(img_id): i for i, img_id in enumerate(self.img_ids)}

        # get annotations
        with open(data_path / 'annotations' / f'{split}.json', "r") as f:
            self.annotations: List[dict] = json.load(f)

        # Get maximum number of ground truth images (for padding when loading the images)
        self.max_num_gts = 23  # Maximum number of ground truth images

        print(f"Dataset initialized")

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
    """

        # Get the query id
        query_id = str(self.annotations[index]['id'])

        # Get relative caption and shared concept
        relative_caption = self.annotations[index]['relative_caption']
        shared_concept = self.annotations[index]['shared_concept']

        # Get the reference image
        reference_img_id = str(self.annotations[index]['reference_img_id'])
        reference_img_path = self.img_paths[self.img_ids_indexes_map[reference_img_id]]
        reference_img = self.preprocess(PIL.Image.open(reference_img_path))

        if self.split == 'val':
            # Get the target image and ground truth images
            target_img_id = str(self.annotations[index]['target_img_id'])
            gt_img_ids = [str(x) for x in self.annotations[index]['gt_img_ids']]
            target_img_path = self.img_paths[self.img_ids_indexes_map[target_img_id]]
            target_img = self.preprocess(PIL.Image.open(target_img_path))

            # Pad ground truth image IDs with zeros for collate_fn
            gt_img_ids += [''] * (self.max_num_gts - len(gt_img_ids))

            return {
                'reference_img': reference_img,
                'reference_imd_id': reference_img_id,
                'target_img': target_img,
                'target_img_id': target_img_id,
                'relative_caption': relative_caption,
                'shared_concept': shared_concept,
                'gt_img_ids': gt_img_ids,
                'query_id': query_id,
            }

        elif self.split == 'test':
            return {
                'reference_img': reference_img,
                'reference_imd_id': reference_img_id,
                'relative_caption': relative_caption,
                'shared_concept': shared_concept,
                'query_id': query_id,
            }

    def __len__(self):
        """
        Returns the length of the dataset.
        """
        return len(self.annotations)