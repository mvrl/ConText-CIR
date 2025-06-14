import os
import glob
import argparse
import torch
import yaml
import numpy as np
import pytorch_lightning as pl
from torchvision import transforms
import torch.nn.functional as F
from Models import ConText
from Dataset import CIRDataset, GalleryDataset, CIRRDataset, CIRCODataset
from tqdm import tqdm


import json
from typing import List, Tuple
from torch.utils.data import DataLoader
from typing import List, Tuple, Dict
from argparse import ArgumentParser

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cfg = yaml.safe_load(open('config.yaml', 'r'))

def extract_image_features(dataset, model):
    """
    Extract the image features for the given dataset using the given model
    """

    gallery_loader = DataLoader(dataset, batch_size=256, num_workers=10, pin_memory=True, shuffle=False)

    index_features, index_names = [], []
    with torch.no_grad():
        for batch in tqdm(gallery_loader, desc="Gallery Processing"):
            imgs = batch.get('image').to(device)
            names = batch.get('image_name')
            embeddings = model.get_image_features(imgs)
            index_features.append(embeddings)
            index_names.extend(names)

    assert len(index_names) == len(list(set(index_names)))  # Ensure no duplicates

    index_features = torch.cat(index_features)  # Stack all embeddings at once
    index_names = np.array(index_names)

    # Normalize the index features
    index_features = index_features.to(device)
    index_features = F.normalize(index_features, dim=-1).float()

    return index_features, index_names

@torch.no_grad()
def circo_generate_test_submission_file(dataset_path: str, model, preprocess: callable, submission_name: str) -> None:
    """
    Generate the test submission file for the CIRCO dataset given the pseudo tokens
    """

    # Load the CLIP model
    #clip_model, _ = clip.load(clip_model_name, device=device, jit=False)
    #clip_model = clip_model.float().eval().requires_grad_(False)

    # Compute the index features
    classic_test_dataset = CIRCODataset(dataset_path, 'test', 'classic', preprocess)
    index_features, index_names = extract_image_features(classic_test_dataset, model)

    relative_test_dataset = CIRCODataset(dataset_path, 'test', 'relative', preprocess)

    # Get the predictions dict
    queryid_to_retrieved_images = circo_generate_test_dict(relative_test_dataset, model, index_features, index_names)

    submissions_folder_path = os.path.join('./submission', 'circo')
    os.makedirs(submissions_folder_path, exist_ok=True)

    with open(os.path.join(submissions_folder_path, f"{submission_name}.json"), 'w+') as file:
        json.dump(queryid_to_retrieved_images, file, sort_keys=True)


def circo_generate_test_predictions(model, relative_test_dataset: CIRCODataset) -> [torch.Tensor, List[List[str]]]:
    """
    Generate the test prediction features for the CIRCO dataset given the pseudo tokens
    """

    # Create the test dataloader
    relative_test_loader = DataLoader(dataset=relative_test_dataset, batch_size=32, num_workers=10,
                                      pin_memory=False, shuffle=False)

    predicted_features_list = []
    query_ids_list = []

    # Compute the predictions
    for batch in tqdm(relative_test_loader):
        relative_captions = batch['relative_caption']
        query_ids = batch['query_id']
        ref = batch['reference_image'].to(device)

        with torch.no_grad():
            predicted_features = model(ref, relative_captions)

        predicted_features_list.append(predicted_features)
        query_ids_list.extend(query_ids)

    predicted_features = torch.vstack(predicted_features_list)
    predicted_features = F.normalize(predicted_features, dim=-1).float()
    return predicted_features, query_ids_list


def circo_generate_test_dict(relative_test_dataset: CIRCODataset, clip_model, index_features: torch.Tensor,index_names: List[str]) \
        -> Dict[str, List[str]]:
    """
    Generate the test submission dicts for the CIRCO dataset given the pseudo tokens
    """

    # Get the predicted features
    predicted_features, query_ids = circo_generate_test_predictions(clip_model, relative_test_dataset)

    # Normalize the features
    index_features = index_features.float().to(device)
    index_features = F.normalize(index_features, dim=-1)

    # Compute the similarity
    similarity = predicted_features @ index_features.T
    sorted_indices = torch.topk(similarity, dim=-1, k=50).indices.cpu()
    sorted_index_names = np.array(index_names)[sorted_indices]

    # Generate prediction dicts
    queryid_to_retrieved_images = {query_id: query_sorted_names[:50].tolist() for
                                   (query_id, query_sorted_names) in zip(query_ids, sorted_index_names)}

    return queryid_to_retrieved_images

def main():
    parser = ArgumentParser()
    parser.add_argument('--run_name', type=str, default='default', help='Run name')
    parser.add_argument('--submission_name', type=str, default=None, help='Name of the submission')
    parser.add_argument('--path', type=str, default='load_best_val', help='Path to model checkpoint')
    args = parser.parse_args()

    cfg = yaml.safe_load(open('config.yaml', 'r'))

    training_datasets = ''
    match args.run_name.strip('B_').lower():
        case 'soup':
            training_datasets = 'cirr,cirr_r,hotels'
        case 'cirr':
            training_datasets = 'cirr'
        case 'cirr_r':
            training_datasets = 'cirr_r'
        case 'hotels':
            training_datasets = 'hotels'
        case 'cirr_cirr_r':
            training_datasets = 'cirr,cirr_r'
        case _:
            raise ValueError('Invalid run name')

    # Load model checkpoint paths
    paths = []
    if args.path == 'load_best_val':
        model_dir = os.path.join(cfg['model_dir'], f'{training_datasets}/B/{args.run_name}')
        print(f'Loading models from {model_dir}')
        paths = glob.glob(f'{model_dir}/*.ckpt')
        #get val losses
        models = [m for m in paths if 'last' not in m]
        val_losses = np.array([float(m.split('=')[-1].strip('.ckpt').split('-v')[0]) for m in models])
        #only keep top 5 models in terms of val loss
        paths = [models[i] for i in np.argsort(val_losses)[:5]]
        paths.append(os.path.join(model_dir, 'last.ckpt'))
    else:
        paths.append(args.path)

    preproc = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.481, 0.457, 0.408], std=[0.268, 0.261, 0.275]),
    ])

    if args.submission_name is None:
        args.submission_name = args.run_name

    dataset_path = cfg['circo_data_path']
    template = args.submission_name + "_{idx}"
    for idx, path in tqdm(enumerate(paths)):
        model = ConText.load_from_checkpoint(path).eval().to(device)
        circo_generate_test_submission_file(dataset_path, model, preproc, template.format(idx=idx))

if __name__ == '__main__':
    main()