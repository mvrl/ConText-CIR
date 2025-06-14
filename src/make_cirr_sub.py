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
from Dataset import CIRDataset, GalleryDataset, CIRRDataset
from tqdm import tqdm


import json
from typing import List, Tuple
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def cirr_generate_test_predictions(model, relative_test_dataset: CIRRDataset) -> \
        Tuple[torch.Tensor, List[str], List[str], List[List[str]]]:
    """
    Generate the test prediction features for the CIRR dataset given the pseudo tokens
    """

    # Create the test dataloader
    relative_test_loader = DataLoader(dataset=relative_test_dataset, batch_size=256, num_workers=16,shuffle=False,pin_memory=True)

    predicted_features_list = []
    reference_names_list = []
    pair_id_list = []
    group_members_list = []

    # Compute the predictions
    for batch in tqdm(relative_test_loader):
        reference_names = batch['reference_name']
        pairs_id = batch['pair_id']
        relative_captions = batch['relative_caption']
        group_members = batch['group_members']

        group_members = np.array(group_members).T.tolist()
        ref = batch['reference_image'].to(device)

        with torch.no_grad():
            predicted_features = model(ref, relative_captions)
            predicted_features = F.normalize(predicted_features, dim=-1).float()


        predicted_features_list.append(predicted_features)
        reference_names_list.extend(reference_names)
        pair_id_list.extend(pairs_id)
        group_members_list.extend(group_members)

    predicted_features = torch.vstack(predicted_features_list)
    #normalize
    predicted_features = F.normalize(predicted_features, dim=-1).float()

    return predicted_features, reference_names_list, pair_id_list, group_members_list

# Argument Parser
parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default='load_best_val', help='Path to model checkpoint')
parser.add_argument('--backbone_size', type=str, default='B', choices=['B', 'L', 'H'], help='Size of the backbone')
parser.add_argument('--run_name', type=str, default='default', help='Run name')
parser.add_argument('--run_all', action='store_true', help='Run all models in the directory')
parser.add_argument('--submission_name', type=str, default=None, help='Name of the submission')
parser.add_argument('--model_dir', type=str, default='models', help='Directory to load models')
args = parser.parse_args()

cfg = yaml.safe_load(open('config.yaml', 'r'))

# Load model checkpoint paths
paths = []
if args.path == 'load_best_val':
    model_dir = os.path.join(cfg['model_dir'], args.model_dir)
    print(f'Loading models from {model_dir}')
    models = glob.glob(f'{model_dir}/*.ckpt')

    if args.run_all:
        paths = models
    else:
        models = [m for m in models if 'last' not in m]
        val_losses = np.array([float(m.split('=')[-1].strip('.ckpt').split('-v')[0]) for m in models])
        #take top 3
        paths = [models[i] for i in np.argsort(val_losses)[:3]]
else:
    paths.append(args.path)

if args.submission_name is None:
    args.submission_name = args.run_name

# Image Preprocessing
preproc = transforms.Compose([
    transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.481, 0.457, 0.408], std=[0.268, 0.261, 0.275]),
])


gallery_dataset = CIRRDataset(cfg['cirr_data_path'], 'test1', 'classic', preproc)
relative_dataset = CIRRDataset(cfg['cirr_data_path'], 'test1', 'relative', preproc)

# DataLoaders
num_workers = 10
batch_size = 256
gallery_loader = torch.utils.data.DataLoader(gallery_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

for idx, path in enumerate(paths):
    print(f'EVALUATING MODEL FROM {path}')
    model = ConText.load_from_checkpoint(path).eval().to(device)

    # Compute Gallery Embeddings
    print('COMPUTING GALLERY EMBEDDINGS')
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

    predicted_features, reference_names, pairs_id, group_members = \
        cirr_generate_test_predictions(model, relative_test_dataset=relative_dataset)

    # Compute the distances and sort the results
    distances = 1 - predicted_features @ index_features.T
    sorted_indices = torch.argsort(distances, dim=-1).cpu()
    sorted_index_names = np.array(index_names)[sorted_indices]

    # Delete the reference image from the results
    reference_mask = torch.tensor(
        sorted_index_names != np.repeat(np.array(reference_names), len(index_names)).reshape(len(sorted_index_names),
                                                                                             -1))
    sorted_index_names = sorted_index_names[reference_mask].reshape(sorted_index_names.shape[0],
                                                                    sorted_index_names.shape[1] - 1)
    # Compute the subset predictions
    group_members = np.array(group_members)
    group_mask = (sorted_index_names[..., None] == group_members[:, None, :]).sum(-1).astype(bool)
    sorted_group_names = sorted_index_names[group_mask].reshape(sorted_index_names.shape[0], -1)

    # Generate prediction dicts
    pairid_to_retrieved_images = {str(int(pair_id)): prediction[:50].tolist() for (pair_id, prediction) in
                                  zip(pairs_id, sorted_index_names)}
    pairid_to_group_retrieved_images = {str(int(pair_id)): prediction[:3].tolist() for (pair_id, prediction) in
                                        zip(pairs_id, sorted_group_names)}

    submission = {
        'version': 'rc2',
        'metric': 'recall'
    }
    group_submission = {
        'version': 'rc2',
        'metric': 'recall_subset'
    }

    submission.update(pairid_to_retrieved_images)
    group_submission.update(pairid_to_group_retrieved_images)

    submissions_folder_path = os.path.join('./submission', 'cirr')
    os.makedirs(submissions_folder_path, exist_ok=True)

    with open(os.path.join(submissions_folder_path, f"{args.submission_name}_{idx}.json"), 'w+') as file:
        json.dump(submission, file, sort_keys=True)

    with open(os.path.join(submissions_folder_path, f"subset_{args.submission_name}_{idx}.json"), 'w+') as file:
        json.dump(group_submission, file, sort_keys=True)
