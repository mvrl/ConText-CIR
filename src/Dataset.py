"""Backward-compatibility shim: the dataset classes now live in cir_dataset.py."""
from cir_dataset import (  # noqa: F401
    CIRTripletDataset,
    CIRRDataset,
    CIRCODataset,
    collate_fn_with_nps,
)
