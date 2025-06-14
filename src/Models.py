import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from transformers import CLIPTextModel, CLIPTokenizer
from transformers import CLIPVisionModel, CLIPImageProcessor    

import einops
from einops import rearrange
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiheadCrossAttention(nn.Module):
    def __init__(self, query_embed_dim, key_value_embed_dim, num_heads, dropout=0.0, learnable_scale=True):
        super().__init__()
        self.query_embed_dim = query_embed_dim
        self.key_value_embed_dim = key_value_embed_dim
        self.num_heads = num_heads
        self.query_head_dim = query_embed_dim // num_heads
        self.kv_head_dim = key_value_embed_dim // num_heads
        assert self.query_head_dim * num_heads == query_embed_dim, "query_embed_dim must be divisible by num_heads"
        assert self.kv_head_dim * num_heads == key_value_embed_dim, "key_value_embed_dim must be divisible by num_heads"
        
        # Linear projections for query, key, and value embeddings
        self.q_proj = nn.Linear(query_embed_dim, query_embed_dim)
        self.k_proj = nn.Linear(key_value_embed_dim, query_embed_dim)  # project KV to query dimension
        self.v_proj = nn.Linear(key_value_embed_dim, query_embed_dim)  # project KV to query dimension
        self.out_proj = nn.Linear(query_embed_dim, query_embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        # Learnable or configurable scaling factor
        if learnable_scale:
            self.scale = nn.Parameter(torch.ones(1) * (query_embed_dim ** -0.5))
        else:
            self.scale = query_embed_dim ** -0.5

    def forward(self, q, kv):
        """
        q: Tensor of shape (batch_size, query_len, query_embed_dim)
        kv: Tensor of shape (batch_size, kv_len, key_value_embed_dim)
        """
        batch_size, q_len, _ = q.shape
        _, kv_len, _ = kv.shape
        
        # Linear projections
        q = self.q_proj(q).view(batch_size, q_len, self.num_heads, self.query_head_dim).transpose(1, 2)
        k = self.k_proj(kv).view(batch_size, kv_len, self.num_heads, self.query_head_dim).transpose(1, 2)
        v = self.v_proj(kv).view(batch_size, kv_len, self.num_heads, self.query_head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Softmax normalization of attention weights
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Compute attention output
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, q_len, self.query_embed_dim)
        
        return self.out_proj(attn_output), attn_weights


class AttentionPooler(nn.Module):
    def __init__(self, embed_dim, num_heads, num_queries):
        super().__init__()
        self.num_queries = num_queries
        self.queries = nn.Parameter(torch.randn(1, num_queries, embed_dim))
        self.attention = MultiheadCrossAttention(embed_dim, embed_dim, num_heads)
    
    def forward(self, encoder_outputs):
        """
        encoder_outputs: Tensor of shape (batch_size, seq_len, embed_dim)
        """
        batch_size = encoder_outputs.shape[0]
        q = self.queries.expand(batch_size, -1, -1)
        pooled_output, _ = self.attention(q, encoder_outputs)
        return pooled_output


class MLP(nn.Module):

        def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
                super().__init__()
                self.output_dim = output_dim
                self.num_layers = num_layers
                h = [hidden_dim] * (num_layers - 1)
                self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
                self.bns = nn.ModuleList(nn.BatchNorm1d(k) for k in h + [output_dim])

        def forward(self, x):
                B, N, D = x.size()
                x = x.reshape(B*N, D)
                for i, (bn, layer) in enumerate(zip(self.bns, self.layers)):
                        x = F.relu(bn(layer(x))) if i < self.num_layers - 1 else layer(x)
                x = x.view(B, N, self.output_dim)
                return x

class TextEncoder(pl.LightningModule):
        def __init__(self, size="B"):
                super().__init__()
                if size == "B":
                        self.model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
                        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
                elif size == "L":
                        self.model = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
                        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
                elif size == "H":
                        self.model = CLIPTextModel.from_pretrained("laion/clip-ViT-H-14-laion2B-s32B-b79K")
                        self.tokenizer = CLIPTokenizer.from_pretrained("laion/clip-ViT-H-14-laion2B-s32B-b79K")

        def forward(self, texts):
                inputs = self.tokenizer(texts, padding="max_length", return_tensors="pt", truncation=True).to(self.device)
                outputs = self.model(**inputs)
                return outputs.last_hidden_state  # Shape: (batch_size, seq_len, embedding_dim)


class VisionEncoder(nn.Module):
    def __init__(self, size="B"):
        super().__init__()
        if size == "B":
            self.model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
            self.processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
        elif size == "L":
            self.model = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch32")
            self.processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch32")
        elif size == "H":
            self.model = CLIPVisionModel.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
            self.processor = CLIPImageProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")

    def forward(self, images, get_embeddings=False):
        # Preprocess images using the processor
        inputs = {'pixel_values' : images}
        
        # Pass through the model
        outputs = self.model(**inputs)

        # If get_embeddings is True, return image embeddings, otherwise return the patch representations
        if get_embeddings:
            return outputs.pooler_output  # Image embeddings are present here
        return outputs.last_hidden_state  # Patch representations: (batch_size, num_patches+1, embedding_dim)
        

class ConText(pl.LightningModule):
        def __init__(
                self, 
                args,
                num_cross_attn_layers=4, 
                heads=8, 
                dropout=0.0,
                weight_decay=1e-3,
                temperature=0.07,
                steps_per_epoch=None,
                max_epochs=None,
                warmup_steps=500,
        ):
                super().__init__()

                self.steps_per_epoch = steps_per_epoch
                self.max_epochs = max_epochs
                self.T_mult = 2
                self.size = args.backbone_size
                self.learning_rate = args.learning_rate
                self.temperature = temperature
                self.warmup_steps = warmup_steps

                self.save_hyperparameters()  # Auto-saves hyperparameters

                self.weight_decay = weight_decay

                self.text_encoder = TextEncoder(self.size)
                self.vision_encoder = VisionEncoder(self.size)

                self.cross_attn_layers = nn.ModuleList([
                        MultiheadCrossAttention(query_embed_dim=768, key_value_embed_dim=512, num_heads=heads, dropout=dropout)
                        for _ in range(num_cross_attn_layers)
                ])

                self.attn_pooler = AttentionPooler(embed_dim=768, num_heads=heads, num_queries=1)

        def forward(self, vision_input, text_input):

                text_features = self.text_encoder(text_input)  # Shape: (B, 77, D)
                vision_features = self.vision_encoder(vision_input)  # Shape: (B, N, D)

                #import code; code.interact(local=dict(globals(), **locals()))

                # Apply cross-attention layers sequentially
                for cross_attn in self.cross_attn_layers:
                        vision_features = cross_attn(vision_features, text_features)[0]

                pooled_features = self.attn_pooler(vision_features)  # Shape: (B, 1, D)
                pooled_features = pooled_features.squeeze(1)  # Shape: (B, D)
                return pooled_features


        def infonce(self, features_a, features_b, normalize=True):
                """
                standard infonce loss with implied diagonal positive pairs
                
                args:
                features_a: tensor of shape [batch_size, dim]
                features_b: tensor of shape [batch_size, dim]
                normalize: whether to L2-normalize the features 
                
                """

                batch_size = features_a.shape[0]
                
                # L2 normalization if requested
                if normalize:
                        features_a = F.normalize(features_a, dim=1)
                        features_b = F.normalize(features_b, dim=1)
                
                # Calculate similarity matrix - contains similarities between all possible pairs
                # Dividing by temperature sharpens the distribution
                similarity_matrix = torch.matmul(features_a, features_b.T) / self.temperature
                
                # Labels are the diagonal indices (positive pairs)
                labels = torch.arange(batch_size, device=features_a.device)
                
                # Compute loss in both directions (a→b and b→a) for symmetry
                loss_a = F.cross_entropy(similarity_matrix, labels)
                loss_b = F.cross_entropy(similarity_matrix.T, labels)
                
                # Average the loss from both directions
                loss = (loss_a + loss_b) / 2.0
                
                return loss
        
        
        def configure_optimizers(self):

                optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate,
                weight_decay=self.weight_decay,
                betas=(0.9, 0.98),
                eps=1e-8)

                if self.steps_per_epoch is None:
                        raise ValueError("steps_per_epoch is not set. Make sure your dataloader has __len__ defined or pass it via args.")

                # Warmup scheduler: linearly increases the LR from 0 to the base LR over warmup_steps.
                # (Requires PyTorch 1.11+)
                warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                        optimizer, start_factor=1e-8, end_factor=1.0, total_iters=self.warmup_steps
                )

                # Cosine Annealing Warm Restarts scheduler after warmup.
                cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                        optimizer, T_0=2500, T_mult=self.T_mult, eta_min=0.0
                )

                # Chain the warmup scheduler and the cosine annealing with warm restarts scheduler.
                scheduler = torch.optim.lr_scheduler.SequentialLR(
                        optimizer,
                        schedulers=[warmup_scheduler, cosine_scheduler],
                        milestones=[self.warmup_steps]
                )

                self.optimizer = optimizer
                self.scheduler = scheduler

                return {
                        'optimizer': self.optimizer,
                        'lr_scheduler': {
                        'scheduler': self.scheduler,
                        'interval': 'step',  # Update scheduler at every step
                        'frequency': 1,
                        'name': 'lr_scheduler'
                        }
                }

        def get_image_features(self, image):
                embeds = self.vision_encoder(image, get_embeddings=True)
                return embeds
        
        def training_step(self, batch, batch_idx):
                
                query_image = batch["reference"]  # Query Image
                target_image = batch["target"]    # Target Image
                diff_text = batch["caption"]      # Modification Text

                # Ensure tensors are on the correct device
                query_image = query_image.to(self.device)
                target_image = target_image.to(self.device)

                # Compute VL Embeddings: Query Image + Difference Text
                vl_embeddings = self.forward(query_image, diff_text)  # Shape: (B, D)

                # Compute Target Embeddings: Target Image + Empty Text
                # target_embeddings = self.forward(self.empty_text_input.copy().to(self.device), target_image)  # Shape: (B, D)
                target_embeddings = self.vision_encoder(target_image, get_embeddings=True)  # Shape: (B, D)
                # target_embeddings = self.forward("", target_image)  # Shape: (B, D)

                # Compute contrastive loss
                # loss = self.contrastive_loss(vl_embeddings, target_embeddings, query_embeddings)
                loss = self.infonce(vl_embeddings, target_embeddings)

                # Logging loss for monitoring
                self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
                return loss

        
        def validation_step(self, batch, batch_idx):
                """
                Validation step for composed image retrieval.
                """
                query_image = batch["reference"]  # Query Image
                target_image = batch["target"]    # Target Image
                diff_text = batch["caption"]      # Modification Text

                # Ensure tensors are on the correct device
                query_image = query_image.to(self.device)
                target_image = target_image.to(self.device)

                # Compute VL Embeddings: Query Image + Difference Text
                vl_embeddings = self.forward(query_image, diff_text)  # Shape: (B, D)

                # Compute Target Embeddings: Target Image + Empty Text
                # target_embeddings = self.forward(self.empty_text_input.copy().to(self.device), target_image)  # Shape: (B, D)
                target_embeddings = self.vision_encoder(target_image, get_embeddings=True)
                # Compute contrastive loss
                val_loss = self.infonce(vl_embeddings, target_embeddings)

                # Logging validation loss
                self.log("val_loss", val_loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

                return val_loss

