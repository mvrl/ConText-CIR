
# ==================== Models.py ====================
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModel, CLIPImageProcessor
import math
from einops import rearrange
from typing import Dict


class MultiheadCrossAttention(nn.Module):
    def __init__(self, query_embed_dim, key_value_embed_dim, num_heads, dropout=0.0, learnable_scale=True):
        super().__init__()
        self.query_embed_dim = query_embed_dim
        self.key_value_embed_dim = key_value_embed_dim
        self.num_heads = num_heads
        self.query_head_dim = query_embed_dim // num_heads
        self.kv_head_dim = key_value_embed_dim // num_heads
        
        assert self.query_head_dim * num_heads == query_embed_dim
        assert self.kv_head_dim * num_heads == key_value_embed_dim
        
        # Project to same dimension for compatibility
        self.q_proj = nn.Linear(query_embed_dim, query_embed_dim)
        self.k_proj = nn.Linear(key_value_embed_dim, query_embed_dim)
        self.v_proj = nn.Linear(key_value_embed_dim, query_embed_dim)
        self.out_proj = nn.Linear(query_embed_dim, query_embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        if learnable_scale:
            self.scale = nn.Parameter(torch.ones(1) * (query_embed_dim ** -0.5))
        else:
            self.scale = query_embed_dim ** -0.5

    def forward(self, q, kv, return_attention=False):
        batch_size, q_len, _ = q.shape
        _, kv_len, _ = kv.shape
        
        # Linear projections
        q = self.q_proj(q).view(batch_size, q_len, self.num_heads, self.query_head_dim).transpose(1, 2)
        k = self.k_proj(kv).view(batch_size, kv_len, self.num_heads, self.query_head_dim).transpose(1, 2)
        v = self.v_proj(kv).view(batch_size, kv_len, self.num_heads, self.query_head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Compute attention output
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, q_len, self.query_embed_dim)
        
        output = self.out_proj(attn_output)
        
        if return_attention:
            return output, attn_weights
        return output


class AttentionPooler(nn.Module):
    def __init__(self, embed_dim, num_heads, num_queries):
        super().__init__()
        self.num_queries = num_queries
        self.queries = nn.Parameter(torch.randn(1, num_queries, embed_dim))
        self.attention = MultiheadCrossAttention(embed_dim, embed_dim, num_heads)
    
    def forward(self, encoder_outputs):
        batch_size = encoder_outputs.shape[0]
        q = self.queries.expand(batch_size, -1, -1)
        pooled_output = self.attention(q, encoder_outputs)
        return pooled_output


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
            self.model = CLIPTextModel.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
            self.tokenizer = CLIPTokenizer.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")

    def forward(self, texts):
        inputs = self.tokenizer(texts, padding="max_length", return_tensors="pt", truncation=True).to(self.device)
        outputs = self.model(**inputs)
        return outputs.last_hidden_state


class VisionEncoder(nn.Module):
    def __init__(self, size="B"):
        super().__init__()
        if size == "B":
            self.model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
            self.processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
        elif size == "L":
            self.model = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14")
            self.processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
        elif size == "H":
            self.model = CLIPVisionModel.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
            self.processor = CLIPImageProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")

    def forward(self, images, get_embeddings=False):
        inputs = {'pixel_values': images}
        outputs = self.model(**inputs)
        
        if get_embeddings:
            return outputs.pooler_output
        return outputs.last_hidden_state


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
        lambda_cc=0.08,
        epsilon_cc=0.01,
        max_nps=10,
    ):
        super().__init__()
        
        self.save_hyperparameters()
        
        self.steps_per_epoch = steps_per_epoch
        self.max_epochs = max_epochs
        self.T_mult = 2
        self.size = args.backbone_size
        self.learning_rate = args.learning_rate
        self.temperature = temperature
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay
        self.lambda_cc = lambda_cc
        self.epsilon_cc = epsilon_cc
        self.max_nps = max_nps
        
        # Initialize encoders
        self.text_encoder = TextEncoder(self.size)
        self.vision_encoder = VisionEncoder(self.size)
        
        # Get embedding dimensions
        if self.size == "B":
            self.text_dim = 512
            self.vision_dim = 768
        elif self.size == "L":
            self.text_dim = 768
            self.vision_dim = 1024
        else:  # H
            self.text_dim = 1024
            self.vision_dim = 1280
            
        # Cross-attention layers
        self.cross_attn_layers = nn.ModuleList([
            MultiheadCrossAttention(
                query_embed_dim=self.vision_dim, 
                key_value_embed_dim=self.text_dim, 
                num_heads=heads, 
                dropout=dropout
            )
            for _ in range(num_cross_attn_layers)
        ])
        
        # Attention pooler
        self.attn_pooler = AttentionPooler(
            embed_dim=self.vision_dim, 
            num_heads=heads, 
            num_queries=1
        )
        
        # Initialize parser for noun phrase extraction
        from Parser import Parser
        self.parser = Parser(self.text_encoder.model, self.text_encoder.tokenizer)

    def forward(self, vision_input, text_input, return_attention=False):
        # Encode text and vision
        text_features = self.text_encoder(text_input)  # (B, 77, D_text)
        vision_features = self.vision_encoder(vision_input)  # (B, N, D_vision)
        
        # Store attention maps if needed
        attention_maps = []
        
        # Apply cross-attention layers sequentially
        for cross_attn in self.cross_attn_layers:
            if return_attention:
                vision_features, attn = cross_attn(vision_features, text_features, return_attention=True)
                attention_maps.append(attn)
            else:
                vision_features = cross_attn(vision_features, text_features)
        
        # Pool features
        pooled_features = self.attn_pooler(vision_features)  # (B, 1, D)
        pooled_features = pooled_features.squeeze(1)  # (B, D)
        
        if return_attention:
            return pooled_features, attention_maps
        return pooled_features

    def get_cross_attention_for_tokens(self, vision_input, text_input, token_indices):
        """Get cross-attention weights for specific tokens."""
        _, attention_maps = self.forward(vision_input, text_input, return_attention=True)
        
        # Average attention across layers and heads
        # attention_maps: list of (B, heads, vision_tokens, text_tokens)
        avg_attention = torch.stack(attention_maps).mean(dim=0)  # (B, heads, V, T)
        avg_attention = avg_attention.mean(dim=1)  # (B, V, T)
        
        # Extract attention for specific tokens
        token_attention = avg_attention[:, :, token_indices]  # (B, V, num_tokens)
        return token_attention.mean(dim=-1)  # (B, V)

    def compute_text_cc_loss(self, batch):
        """Compute text concept-consistency loss."""
        images = batch["reference"]
        full_texts = batch["caption"]
        nps_list = batch["noun_phrases"]
        spans_list = batch["np_spans"]
        
        batch_size = images.shape[0]
        total_loss = 0.0
        num_valid_nps = 0
        
        for i in range(batch_size):
            nps = nps_list[i]
            spans = spans_list[i]
            
            if not nps or len(nps) == 0:
                continue
                
            full_text = full_texts[i]
            image = images[i:i+1]  # Keep batch dimension
            
            for np_text, span in zip(nps, spans):
                if not np_text:  # Skip empty NPs
                    continue
                    
                # Get token indices for NP in full text
                tokens = self.text_encoder.tokenizer(full_text, return_tensors="pt")
                np_tokens = self.text_encoder.tokenizer(np_text, return_tensors="pt")
                
                # Find token positions (simplified - in practice need better alignment)
                token_start = span.left + 1  # +1 for CLS token
                token_end = min(span.right + 1, 77)  # Cap at max length
                token_indices = list(range(token_start, token_end))
                
                # Get attention for NP tokens in context
                attn_in_context = self.get_cross_attention_for_tokens(
                    image, [full_text], token_indices
                )
                
                # Get attention for isolated NP
                attn_isolated = self.get_cross_attention_for_tokens(
                    image, [np_text], list(range(1, len(np_tokens['input_ids'][0])-1))
                )
                
                # Compute L1 difference with ReLU and epsilon
                diff = F.relu(torch.abs(attn_in_context - attn_isolated) - self.epsilon_cc)
                total_loss += diff.mean()
                num_valid_nps += 1
        
        if num_valid_nps > 0:
            return total_loss / num_valid_nps
        return torch.tensor(0.0, device=images.device)

    def infonce(self, features_a, features_b, normalize=True):
        """Standard InfoNCE loss with implied diagonal positive pairs."""
        batch_size = features_a.shape[0]
        
        if normalize:
            features_a = F.normalize(features_a, dim=1)
            features_b = F.normalize(features_b, dim=1)
        
        similarity_matrix = torch.matmul(features_a, features_b.T) / self.temperature
        labels = torch.arange(batch_size, device=features_a.device)
        
        loss_a = F.cross_entropy(similarity_matrix, labels)
        loss_b = F.cross_entropy(similarity_matrix.T, labels)
        
        return (loss_a + loss_b) / 2.0
    
    def get_image_features(self, image):
        return self.vision_encoder(image, get_embeddings=True)
    
    def training_step(self, batch, batch_idx):
        query_image = batch["reference"]
        target_image = batch["target"]
        diff_text = batch["caption"]
        
        # Compute embeddings
        vl_embeddings = self.forward(query_image, diff_text)
        target_embeddings = self.vision_encoder(target_image, get_embeddings=True)
        
        # Add query image as hard negative
        query_embeddings = self.vision_encoder(query_image, get_embeddings=True)
        
        # Compute contrastive loss with hard negative
        all_target_embeddings = torch.cat([target_embeddings, query_embeddings], dim=0)
        vl_embeddings_doubled = torch.cat([vl_embeddings, vl_embeddings], dim=0)
        
        loss_cont = self.infonce(vl_embeddings_doubled, all_target_embeddings)
        
        # Compute text CC loss if lambda > 0
        if self.lambda_cc > 0:
            loss_cc = self.compute_text_cc_loss(batch)
            loss = loss_cont + self.lambda_cc * loss_cc
            self.log("train_loss_cc", loss_cc, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        else:
            loss = loss_cont
        
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train_loss_cont", loss_cont, on_step=True, on_epoch=True, sync_dist=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        query_image = batch["reference"]
        target_image = batch["target"]
        diff_text = batch["caption"]
        
        # Compute embeddings
        vl_embeddings = self.forward(query_image, diff_text)
        target_embeddings = self.vision_encoder(target_image, get_embeddings=True)
        
        # Compute validation loss
        val_loss = self.infonce(vl_embeddings, target_embeddings)
        
        self.log("val_loss", val_loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return val_loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.98),
            eps=1e-8
        )
        
        if self.steps_per_epoch is None:
            raise ValueError("steps_per_epoch is not set.")
        
        # Warmup scheduler
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1e-8, end_factor=1.0, total_iters=self.warmup_steps
        )
        
        # Cosine annealing with warm restarts
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=2500, T_mult=self.T_mult, eta_min=0.0
        )
        
        # Sequential scheduler
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[self.warmup_steps]
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1,
                'name': 'lr_scheduler'
            }
        }


