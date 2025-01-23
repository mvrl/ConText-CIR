import pytorch_lightning as pl

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from transformers import AutoProcessor, CLIPVisionModel
from transformers import AutoTokenizer, CLIPTextModel

import stanza
from nltk.tree import Tree
from Extractor.py import *

import einops
from einops import rearrange

class CrossAttention(nn.Module):
        def __init__(self, embed_dim, num_heads):
                super().__init__()
                self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
                self.layer_norm = nn.LayerNorm(embed_dim)

        def forward(self, x):
                # x is of shape (B, T, D)
                x = x.transpose(0, 1)  # Transpose to (T, B, D) for multihead attention
                attn_output, _ = self.multihead_attn(x, x, x)
                attn_output = attn_output.transpose(0, 1)  # Transpose back to (B, T, D)
                return self.layer_norm(attn_output + x)
        
class AttentionPooler(nn.Module):
    def __init__(self, nquery, dim=256, dim_head=64, heads=8, dropout=0.0):
        super().__init__()
        self.nquery = nquery
        self.heads = heads
        self.dim_head = dim_head
        inner_dim = dim_head * heads

        # Learnable queries of shape (nquery, dim)
        self.queries = nn.Parameter(torch.randn(nquery, dim))
        
        # Multi-head attention components
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)
        
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, encoder_outputs):
        """
        encoder_outputs: Tensor of shape (batch_size, seq_len, dim)
        Returns: Pooled output of shape (batch_size, nquery, dim)
        """
        b, _, _ = encoder_outputs.shape
        
        # Apply normalization
        encoder_outputs = self.norm(encoder_outputs)
        
        # Expand learnable queries for batch dimension
        queries = self.queries.unsqueeze(0).expand(b, -1, -1)
        
        # Compute query, key, and value projections
        q = self.to_q(queries)
        k, v = self.to_kv(encoder_outputs).chunk(2, dim=-1)

        # Reshape for multi-head attention
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.heads)
        
        # Scaled dot-product attention
        attn_scores = torch.einsum('b h i d, b h j d -> b h i j', q, k) / (self.dim_head ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = torch.einsum('b h i j, b h j d -> b h i d', attn_weights, v)
        
        # Merge heads and apply output projection
        attn_output = rearrange(attn_output, 'b h n d -> b n (h d)')
        pooled_output = self.to_out(attn_output)
        pooled_output = self.dropout(pooled_output)
        
        return pooled_output


class CrossAttention(nn.Module):
    def __init__(
        self,
        *,
        dim=512,
        context_dim=512,
        dim_head=64,
        heads=8,
        dropout=0.0,
        norm_context=False,
        cosine_sim=False,
        cosine_sim_scale=16
    ):
        super().__init__()
        self.cosine_sim = cosine_sim
        self.scale = cosine_sim_scale if cosine_sim else (dim_head ** -0.5)
        self.heads = heads
        inner_dim = dim_head * heads

        context_dim = context_dim if context_dim is not None else dim

        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if norm_context else nn.Identity()
        self.dropout = nn.Dropout(dropout)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim, bias=False),
            nn.LayerNorm(dim)
        )

    def forward(self, x, context, mask=None):
        b, n, device = *x.shape[:2], x.device

        x = self.norm(x)
        context = self.norm_context(context)

        q, k, v = self.to_q(x), *self.to_kv(context).chunk(2, dim=-1)

        q, k, v = rearrange(q, 'b n (h d) -> b h n d', h=self.heads), \
                  rearrange(k, 'b n (h d) -> b h n d', h=self.heads), \
                  rearrange(v, 'b n (h d) -> b h n d', h=self.heads)

        if self.cosine_sim:
            q = F.normalize(q, dim=-1)
            k = F.normalize(k, dim=-1)

        q, k = q * self.scale, k * self.scale

        sim = torch.einsum('b h i d, b h j d -> b h i j', q, k)
        max_neg_value = -torch.finfo(sim.dtype).max

        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)
            sim = sim.masked_fill(~mask, max_neg_value)

        attn = sim.softmax(dim=-1, dtype=torch.float32)
        attn = attn.to(sim.dtype)

        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

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



class TextEncoder(nn.Module):
        def __init__(self, size="B"):
                super().__init__()
                match size:
                        case "B":
                                self.model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
                                self.tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
                        case "L":
                                self.model = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch32")
                                self.tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-large-patch32")
                        case "H":
                                #OpenCLIP ViT-H/14
                                self.model = CLIPTextModel.from_pretrained('laion/CLIP-ViT-H-14-laion2B-s32B-b79K')
                                self.tokenizer = AutoTokenizer.from_pretrained('laion/CLIP-ViT-H-14-laion2B-s32B-b79K')

        def forward(self, x):
                inputs = self.tokenizer(x, padding=True, return_tensors="pt")
                return self.model(**inputs)

class VisionEncoder(nn.Module):
        def __init__(self, size="B"):
                super().__init__()
                match size:
                        case "B":
                                self.model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
                                self.processor = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
                        case "L":
                                self.model = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch32")
                                self.processor = AutoTokenizer.from_pretrained("openai/clip-vit-large-patch32")
                        case "H":
                                #OpenCLIP ViT-H/14
                                self.model = CLIPVisionModel.from_pretrained('laion/CLIP-ViT-H-14-laion2B-s32B-b79K')
                                self.processor = AutoTokenizer.from_pretrained('laion/CLIP-ViT-H-14-laion2B-s32B-b79K')


        def forward(self, x):
                inputs = self.processor(x, return_tensors="pt")
                return self.model(**inputs)

class ConText(pl.LightningModule):
        def __init__(
                        self, 
                        size="B", 
                        dim=512, 
                        nquery=256, 
                        num_cross_attn_layers=4, 
                        dim_head=64, 
                        heads=8, 
                        dropout=0.0, 
                        learning_rate=2e-5,
                        weight_decay=1e-2
                        ):

                super().__init__()

                self.learning_rate = learning_rate
                self.weight_decay = weight_decay
                
                self.text_encoder = TextEncoder(size)
                self.vision_encoder = VisionEncoder(size)
                
                self.cross_attn_layers = nn.ModuleList([
                        CrossAttention(dim, context_dim=dim, dim_head=dim_head, 
                                                        heads=heads, dropout=dropout)
                        for _ in range(num_cross_attn_layers)
                ])
                
                self.pooler = AttentionPooler(dim, nquery, dim_head, heads, dropout)
        
        def forward(self, text_input, vision_input):
                text_features = self.text_encoder(text_input)
                vision_features = self.vision_encoder(vision_input)
                
                # Apply cross-attention layers sequentially
                for cross_attn in self.cross_attn_layers:
                        vision_features = cross_attn(vision_features, text_features)

                return self.pooler(vision_features)

        def get_attn(self, text_input, vision_input):
                """
                for visualization
                """
                text_features = self.text_encoder(text_input)
                vision_features = self.vision_encoder(vision_input)
                
                maps = []
                for cross_attn in self.cross_attn_layers:
                        q, k, v = cross_attn.to_q(vision_features), *cross_attn.to_kv(text_features).chunk(2, dim=-1)
                        q, k = q * cross_attn.scale, k * cross_attn.scale
                        attn_map = torch.einsum('b h i d, b h j d -> b h i j', q, k).softmax(dim=-1)
                        maps.append(attn_map)
                        vision_features = cross_attn(vision_features, text_features)
                
                return maps
        
        def text_cc_loss(attns, spans, slack=0.1):
                loss = 0
                for idx, span in enumerate(spans):
                        left = span.left
                        right = span.right
                        diff = attns[0][left:right+1] - torch.mean(attns[idx + 1][left:right+1], dim=0)
                        loss += torch.max(F.relu(diff - slack))
                return loss

        def contrastive_loss(vl_embeddings, target_embeddings, query_embeddings=None, temp=0.07):
                """
                InfoNCE loss function. If query embeddings are provided, they are used as hard negatives.
                Otherwise, a standard contrastive InfoNCE loss is computed.

                Args:
                        vl_embeddings (torch.Tensor): Vision-language embeddings of shape (B, D).
                        target_embeddings (torch.Tensor): Target embeddings of shape (B, D).
                        query_embeddings (torch.Tensor, optional): Query (negative) embeddings of shape (B, D). Default is None.
                        temp (float): Temperature parameter for scaling logits.

                Returns:
                        torch.Tensor: The computed contrastive loss.
                """

                vl_embeddings = F.normalize(vl_embeddings, p=2, dim=1)
                target_embeddings = F.normalize(target_embeddings, p=2, dim=1)

                # Positive similarities (B,)
                pos_sim = torch.sum(vl_embeddings * target_embeddings, dim=1, keepdim=True)

                if query_embeddings is not None:
                        query_embeddings = F.normalize(query_embeddings, p=2, dim=1)

                        # Negative similarities (B, B)
                        neg_sim = torch.matmul(vl_embeddings, query_embeddings.T)

                        # (B, B+1)
                        logits = torch.cat([pos_sim, neg_sim], dim=1)
                else:
                        # Compute all-to-all similarities if no hard negatives are provided
                        all_embeddings = torch.cat([vl_embeddings, target_embeddings], dim=0)  # (2B, D)
                        all_sim = torch.matmul(vl_embeddings, all_embeddings.T)  # (B, 2B)

                        # Construct logits with positive similarities in diagonal positions
                        logits = all_sim
                        labels = torch.arange(vl_embeddings.shape[0], device=vl_embeddings.device)

                # Apply temperature scaling
                logits /= temp

                if query_embeddings is not None:
                        # Create labels (positive pair is the first element in each row)
                        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=vl_embeddings.device)

                # Compute contrastive loss using cross-entropy
                loss = F.cross_entropy(logits, labels)

                return loss

        
        def configure_optimizers(self):
                print(f'Initial Learning rate {self.learning_rate}')
                
                params = self.parameters()
                
                self.optim = torch.optim.AdamW(params=params,
                                                lr=self.learning_rate,
                                                weight_decay=self.weight_decay,
                                                betas=(0.9,0.98),
                                                eps=1e-6
                                                )
                                
                self.warm_up_iterations = 5000
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                                optimizer = self.optim,
                                T_0 = self.warm_up_iterations
                )
                return {'optimizer': self.optim,
                                'lr_scheduler': {
                                                'name':'train/lr',
                                                'scheduler': self.scheduler,
                                                'interval': 'step',
                                                'frequency': 1
                                }
                }
