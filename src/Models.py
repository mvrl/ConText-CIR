import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from transformers import (
    CLIPModel, CLIPTextModel, CLIPTokenizer, CLIPVisionModel, CLIPImageProcessor,
)

_HF_NAME = {
    "B": "openai/clip-vit-base-patch32",
    "B16": "openai/clip-vit-base-patch16",
    "L": "openai/clip-vit-large-patch14",
    "H": "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
}


class MultiheadCrossAttention(nn.Module):
    def __init__(self, query_embed_dim, key_value_embed_dim, num_heads, dropout=0.0):
        super().__init__()
        self.query_embed_dim = query_embed_dim
        self.num_heads = num_heads
        self.head_dim = query_embed_dim // num_heads
        assert self.head_dim * num_heads == query_embed_dim
        assert (key_value_embed_dim // num_heads) * num_heads == key_value_embed_dim

        self.q_proj = nn.Linear(query_embed_dim, query_embed_dim)
        self.k_proj = nn.Linear(key_value_embed_dim, query_embed_dim)
        self.v_proj = nn.Linear(key_value_embed_dim, query_embed_dim)
        self.out_proj = nn.Linear(query_embed_dim, query_embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(self, q, kv, return_attention=False, key_padding_mask=None):
        B, q_len, _ = q.shape
        kv_len = kv.shape[1]
        q = self.q_proj(q).view(B, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(kv).view(B, kv_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(kv).view(B, kv_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if key_padding_mask is not None:
            pad = (key_padding_mask == 0)[:, None, None, :]
            attn = attn.masked_fill(pad, torch.finfo(attn.dtype).min)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, q_len, self.query_embed_dim)
        out = self.out_proj(out)
        if return_attention:
            return out, attn
        return out


class AttentionPooler(nn.Module):
    def __init__(self, embed_dim, num_heads, num_queries=1):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(1, num_queries, embed_dim) * (embed_dim ** -0.5))
        self.attention = MultiheadCrossAttention(embed_dim, embed_dim, num_heads)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, encoder_outputs):
        q = self.queries.expand(encoder_outputs.shape[0], -1, -1)
        return self.norm(self.attention(q, encoder_outputs))


class TextEncoder(pl.LightningModule):
    def __init__(self, size="B16"):
        super().__init__()
        name = _HF_NAME[size]
        self.model = CLIPTextModel.from_pretrained(name)
        self.tokenizer = CLIPTokenizer.from_pretrained(name)

    def forward(self, texts):
        inputs = self.tokenizer(texts, padding="max_length", return_tensors="pt",
                                truncation=True).to(self.device)
        outputs = self.model(**inputs)
        # also return the mask so cross-attention can mask PAD keys (captions are padded to 77)
        return outputs.last_hidden_state, inputs["attention_mask"]


class VisionEncoder(nn.Module):
    def __init__(self, size="B16"):
        super().__init__()
        name = _HF_NAME[size]
        self.model = CLIPVisionModel.from_pretrained(name)
        self.processor = CLIPImageProcessor.from_pretrained(name)

    def forward(self, images, get_embeddings=False, return_both=False):
        outputs = self.model(pixel_values=images)
        if return_both:
            return outputs.last_hidden_state, outputs.pooler_output
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
        epsilon_cc=0.0,
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
        self.lr_schedule = getattr(args, 'lr_schedule', 'cosine_restarts')
        self.cosine_t0 = int(getattr(args, 'cosine_t0', 2500))
        self.backbone_lr_scale = float(getattr(args, 'backbone_lr_scale', 1.0))
        self.grad_checkpoint = bool(getattr(args, 'grad_checkpoint', False))

        self.text_encoder = TextEncoder(self.size)
        self.vision_encoder = VisionEncoder(self.size)

        if self.size in ("B", "B16"):
            self.text_dim, self.vision_dim = 512, 768
        elif self.size == "L":
            self.text_dim, self.vision_dim = 768, 1024
        else:  # H
            self.text_dim, self.vision_dim = 1024, 1280

        # shared retrieval space: CLIP's visual_projection for candidates, a learned one for the query
        clip = CLIPModel.from_pretrained(_HF_NAME[self.size])
        self.proj_dim = clip.config.projection_dim
        self.visual_projection = nn.Linear(self.vision_dim, self.proj_dim, bias=False)
        self.visual_projection.load_state_dict(clip.visual_projection.state_dict())
        del clip
        self.query_projection = nn.Linear(self.vision_dim, self.proj_dim)

        if self.grad_checkpoint:
            for enc in (self.text_encoder.model, self.vision_encoder.model):
                try:
                    enc.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
                except TypeError:
                    enc.gradient_checkpointing_enable()

        # residual cross-attn fusion; out_proj + 2nd FFN linear zero-init -> identity block at step 0
        self.cross_attn_layers = nn.ModuleList([
            MultiheadCrossAttention(self.vision_dim, self.text_dim, heads, dropout)
            for _ in range(num_cross_attn_layers)
        ])
        for ca in self.cross_attn_layers:
            nn.init.zeros_(ca.out_proj.weight)
            nn.init.zeros_(ca.out_proj.bias)
        self.fusion_attn_norms = nn.ModuleList([nn.LayerNorm(self.vision_dim)
                                                for _ in range(num_cross_attn_layers)])
        self.fusion_ffn_norms = nn.ModuleList([nn.LayerNorm(self.vision_dim)
                                               for _ in range(num_cross_attn_layers)])
        self.fusion_ffns = nn.ModuleList()
        for _ in range(num_cross_attn_layers):
            lin2 = nn.Linear(self.vision_dim * 4, self.vision_dim)
            nn.init.zeros_(lin2.weight)
            nn.init.zeros_(lin2.bias)
            self.fusion_ffns.append(nn.Sequential(
                nn.Linear(self.vision_dim, self.vision_dim * 4), nn.GELU(), nn.Dropout(dropout), lin2,
            ))

        self.attn_pooler = AttentionPooler(self.vision_dim, heads, num_queries=1)

        from Parser import Parser
        self.parser = Parser(self.text_encoder.model, self.text_encoder.tokenizer)

    def forward(self, vision_input, text_input, return_attention=False, return_ref_embedding=False):
        text_features, text_mask = self.text_encoder(text_input)
        vision_features, ref_embedding = self.vision_encoder(vision_input, return_both=True)

        attention_maps = []
        for i, cross_attn in enumerate(self.cross_attn_layers):
            normed_v = self.fusion_attn_norms[i](vision_features)
            if return_attention:
                attn_out, attn = cross_attn(normed_v, text_features, return_attention=True,
                                            key_padding_mask=text_mask)
                attention_maps.append(attn)
            else:
                attn_out = cross_attn(normed_v, text_features, key_padding_mask=text_mask)
            vision_features = vision_features + attn_out
            vision_features = vision_features + self.fusion_ffns[i](self.fusion_ffn_norms[i](vision_features))

        pooled = self.attn_pooler(vision_features).squeeze(1)
        query = self.query_projection(pooled)
        ref_embedding = self.visual_projection(ref_embedding)

        out = (query,)
        if return_attention:
            out = out + (attention_maps,)
        if return_ref_embedding:
            out = out + (ref_embedding,)
        return out[0] if len(out) == 1 else out

    @staticmethod
    def _avg_attention(attention_maps):
        return torch.stack(attention_maps, dim=0).mean(dim=0).mean(dim=1)  # (B, V, T)

    @staticmethod
    def _find_subsequence(haystack, needle):
        n, m = len(haystack), len(needle)
        if m == 0 or m > n:
            return None
        for s in range(n - m + 1):
            if haystack[s:s + m] == needle:
                return s, s + m
        return None

    def get_cross_attention_for_tokens(self, vision_input, text_input, token_indices):
        _, attention_maps = self.forward(vision_input, text_input, return_attention=True)
        return self._avg_attention(attention_maps)[:, :, token_indices].mean(dim=-1)

    def compute_text_cc_loss(self, batch, incontext_maps=None):

        images = batch["reference"]
        full_texts = batch["caption"]
        nps_list = batch["noun_phrases"]
        device = images.device
        B = images.shape[0]

        if incontext_maps is None:
            _, incontext_maps = self.forward(images, list(full_texts), return_attention=True)
        incontext = self._avg_attention(incontext_maps)
        T = incontext.shape[2]

        tok = self.text_encoder.tokenizer
        ic_vecs, iso_imgs, iso_texts, iso_lens = [], [], [], []
        for i in range(B):
            nps = nps_list[i]
            if not nps:
                continue
            full_ids = tok(full_texts[i], truncation=True, max_length=T)["input_ids"]
            for np_text in nps:
                if not np_text:
                    continue
                np_ids = tok(np_text, truncation=True, max_length=T)["input_ids"]
                np_core = np_ids[1:-1] if len(np_ids) >= 2 else np_ids
                if not np_core:
                    continue
                span = self._find_subsequence(full_ids, np_core)
                if span is None:
                    continue
                s, e = span[0], min(span[1], T)
                if s >= T or e <= s:
                    continue
                ic_vecs.append(incontext[i, 1:, s:e].mean(dim=-1))  # drop CLS vision token
                iso_imgs.append(images[i])
                iso_texts.append(np_text)
                iso_lens.append(len(np_core))

        if not ic_vecs:
            z = torch.zeros((), device=device)
            return z, z.detach(), 0
        ic_vecs = torch.stack(ic_vecs, dim=0)

        iso_vecs = []
        with torch.no_grad():
            for c in range(0, len(iso_texts), 32):
                imgs_c = torch.stack(iso_imgs[c:c + 32], dim=0)
                _, amaps = self.forward(imgs_c, iso_texts[c:c + 32], return_attention=True)
                attn_c = self._avg_attention(amaps)
                for k, L in enumerate(iso_lens[c:c + 32]):
                    iso_vecs.append(attn_c[k, 1:, 1:1 + L].mean(dim=-1))
        iso_vecs = torch.stack(iso_vecs, dim=0)

        div = F.relu((ic_vecs - iso_vecs).abs() - self.epsilon_cc).mean(dim=-1)
        return div.mean(), div.mean().detach(), ic_vecs.shape[0]

    def infonce(self, features_a, features_b, normalize=True):
        if normalize:
            features_a = F.normalize(features_a, dim=1)
            features_b = F.normalize(features_b, dim=1)
        sim = torch.matmul(features_a, features_b.T) / self.temperature
        labels = torch.arange(features_a.shape[0], device=features_a.device)
        return (F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels)) / 2.0

    def get_image_features(self, image):
        return self.visual_projection(self.vision_encoder(image, get_embeddings=True))

    @staticmethod
    def _gather(t):
        """Differentiable all-gather across DDP ranks -> (world*B, D). No-op off-distributed."""
        import torch.distributed as dist
        if not (dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1):
            return t, 0, 1
        import torch.distributed.nn as dist_nn
        world, rank = dist.get_world_size(), dist.get_rank()
        gathered = list(dist_nn.functional.all_gather(t.contiguous()))
        gathered[rank] = t
        return torch.cat(gathered, dim=0), rank, world

    def cir_infonce(self, vl, target, reference):
        """Symmetric InfoNCE with the reference image as a pure in-denominator hard negative and
        cross-rank gathered negatives. Each query softmaxes over the gathered [targets | references]
        with its own target as the only positive."""
        B = vl.shape[0]
        vl = F.normalize(vl, dim=1)
        target = F.normalize(target, dim=1)
        reference = F.normalize(reference, dim=1)
        g_vl, rank, _ = self._gather(vl)
        g_target, _, _ = self._gather(target)
        g_reference, _, _ = self._gather(reference)
        candidates = torch.cat([g_target, g_reference], dim=0)
        labels = torch.arange(B, device=vl.device) + rank * B
        scale = 1.0 / self.temperature
        loss_q = F.cross_entropy(torch.matmul(vl, candidates.T) * scale, labels)
        loss_t = F.cross_entropy(torch.matmul(target, g_vl.T) * scale, labels)
        return (loss_q + loss_t) / 2.0

    def training_step(self, batch, batch_idx):
        query_image = batch["reference"]
        target_image = batch["target"]
        diff_text = batch["caption"]

        # force train mode so HF gradient checkpointing engages (numerically a no-op for CLIP)
        if not self.vision_encoder.model.training:
            self.vision_encoder.model.train()
            self.text_encoder.model.train()

        if self.lambda_cc > 0:
            vl_emb, incontext_maps, ref_emb = self.forward(
                query_image, diff_text, return_attention=True, return_ref_embedding=True)
        else:
            vl_emb, ref_emb = self.forward(query_image, diff_text, return_ref_embedding=True)
            incontext_maps = None
        target_emb = self.get_image_features(target_image)

        loss = self.cir_infonce(vl_emb, target_emb, ref_emb)
        if self.lambda_cc > 0:
            loss_cc, cc_div, cc_n = self.compute_text_cc_loss(batch, incontext_maps)
            loss = loss + self.lambda_cc * loss_cc
            self.log("train_loss_cc", loss_cc, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log("train_cc_n_concepts", float(cc_n), on_step=True, sync_dist=True)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        vl_emb = self.forward(batch["reference"], batch["caption"])
        target_emb = self.get_image_features(batch["target"])
        val_loss = self.infonce(vl_emb, target_emb)
        self.log("val_loss", val_loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return val_loss

    def configure_optimizers(self):
        common = dict(weight_decay=self.weight_decay, betas=(0.9, 0.98), eps=1e-8)
        if self.backbone_lr_scale != 1.0:
            backbone, new = [], []
            for name, p in self.named_parameters():
                if not p.requires_grad:
                    continue
                if name.startswith(('text_encoder.', 'vision_encoder.', 'visual_projection.')):
                    backbone.append(p)
                else:
                    new.append(p)
            optimizer = torch.optim.AdamW(
                [{'params': backbone, 'lr': self.learning_rate * self.backbone_lr_scale},
                 {'params': new, 'lr': self.learning_rate}],
                lr=self.learning_rate, **common)
        else:
            optimizer = torch.optim.AdamW([p for p in self.parameters() if p.requires_grad],
                                          lr=self.learning_rate, **common)

        if self.steps_per_epoch is None:
            raise ValueError("steps_per_epoch is not set.")

        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1e-8, end_factor=1.0, total_iters=self.warmup_steps)
        if self.lr_schedule == 'cosine':
            t_max = max(1, int(self.steps_per_epoch) - int(self.warmup_steps))
            main = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=0.0)
        else:
            main = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=self.cosine_t0, T_mult=self.T_mult, eta_min=0.0)
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, main], milestones=[self.warmup_steps])

        return {'optimizer': optimizer,
                'lr_scheduler': {'scheduler': scheduler, 'interval': 'step', 'frequency': 1,
                                 'name': 'lr_scheduler'}}
