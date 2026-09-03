"""
DETERMINISTIC VQA MODEL (No Latent Reasoning)
==============================================

Pure cross-attention fusion without VAE/KL regularization.
Focus on accuracy and stability for low-resource Vietnamese VQA.

Key differences from model.py:
- NO CompressedLatentReasoning module
- NO KL divergence loss
- NO free bits, no VAE sampling
- Direct cross-attention: decoder → (vision + text) features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple
import re

from transformers import (
    AutoModel,
    AutoImageProcessor,
    AutoTokenizer,
    BartphoTokenizer,
    MBartForConditionalGeneration
)


def shift_tokens_right(input_ids, pad_token_id, decoder_start_token_id):
    """Shift tokens right for teacher forcing"""
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)
    return shifted_input_ids


# ============================================================================
# ============================================================================
# FLAMINGO-STYLE GATED CROSS ATTENTION
# ============================================================================
# Note: Manual LoRALayer removed - PEFT library handles all LoRA functionality

class FlamingoGatedCrossAttention(nn.Module):
    """
    Flamingo-style Gated Cross Attention with configurable fusion direction
    
    Fusion types:
        - 'text2vision': Vision attends to text (original Flamingo)
        - 'vision2text': Text attends to vision (inverse)
        - 'bidirectional': Both directions with separate gates
    """
    def __init__(self, hidden_dim=1024, num_heads=16, dropout=0.1, fusion_type='text2vision'):
        super().__init__()
        
        self.fusion_type = fusion_type
        
        # Primary cross-attention (text2vision or vision2text)
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        
        # Bidirectional: add reverse cross-attention
        if fusion_type == 'bidirectional':
            self.cross_attn_reverse = nn.MultiheadAttention(
                hidden_dim, num_heads, dropout=dropout, batch_first=True
            )
        
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
        
        self.norm_cross = nn.LayerNorm(hidden_dim)
        self.norm_ffn = nn.LayerNorm(hidden_dim)
        
        # Init gates to 0.5 instead of 0.0 so tanh(0.5)≈0.46 ≠ 0 at epoch 0.
        # With zero init, the entire cross-attention contribution is dead at start,
        # forcing the model to find a text-only solution before fusion activates.
        self.gate_cross = nn.Parameter(torch.full((1,), 0.5))
        self.gate_ffn = nn.Parameter(torch.full((1,), 0.5))
        
        # Bidirectional: add reverse gates
        if fusion_type == 'bidirectional':
            self.norm_cross_reverse = nn.LayerNorm(hidden_dim)
            self.norm_ffn_reverse = nn.LayerNorm(hidden_dim)
            self.gate_cross_reverse = nn.Parameter(torch.full((1,), 0.5))
            self.gate_ffn_reverse = nn.Parameter(torch.full((1,), 0.5))
            
            # Separate FFN for text in bidirectional mode
            self.ffn_reverse = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 4, hidden_dim),
                nn.Dropout(dropout)
            )
        
    def forward(self, vision_features, text_features, text_attention_mask=None,
                residual_scale=None, extra_kv=None, extra_kv_mask=None):
        """residual_scale: [B, P, 1] — he so theo TUNG PATCH cho residual cross-attn.
        Mac dinh None = hanh vi goc (chi co scalar tanh(gate_cross) dung chung).

        Y nghia: gate_cross la MOT scalar toan cuc nen luong thong tin cau hoi nhet vao
        moi patch chi duoc dieu chinh boi trong so attention. Khi truyen residual_scale
        theo tung patch va theo LOAI cau hoi, TCVG dieu khien 'nhet bao nhieu vao patch nao'
        — tuc doi NOI DUNG cua patch, khong phai trong so cua no. Decoder attention chi
        danh trong so tren token nhan duoc nen KHONG THE tai tao thay doi noi dung nay.
        """
        """
        Args:
            vision_features: [B, num_patches, D]
            text_features: [B, seq_len, D]
            text_attention_mask: [B, seq_len]
        
        Returns:
            vision_features, text_features (both updated if bidirectional)
        """
        # Prepare padding masks
        text_key_padding_mask = None
        if text_attention_mask is not None:
            text_key_padding_mask = (text_attention_mask == 0)
        
        if self.fusion_type == 'text2vision':
            # Vision attends to text (original Flamingo)
            # 🔬 extra_kv: NOI THEM token box vao tap key/value. Moi patch attend qua CA token text
            #   VA token box -> thong tin cau truc vat the vao TRUOC gate, nen alpha duoc tinh tren
            #   feature da biet-vat-the. Model tu hoc box nao quan trong voi patch nao, khong phai
            #   minh thiet ke tay box_feat. Output van [B,P,D] nen TCVG phia sau khong doi gi, va
            #   KHONG vi pham rang buoc "TCVG phai sau Flamingo".
            _k = text_features
            _kpm = text_key_padding_mask
            if extra_kv is not None:
                _k = torch.cat([text_features, extra_kv], dim=1)
                if _kpm is None:
                    _kpm = torch.zeros(text_features.size(0), text_features.size(1),
                                       dtype=torch.bool, device=text_features.device)
                _em = (extra_kv_mask == 0) if extra_kv_mask is not None else torch.zeros(
                    extra_kv.size(0), extra_kv.size(1), dtype=torch.bool, device=extra_kv.device)
                _kpm = torch.cat([_kpm, _em], dim=1)
            attn_out, attn_weights = self.cross_attn(
                query=vision_features,
                key=_k,
                value=_k,
                key_padding_mask=_kpm
            )
            if extra_kv is not None:
                # cat bo phan trong so cua token box de last_attn_weights giu dung nghia [B,P,L]
                attn_weights = attn_weights[:, :, :text_features.size(1)]
            # 🔬 HYPOTHESIS #1 (2026-08-10): luu lai trong so attention [B,P,L] (query=patch,
            # key=text token) — nn.MultiheadAttention da tinh SAN cho fusion, khong ton them
            # chi phi. TCVG co the tai su dung lam alpha thay vi hoc gate rieng (VisionGating
            # alpha_from_gca=True). Mac dinh khong dung toi -> khong doi hanh vi hien tai.
            self.last_attn_weights = attn_weights.detach()  # [B, P, L]

            _rs = 1.0 if residual_scale is None else residual_scale
            # 🔬 gca_strength: nhan CA residual de LAM YEU GCA co chu dich (thi nghiem: GCA yeu di
            # thi TCVG co keo lai perf khong -> do 'nang luc bi che' cua TCVG). =1.0 la GCA day du,
            # =0.0 la GCA khong dong gop gi (patch giu nguyen SigLIP, khong thay cau hoi qua GCA).
            _gs = getattr(self, 'gca_strength', 1.0)
            vision_features = vision_features + _gs * _rs * torch.tanh(self.gate_cross) * self.norm_cross(attn_out)
            
            ffn_out = self.ffn(vision_features)
            vision_features = vision_features + _gs * torch.tanh(self.gate_ffn) * self.norm_ffn(ffn_out)
            
            return vision_features, text_features
        
        elif self.fusion_type == 'vision2text':
            # Text attends to vision (inverse)
            attn_out, _ = self.cross_attn(
                query=text_features,
                key=vision_features,
                value=vision_features,
                key_padding_mask=None  # Vision has no padding
            )
            
            text_features = text_features + torch.tanh(self.gate_cross) * self.norm_cross(attn_out)
            
            ffn_out = self.ffn(text_features)
            text_features = text_features + torch.tanh(self.gate_ffn) * self.norm_ffn(ffn_out)
            
            return vision_features, text_features
        
        elif self.fusion_type == 'bidirectional':
            # Both directions — use ORIGINAL features as keys/values so the two
            # cross-attentions are truly parallel (no circular dependency).
            orig_vision = vision_features  # save before any update
            orig_text   = text_features    # save before any update

            # 1. Vision attends to ORIGINAL text
            attn_v2t, _ = self.cross_attn(
                query=vision_features,
                key=orig_text,
                value=orig_text,
                key_padding_mask=text_key_padding_mask
            )
            vision_updated = vision_features + torch.tanh(self.gate_cross) * self.norm_cross(attn_v2t)

            # 2. Text attends to ORIGINAL vision (not the one already enriched in step 1)
            attn_t2v, _ = self.cross_attn_reverse(
                query=text_features,
                key=orig_vision,
                value=orig_vision,
                key_padding_mask=None
            )
            text_updated = text_features + torch.tanh(self.gate_cross_reverse) * self.norm_cross_reverse(attn_t2v)

            # 3. FFN for vision
            ffn_out_v = self.ffn(vision_updated)
            vision_features = vision_updated + torch.tanh(self.gate_ffn) * self.norm_ffn(ffn_out_v)

            # 4. FFN for text
            ffn_out_t = self.ffn_reverse(text_updated)
            text_features = text_updated + torch.tanh(self.gate_ffn_reverse) * self.norm_ffn_reverse(ffn_out_t)

            return vision_features, text_features
        
        else:
            raise ValueError(f"Unknown fusion_type: {self.fusion_type}")


# ============================================================================
# TYPE PREDICTION HEAD (Auxiliary Task for Multi-task Learning)
# ============================================================================

class TypePredictionHead(nn.Module):
    """
    Auxiliary head for question type classification
    
    Types:
        0 = OBJECT (Đây là gì? Cái gì?)
        1 = COUNT (Có bao nhiêu? Mấy cái?)
        2 = COLOR (Màu gì?)
        3 = LOCATION (Ở đâu? Phía nào? Trên/dưới?)
    
    Purpose: Force question encoder to learn type-level patterns
    NOT used for hard decision - just auxiliary signal!
    """
    def __init__(self, hidden_dim=1024, num_types=4, dropout=0.1):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_types)
        )
    
    def forward(self, text_cls):
        """
        Args:
            text_cls: [B, D] - CLS token from question encoder
        Returns:
            type_logits: [B, 4] - logits over 4 question types
        """
        return self.classifier(text_cls)


# ============================================================================
# VISION GATING (Learnable Attention-Based)
# ============================================================================

class TypeCodebook(nn.Module):
    """Phat hien loai cau hoi KHONG GIAM SAT — thay TypePredictionHead va nhan loai.

    Dung dung de xuat Future Work cua paper (§5, tham chieu [23] van den Oord et al.):
    luong tu hoa t_cls ve prototype gan nhat trong mot codebook, prototype do dong vai tro
    e_type. Bo hoan toan nhu cau co nhan loai -> khong con phai preprocessing type cho tung
    bo du lieu (ViVQA, ViVQA-X, ...), va bo luon bug _INT_TO_TYPE gan nhan sai giua hai bo.

    Vi sao rui ro thap o day (do duoc, khong phai phong doan):
      - type head giam sat dat 99.64 +- 0.12 -> loai cau hoi gan nhu xac dinh hoan toan boi
        cu phap cau hoi tieng Viet ("mau gi", "bao nhieu"), tuc rat de phan cum.
      - oracle gap (thay bang nhan that) chi +0.07 -> chat luong tin hieu loai KHONG phai
        nut that, nen thay bang tin hieu tu phat hien it rui ro.

    Cai dat:
      - luong tu hoa theo L2 toi prototype gan nhat, straight-through estimator
      - loss VQ = ||sg[z] - e||^2 + beta * ||z - sg[e]||^2  (chuan VQ-VAE)
      - HOI SINH MA CHET: che do sup do (moi mau ve cung mot prototype) la kieu that bai
        chinh cua VQ va no se lang le bien mo hinh thanh 'khong co loai'. Prototype khong
        duoc dung qua `dead_after` buoc se duoc khoi tao lai bang mot z ngau nhien trong batch.
      - bao cao perplexity de theo doi so ma thuc su duoc dung.
    """

    def __init__(self, dim, num_codes=4, beta=0.25, decay=0.99, dead_after=200):
        super().__init__()
        self.num_codes = num_codes
        self.beta = beta
        self.decay = decay
        self.dead_after = dead_after
        self.codebook = nn.Parameter(torch.randn(num_codes, dim) * 0.02)
        # buffer -> khong phai tham so hoc, nhung van duoc luu/khoi phuc theo checkpoint
        self.register_buffer('usage', torch.zeros(num_codes))
        self.register_buffer('idle', torch.zeros(num_codes))

    @torch.no_grad()
    def _revive(self, z):
        """Khoi tao lai cac prototype khong duoc dung lau -> chong sup do codebook."""
        dead = self.idle > self.dead_after
        if not bool(dead.any()):
            return
        idx = torch.randint(0, z.size(0), (int(dead.sum()),), device=z.device)
        self.codebook.data[dead] = z[idx].detach().to(self.codebook.dtype)
        self.idle[dead] = 0.0

    def forward(self, z):
        """z: [B, D] (t_cls)  ->  (e_st [B, D], idx [B], vq_loss, perplexity)"""
        zf = z.float()
        cb = self.codebook.float()
        # khoang cach L2 binh phuong ||z||^2 - 2 z.e + ||e||^2
        d = (zf.pow(2).sum(1, keepdim=True)
             - 2 * zf @ cb.t()
             + cb.pow(2).sum(1).unsqueeze(0))          # [B, K]
        idx = d.argmin(dim=1)                           # [B]
        e = cb[idx]                                     # [B, D]

        vq_loss = F.mse_loss(e, zf.detach()) + self.beta * F.mse_loss(zf, e.detach())
        e_st = zf + (e - zf).detach()                   # straight-through
        e_st = e_st.to(z.dtype)

        with torch.no_grad():
            counts = F.one_hot(idx, self.num_codes).float().sum(0)
            probs = counts / counts.sum().clamp(min=1)
            perplexity = torch.exp(-(probs * (probs + 1e-10).log()).sum())
            # CHI cap nhat trang thai khi TRAIN. Neu cap nhat ca luc eval thi eval khong con
            # idempotent va bo dem idle tang trong eval se kich hoat hoi sinh ma sai luc quay lai train.
            if self.training:
                self.usage.mul_(self.decay).add_(counts, alpha=1 - self.decay)
                self.idle = torch.where(counts > 0, torch.zeros_like(self.idle), self.idle + 1)
                self._revive(zf)

        return e_st, idx, vq_loss, perplexity


class TypeSlotAttention(nn.Module):
    """🔬 TIEN HOA TCVG: slot attention CANH TRANH co dieu kien loai.
    K slot gom patch thanh instance (softmax OVER SLOTS = patch cạnh tranh về slot).
    Khac summary_token (1 readout, trung voi decoder): GOM instance la thu decoder-1-luot
    KHONG lam duoc. K slot THEM vao decoder (patch nguyen ven -> hai none ve cau truc).
    Slot init co dieu kien q_type = W[t_cls; e_type] -> loai anh huong cach gom.
    Aggregation chiu patch tho tot hon selection (top-k)."""
    def __init__(self, dim, num_slots=4, num_iters=3, num_types=4,
                 init_std=0.02, tanh_gate=False):
        super().__init__()
        self.num_slots = num_slots
        self.num_iters = num_iters
        self.type_emb = nn.Embedding(num_types, dim)
        nn.init.normal_(self.type_emb.weight, std=0.02)
        # 🔬 PHA VO DOI XUNG: voi init_std=0.02 (ban cu), slot_init bi `+ c` ap dao hoan toan ->
        # do duoc: cosine giua cac cap slot = 0.997, std giua slot = 0.0506, tuc K slot la K BAN SAO.
        # Ma slot attention hoat dong BANG pha vo doi xung: cac slot phai khac nhau tu dau moi canh
        # tranh chiem cac vung khac nhau. init_std lon hon -> slot phan hoa that.
        self.slot_init = nn.Parameter(torch.randn(1, num_slots, dim) * init_std)
        # 🔬 KHONG-GAY-HAI TAI INIT (quy uoc cua repo nay: type_experts model.py:1689, l6_fuse:740,
        # blend_head:731 deu zero-init de = identity luc init). TypeSlotAttention thieu han dieu do,
        # va out_norm=LayerNorm CUONG BUC ||slot|| = sqrt(D) = 32.0 -- do duoc BANG NHAU voi token
        # patch (32.00 vs 32.00). Nen K token nhieu canh tranh BINH DANG voi patch that ngay tu buoc
        # 0, hut khoi luong attention cua decoder. Dau hieu khop: LOCATION tut manh nhat (-1.89) o
        # ca hai arm -- loai can bo cuc khong gian trai tren nhieu patch, nhay nhat voi pha loang.
        # tanh_gate: nhan tanh(g) voi g init 0 -> slot ~ 0 luc init (thu thuat cong cua Flamingo,
        # repo da dung trong FlamingoGatedCrossAttention), model tu mo dan neu slot co ich.
        self.cond = nn.Linear(dim * 2, dim)
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        self.gru = nn.GRUCell(dim, dim)
        self.norm_slots = nn.LayerNorm(dim)
        self.norm_in = nn.LayerNorm(dim)
        self.norm_ff = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.out_norm = nn.LayerNorm(dim)
        self.scale = dim ** -0.5
        self.tanh_gate = bool(tanh_gate)
        self.slot_g = nn.Parameter(torch.zeros(1)) if tanh_gate else None

    def forward(self, patches, text_cls, type_ids):
        B, P, D = patches.shape
        if getattr(self, 'no_type', False):
            # 🔬 ABLATION: bo tin hieu LOAI khoi slot init. Neu slot van tot bang -> gain KHONG den
            # tu dieu kien hoa theo loai, va KHONG duoc trinh bay no nhu TCVG mo rong.
            e = torch.zeros(B, D, device=patches.device, dtype=patches.dtype)
        elif type_ids is not None:
            e = self.type_emb(type_ids)
        else:
            e = self.type_emb.weight.mean(0, keepdim=True).expand(B, -1)
        c = self.cond(torch.cat([text_cls, e], dim=-1)).unsqueeze(1)          # [B,1,D]
        slots = self.slot_init.expand(B, -1, -1) + c                          # [B,K,D] dieu kien loai
        pin = self.norm_in(patches)
        k = self.to_k(pin); v = self.to_v(pin)                               # [B,P,D]
        for _ in range(self.num_iters):
            prev = slots
            q = self.to_q(self.norm_slots(slots))                            # [B,K,D]
            logits = torch.einsum('bkd,bpd->bkp', q, k) * self.scale         # [B,K,P]
            attn = logits.softmax(dim=1) + 1e-8                              # softmax OVER SLOTS (canh tranh)
            attn = attn / attn.sum(dim=-1, keepdim=True)                     # chuan hoa theo P de tinh mean
            updates = torch.einsum('bkp,bpd->bkd', attn, v)                  # [B,K,D]
            # 🔬 GIU ma tran GAN de co the giam sat binding (--slot_bind_lambda). Luu logits.softmax
            # theo SLOT (dim=1) chu KHONG phai attn da chuan hoa theo P o tren: cai can giam sat la
            # "patch nay thuoc slot nao" (phan bo tren K, tong=1), khong phai "slot nay lay bao nhieu
            # tu patch nao". Hai thu khac nhau; dung sai cai la giam sat nguoc chieu.
            self.last_assign = logits.softmax(dim=1)                          # [B,K,P] tong theo K = 1
            slots = self.gru(updates.reshape(-1, D), prev.reshape(-1, D)).reshape(B, self.num_slots, D)
            slots = slots + self.mlp(self.norm_ff(slots))
        out = self.out_norm(slots)                                           # [B,K,D]
        if self.slot_g is not None:
            out = torch.tanh(self.slot_g) * out      # init 0 -> slot vo hinh, KHONG gay hai
        return out


class SummaryToken(nn.Module):
    """🔬 Token tom tat co dieu kien LOAI, THEM vao chuoi encoder — PATCH GIU NGUYEN.
    summary = LN(out_proj( Attn(q_type, patches) )),  q_type = q_proj([t_cls; e_type]).
    'Hai none' theo CAU TRUC: patch khong bi dung, decoder co the gan attention ~0 cho token
    -> nghiem T0 (bo qua token) NAM TRONG khong gian tim kiem -> optimum >= T0.
    Khac gate (tru/sua patch -> co the pha): token nay chi CONG mot lua chon."""
    def __init__(self, dim, num_types=4):
        super().__init__()
        self.type_emb = nn.Embedding(num_types, dim)
        nn.init.normal_(self.type_emb.weight, std=0.02)
        self.q_proj = nn.Linear(dim * 2, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, patches, text_cls, type_ids):
        # patches [B,P,D], text_cls [B,D], type_ids [B] hoac None
        if type_ids is not None:
            e = self.type_emb(type_ids)                       # [B,D]
        else:
            e = self.type_emb.weight.mean(0, keepdim=True).expand(text_cls.size(0), -1)
        q = self.q_proj(torch.cat([text_cls, e], dim=-1)).unsqueeze(1)   # [B,1,D]
        k = self.k_proj(patches)                              # [B,P,D]
        v = self.v_proj(patches)                              # [B,P,D]
        attn = torch.softmax((q @ k.transpose(1, 2)) / (patches.size(-1) ** 0.5), dim=-1)  # [B,1,P]
        summary = attn @ v                                    # [B,1,D]
        return self.norm(self.out_proj(summary))              # [B,1,D]


class VisionGating(nn.Module):
    """
    🔥 TYPE-CONDITIONED Vision Gating with Attention
    
    Key idea: (question + type) → gate → select vision
    
    Different question types need different vision features:
        - COLOR → attend to color-rich patches
        - COUNT → attend to object distribution globally  
        - LOCATION → attend to spatial arrangement
        - OBJECT → attend to salient regions
    
    Implementation:
        1. Embed question type as learnable vector
        2. Combine (question_cls + type_emb) as "query"
        3. Attention query @ vision → importance scores α
        4. Gated vision = α * vision + (1-α) * text_context
    """
    def __init__(self, hidden_dim=1024, num_types=4, init_bias=1.0, min_alpha=0.0,
                 max_alpha=1.0, min_alpha_pertype=None, max_alpha_pertype=None, use_delta_gate=False, type_emb_std=None,
                 type_emb_init=None,
                 gate_mode='blend', norm_type_emb=False, type_null=False,
                 type_bias=False, type_ctx=False, ln_mode='post', attn_gate=False,
                 refine_gate=False, proto_gate=False,
                 gate_layerscale_pertype=False, gate_layerscale_init=1.0,
                 gate_blend_learned=False, gate_no_type_emb=False, gate_no_text_cls=False,
                 gate_blend_vorig=False,
                 gate_alpha_budget=False, gate_budget_init=0.72,
                 gate_pertype_net=False,
                 gate_per_channel=False, gate_gca_residual=False,
                 gate_blend_l6=False, gate_l6_fuse=False,
                 gate_l6_fuse_bottleneck=256, global_scalar_gate=False,
                 spatial_blend=False, dynamic_peek=False, alpha_from_gca=False,
                 box_content=False, box_max_inst=32, box_class_vocab=0,
                 spatial_pertype=None):
        super().__init__()
        # 🔬 box_content: nap NOI DUNG tu box COCO vao dung hai cho ma TCVG dang rong.
        #
        #   Van de do duoc: blend_target hien tai la `text_pooled` — MOT vector dung chung cho ca
        #   197 patch, nen (1-alpha) khong chen vao noi dung nao moi. Va spatial_blend/gate_blend_l6
        #   da thay text_pooled bang pool cua v_proj / L6 -> van la HAM CUA v_proj, tuc trong span
        #   cua nhung gi decoder da doc duoc -> ca hai net zero. Slot cung vay: slot = f(patch)
        #   nen I(dap an; slot | patch) = 0 theo dinh ly.
        #
        #   Box thi khac: nhan nguoi gan, model KHONG tinh ra duoc tu patch SigLIP dong bang
        #   (neu tinh duoc thi viec dem da xong). Do duoc: so box cua category khop dap an gold
        #   76.6% so voi null dao nhan 32.9% +- 0.8 -> tin hieu thuc +43.7pp.
        #
        #   Dac trung dung o day BAT BIEN voi thu tu id (id ca the la tuy y moi anh):
        #     is_obj_i     : patch i co thuoc mot ca the nao khong
        #     reg_size_i   : ti le patch cung vung voi i  (kich thuoc vat the chua i)
        #     n_reg        : so ca the phan biet trong anh / box_max_inst  <- tin hieu DEM truc tiep
        #   Hai diem tiem, ca hai zero-init nen = TCVG goc tai init (quy uoc non-harm cua repo):
        #     (1) blend_target = text_pooled + tanh(box_g) * box_fuse([v_proj ; box_feat])
        #     (2) alpha_logit += tanh(box_ag) * box_alpha_head(box_feat)
        #   (2) cho alpha biet "patch nay co phai vat the khong" — dung thu ma oracle-alpha biet
        #   ma gate khong, va thang oracle noi headroom nam o per-patch (+11.27) chu khong per-type (0).
        if spatial_pertype is not None:
            self.register_buffer('spatial_pertype', torch.tensor(spatial_pertype, dtype=torch.float32))
        else:
            self.spatial_pertype = None
        self.box_content = bool(box_content)
        self.box_max_inst = int(box_max_inst)
        if self.box_content:
            self.box_proj = nn.Sequential(nn.Linear(3, 128), nn.GELU(), nn.Linear(128, hidden_dim))
            # 🔬 box_class_emb: nhung ID LOP COCO cua patch (81 lop, 0 = nen). Do duoc: 6.3 bit/patch,
            #   so voi box_proj cu chi chua (is_obj, region_size, n_reg) — ma is_obj/region_size da
            #   giai ma duoc tuyen tinh tu patch dong bang o AUC 0.8706 (gan nhu khong co gi moi) va
            #   n_reg la MOT so dung chung ca 197 patch. Dung khi region_map mang GIA TRI LOP
            #   (patch_class_map_flat.pkl) thay vi id ca the.
            self.box_class_emb = nn.Embedding(box_class_vocab, hidden_dim) if box_class_vocab else None
            if self.box_class_emb is not None:
                nn.init.normal_(self.box_class_emb.weight, std=0.02)
            self.box_fuse = nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim), nn.GELU(),
                                          nn.Linear(hidden_dim, hidden_dim))
            nn.init.zeros_(self.box_fuse[-1].weight); nn.init.zeros_(self.box_fuse[-1].bias)
            self.box_g = nn.Parameter(torch.zeros(1))
            self.box_alpha_head = nn.Linear(hidden_dim, 1, bias=False)
            nn.init.zeros_(self.box_alpha_head.weight)
            self.box_ag = nn.Parameter(torch.zeros(1))
        # 🔬 global_scalar_gate: kiem dinh gia thuyet "TCVG = shrinkage estimator (co ve
        # text_pooled voi CUONG DO trung binh), khong phai spatial selector". alpha_pre luon = 0
        # (bo qua gate_net, khong phu thuoc v_i hay q) -> alpha = sigma(vision_bias) — MOT so
        # hoc duoc, giong het moi patch/moi cau hoi/moi anh. Neu bien the nay dat gan T2
        # (instance+type-conditioned) thi toan bo tinh chon loc theo patch khong dong gop gi,
        # gia tri that nam o "co bao nhieu", khop voi ket qua flatten_alpha (flat approx normal
        # o 2/3 seed da do). gate_net van duoc tao (khong dung toi) de state_dict tuong thich
        # neu sau nay muon bat lai gate_net.
        self.global_scalar_gate = bool(global_scalar_gate)
        # 🔬 gate_blend_l6: blend_target = proj(L6) (spatial content) thay vi text_pooled (rong).
        #   gated = α·v_L12 + (1−α)·proj(L6). α VAN tinh tu output L12 (semantic, dung bai hoc glL6);
        #   patch bi suppress roi ve BAN L6 SPATIAL cua chinh no -> "nen" lan dau THEM info (dinh vi
        #   ma L12 mat) thay vi tru ve vector chung. Thoat oracle-β=0 (do voi blend rong).
        self.gate_blend_l6 = bool(gate_blend_l6)
        # 🔬 gate_l6_fuse (fix L6-blend that bai: L6 tho decoder khong doc duoc — sai khong gian):
        #   blend_target = text_pooled + l6_fuse([v_L12 ; proj(L6)]).
        #   l6_fuse HOC bien L6 spatial sang KHONG GIAN SEMANTIC (nhan ca v_L12) roi CONG vao.
        #   zero-init lop cuoi -> khoi dau blend_target = text_pooled = gate chuan (non-harm),
        #   roi hoc them L6-spatial trong khong gian decoder dung duoc.
        self.gate_l6_fuse = bool(gate_l6_fuse)
        # 🔬 gate_per_channel: alpha [B,P,D] (moi PATCH moi CHANNEL) thay vi [B,P] (mo ca vector patch
        #   dong deu). Cho phep "voi COLOR giu kenh mau cua patch, bo kenh texture" — bieu dien manh
        #   hon han per-patch scalar. Lop cuoi gate_net xuat D thay vi 1; zero-init weight -> luc dau
        #   moi channel = sigma(bias) DONG DEU (= gate per-patch chuan tai init, non-harm), roi tach dan.
        self.gate_per_channel = bool(gate_per_channel)
        # 🔬 gate_gca_residual: TCVG DIEU KHIEN GCA thay vi lam lai viec cua GCA.
        #   Do duoc (gca_sweep, SigLIP1 seed42): gca=1.0 -> T0 72.48 | gca=0.5 -> 72.11 | gca=0.0 -> 67.78.
        #   Giua 0.5 va 1.0 gan nhu PHANG -> loi ich cua GCA KHONG DONG NHAT giua cac patch:
        #   mot so patch can tiem text, mot so co le dang bi hai. Mot he so TOAN CUC khong bat duoc
        #   dieu do; mot he so THEO TUNG PATCH thi co.
        #     gated = v_proj − γ·(1−α)·(v_proj − v_orig)
        #   γ init 0  -> gated = v_proj DUNG BANG T0 (non-harm tuyet doi tai init)
        #   γ = 1     -> gated = α·v_proj + (1−α)·v_orig, tuc noi suy tung patch giua
        #                "GCA day du" va "khong GCA"
        #   Khac han blend chuan (tron ve text_pooled) va khac use_delta_gate (chi dua delta vao
        #   DAU VAO cua gate net, khong doi CAI GI bi gate).
        self.gate_gca_residual = bool(gate_gca_residual)
        if self.gate_gca_residual:
            # SUA v2 (sau khi v1 sap 26.42): v1 THAY THE phep tron goc nen TCVG mat viec cu va
            # phai tai su dung alpha cho viec moi -> alpha DAO NGUOC (OBJECT 0.9989 -> 0.0008,
            # LOCATION 0.9995 -> 0.0000) va hai loai do sap con 1-3%. Hai vai tro tranh nhau
            # CUNG MOT tham so. v2 cho viec moi mot dau RIENG:
            #     v_ctrl = v_orig + beta_i * (v_proj - v_orig)     beta: dieu khien GCA tung patch
            #     gated  = alpha * v_ctrl + (1-alpha) * text_pooled  <- TCVG goc, GIU NGUYEN
            # beta init 1 -> v_ctrl = v_proj -> DUNG BANG TCVG da cong bo (non-harm tuyet doi).
            self.gcares_proj = nn.Linear(hidden_dim, hidden_dim)
            self.gcares_head = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim // 4), nn.GELU(),
                nn.Linear(hidden_dim // 4, 1))
            nn.init.zeros_(self.gcares_head[-1].weight)
            nn.init.zeros_(self.gcares_head[-1].bias)
            # SUA v3: v2 dung sigmoid(bias=4) NHAN voi gamma init 0 -> HAI cua chan noi tiep,
            # gradient toi gcares_head bang DUNG 0 (do duoc), head khong bao gio hoc duoc.
            # Day dung la bay 'blend_gamma sigmoid(-6)' da mac truoc do. Dang residual tanh:
            #     beta = 1 + tanh(head)      head zero-init -> tanh(0)=0 -> beta = 1 CHINH XAC
            # non-harm tuyet doi, MA tanh'(0)=1 nen gradient day du ngay tu buoc dau.
            # beta in (0,2): co the GIAM (bo bot GCA) hoac TANG (khuech dai GCA) tung patch.

        # 🔬 HYPOTHESIS #1 (2026-08-10): alpha lay TRUC TIEP tu trong so attention cua GCA
        # (khong hoc gate_net/type_embedding/query_proj rieng cho alpha). Xem chi tiet o forward().
        self.alpha_from_gca = bool(alpha_from_gca)

        self.gate_mode = gate_mode   # 'blend' (goc) | 'multiply' (suppress thuc su)
        # 🔬 ln_mode: dat LayerNorm o dau so voi phep tron.
        #   'post' (goc)  gated = LN(a*v + (1-a)*t)   -> LN XOA thanh phan bien do cua gate
        #   'pre'         gated = a*LN(v) + (1-a)*LN(t)  -> giu bien do; token alpha thap NGAN hon
        #   'none'        gated = a*v + (1-a)*t          -> dung phuong trinh §3.3 cua paper
        #
        # Do duoc (measure_ln_erasure.py, checkpoint run87): o alpha=0.45 gate nen token xuong
        # 46.4% do dai, roi LN keo ve 99.99%. Chi phan doi HUONG di qua duoc; 57-79% tac dung
        # bi nuot o vung alpha thuc te. Ma do dai token CHINH LA kenh de cross-attention cua
        # decoder biet "bo qua patch nay" (trong so = tich vo huong). Nen LN-post cat dung kenh
        # ma gate dung de noi chuyen voi decoder.
        # Chinh codebase da ghi nhan dieu nay cho gate_mode='multiply' (phai LN TRUOC roi moi
        # nhan alpha, neu khong LN xoa sach alpha). 'blend' thi van de LN o SAU.
        # KHONG them module moi -> dung lai self.layer_norm -> khong dich RNG -> van ghep cap
        # duoc voi baseline.
        self.ln_mode = ln_mode
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha
        # 🔬 per-type alpha floor: [num_types] — bao ve loai can distributed-view (COUNT) hoac
        # da bao hoa (OBJECT) bang cach nang floor -> gate gan-inert o loai do (soft-interp thuan,
        # trong objective goc). Index: 0=OBJECT 1=COUNT 2=COLOR 3=LOCATION.
        if min_alpha_pertype is not None:
            self.register_buffer('min_alpha_pertype',
                                 torch.tensor(min_alpha_pertype, dtype=torch.float32))
        else:
            self.min_alpha_pertype = None
        # 🔬 per-type alpha CEILING (2026-08-09): alphaclamp (max_alpha=0.9 CHUNG cho moi type)
        # do duoc hai COUNT nhat quan (-2.14pp ca 3 seed) vi COUNT can alpha GAN 1 (giu "distributed
        # view", dung ly thuyet paper §3.3), trong khi tran chung lai giup LOCATION/OBJECT (tranh
        # collapse ve cuc tri co hai, da do qua topk/bottomk probe). Sua: tran RIENG tung type —
        # COUNT duoc mien tran (hoac tran rat cao, gan 1.0), cac type con lai van bi kep. Cung
        # index [OBJECT,COUNT,COLOR,LOCATION] nhu min_alpha_pertype.
        if max_alpha_pertype is not None:
            self.register_buffer('max_alpha_pertype',
                                 torch.tensor(max_alpha_pertype, dtype=torch.float32))
        else:
            self.max_alpha_pertype = None
        self.use_delta_gate = use_delta_gate

        # Type embeddings (learnable per-type representations)
        self.type_embedding = nn.Embedding(num_types, hidden_dim)
        # 🔥 nn.Embedding mac dinh init N(0,1) -> voi hidden_dim=1024 thi norm ~32, rat lon.
        # Bon type embedding vi the khac nhau manh ngay tu buoc 0, nen "loai nao bi gating"
        # duoc gan NGAU NHIEN theo seed (do duoc: khuon mau gating doi theo seed, 11 seed).
        # Ha std xuong (vd 0.02, muc chuan cho embedding transformer) thi ca 4 loai bat dau
        # gan nhu giong nhau -> gate khoi dau type-agnostic, phan biet theo loai chi xuat hien
        # neu du lieu thuc su doi hoi.
        if type_emb_std is not None:
            nn.init.normal_(self.type_embedding.weight, mean=0.0, std=float(type_emb_std))
        # 🔬 2026-08-09: khoi tao tu prototype ngu nghia THAT (mean-pool embedding cua cau hoi
        # that moi type, tu BARTpho dong bang, tinh truoc bang compute_type_prototypes.py) thay
        # vi random hoac tu random-nho. Sua dung nguyen nhan goc cua symmetry-breaking: gate bat
        # dau tu HUONG CO Y NGHIA (tach biet that, do duoc cosine sim 0.70-0.91 giua cac type)
        # thay vi huong ngau nhien (N(0,1)) hay gan-nhu-khong-huong (std nho) -> gradient chi can
        # TINH CHINH, khong phai TU TIM huong tu dau. Tong quat: KHONG hardcode gia tri, tu tinh
        # tu du lieu that -> ap dung duoc voi taxonomy/dataset khac (chi can cot 'type').
        if type_emb_init is not None:
            with torch.no_grad():
                _init_t = type_emb_init if torch.is_tensor(type_emb_init) else torch.tensor(type_emb_init)
                assert _init_t.shape == self.type_embedding.weight.shape, \
                    f"type_emb_init shape {_init_t.shape} != type_embedding.weight shape {self.type_embedding.weight.shape}"
                self.type_embedding.weight.copy_(_init_t.to(self.type_embedding.weight.device))

        # 🔥 B: chuan hoa type_embedding. nn.Embedding init N(0,1) cho norm ~32 (hidden=1024),
        # trong khi buoc AdamW chi ~lr moi toa do -> do duoc: vector chi xoay 0.3-0.6 do sau
        # 21320 buoc, tuc ma loai DONG BANG o gia tri ngau nhien. Chuan hoa tach HUONG (tren
        # mat cau don vi, gradient tac dong hieu qua) khoi BIEN DO (mot scalar hoc duoc).
        self.norm_type_emb = norm_type_emb
        if norm_type_emb:
            self.type_scale = nn.Parameter(torch.tensor(float(hidden_dim) ** 0.5))

        # 🔥 A: dich tron rieng theo loai. Mac dinh (1-a)*text_pooled dung MOT vector chung
        # cho moi loai, nen "nen mot patch" khong mang thong tin gi rieng cua loai cau hoi.
        # Them offset hoc duoc theo loai, init = 0 nen luc bat dau GIONG HET baseline.
        self.use_type_null = type_null
        if type_null:
            self.type_null = nn.Embedding(num_types, hidden_dim)
            nn.init.zeros_(self.type_null.weight)

        # Project fused vision features (used for the gated output)
        self.vision_proj = nn.Linear(hidden_dim, hidden_dim)

        # Project text features
        self.text_proj = nn.Linear(hidden_dim, hidden_dim)

        # Type-aware query projection: concat(text_cls, type_emb) → D
        self.query_proj = nn.Linear(hidden_dim * 2, hidden_dim)

        # 🔬 dynamic_peek (2026-08-09): TCVG dong theo tien trinh sinh cau, khong tinh alpha
        # 1 lan tinh dung chung ca cau. peek_embedding (tin hieu "sap can gi", tu nhan luc train
        # hoac tu draft luc eval) duoc CONG THEM vao query -- init 0 nen KHONG doi hanh vi neu
        # khong dung (an toan, tuong thich nguoc). Xem VisionGating.forward.
        self.dynamic_peek = bool(dynamic_peek)
        if self.dynamic_peek:
            self.peek_proj = nn.Linear(hidden_dim, hidden_dim)
            nn.init.zeros_(self.peek_proj.weight)
            nn.init.zeros_(self.peek_proj.bias)

        if use_delta_gate:
            # Delta gate: gate_input = cat([orig_proj(v_orig), delta_proj(v_delta), q]) → 3D
            # v_orig: raw SigLIP spatial content (pre-Flamingo)
            # v_delta = v_fused − v_orig: Flamingo's per-patch attention fingerprint
            # Together they give strong, per-question, per-patch discriminative signal.
            self.orig_proj = nn.Linear(hidden_dim, hidden_dim)
            self.delta_proj = nn.Linear(hidden_dim, hidden_dim)
            # Identity init for orig_proj: gate sees real spatial content immediately,
            # not random noise. Prevents alpha collapse to 0 in early training.
            nn.init.eye_(self.orig_proj.weight)
            nn.init.zeros_(self.orig_proj.bias)
            # Small-scale init for delta_proj: delta signal is subtle early on,
            # start near-zero and let it grow as training progresses.
            nn.init.normal_(self.delta_proj.weight, std=0.01)
            nn.init.zeros_(self.delta_proj.bias)
            gate_input_dim = hidden_dim * 3
        else:
            gate_input_dim = hidden_dim * 2

        # Gating network: outputs raw logit per patch ([B,P,1]) — hoac per patch×channel ([B,P,D])
        _gate_out = hidden_dim if self.gate_per_channel else 1
        self.gate_net = nn.Sequential(
            nn.Linear(gate_input_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, _gate_out)
        )
        # 🔬 gate_pertype_net: MOI LOAI mot mang gate rieng (y thiet ke ban dau cua tac gia).
        # Khac han type_moe (tach FFN HAU-gate) va khac gate_no_type_emb (van mot mang chung,
        # chi doi query). O day chinh HAM tinh alpha duoc tach theo loai.
        # Khoi tao CA 4 BAN GIONG HET gate_net goc -> epoch 0 hanh xu y het model hien tai,
        # moi phan ky ve sau la HOC duoc, khong phai do khoi tao khac nhau.
        self.gate_pertype_net = bool(gate_pertype_net)
        self.gate_nets = None
        if self.gate_pertype_net:
            import copy as _cp
            self.gate_nets = nn.ModuleList([_cp.deepcopy(self.gate_net) for _ in range(num_types)])
        if self.gate_per_channel:
            # zero-init lop cuoi -> alpha_pre=0 moi channel -> alpha = sigma(bias) DONG DEU tai init
            # (= gate per-patch chuan luc bat dau, khong lech channel), roi hoc tach channel dan.
            nn.init.zeros_(self.gate_net[-1].weight)
            nn.init.zeros_(self.gate_net[-1].bias)

        # Learnable bias — matches paper: α_i = σ(g_θ([v(L)_i; q]) + b)
        #
        # 🔬 type_bias: b THEO TUNG LOAI thay vi mot scalar dung chung.
        #
        # Ly do co che (do duoc, 3 seed): TypeLoss lam COUNT tot len (+1.51, 3/3 seed) nhung
        # TCVG lay lai (2/3 seed am) -> hai thanh phan TRIET TIEU o so tong hop.
        # COUNT can GIU day du instance (paper §3.3: "a Count question should preserve a
        # distributed view of object instances"), con COLOR can CHON LOC. Nhung voi mot b
        # dung chung, moi loai bi buoc vao cung mot diem lam viec cua sigmoid: khong loai nao
        # co the "khong gating" trong khi loai khac gating manh.
        # b theo loai cho COUNT hoc b -> lon (alpha -> 1, giu tat ca) va COLOR hoc b -> nho.
        #
        # Quan he voi paper: §3.3 dat ca thiet ke tren viec dieu kien hoa theo loai, nhung
        # rieng b lai type-blind. Day la HOAN THIEN thiet ke cua paper, khong phai di nguoc.
        # Init = init_bias cho MOI loai -> luc bat dau GIONG HET baseline (kiem chung duoc).
        self.type_bias = bool(type_bias)
        if self.type_bias:
            self.vision_bias = nn.Parameter(torch.full((num_types,), float(init_bias)))
        else:
            self.vision_bias = nn.Parameter(torch.tensor(init_bias))

        # Layer norm for stability
        self.layer_norm = nn.LayerNorm(hidden_dim)

        # 🔬 type_ctx (TAO CUOI CUNG co y: tao module tieu thu RNG, neu tao truoc gate_net
        # thi gate_net bi doi khoi tao va bien the khong con so sanh ghep cap duoc voi
        # baseline — da do duoc hieu ung nay, xem analyze_noise_budget/RNG check).
        # DICH TRON CO DIEU KIEN THEO LOAI, lay tu chinh anh.
        #
        # Van de goc (do duoc): dich tron la `text_pooled` — MOT vector dung chung cho moi patch
        # VA moi loai. Nen "nen mot patch" = "thay bang cung mot vector chung chung", khong mang
        # thong tin gi rieng cua loai cau hoi. Gate vi the chi LAY BOT duoc thong tin, khong bao
        # gio THEM duoc bang chung dung loai — trai voi ly thuyet §3.3 cua paper.
        # Bang chung don lai: top-k theo alpha = chon ngau nhien; che do A ~ che do B;
        # COUNT bi hai vi nen = mat instance; T0 bat kip vi decoder tu danh trong so duoc.
        #
        # Sua: c_type = Attention(query=q_type, keys/values=v^(L)) = tom tat anh THEO LOAI.
        #      blend_target = text_pooled + tanh(g_type) * c_type
        # Nen mot patch giờ = "dung boi canh thi giac lien quan den loai nay thay cho patch cuc bo".
        # Voi COUNT day dung la "a distributed view of object instances" (nguyen van paper §3.3).
        #
        # g_type: MOT gate cho MOI LOAI, init = 0 -> tanh(0)=0 -> tai khoi tao DUNG BANG paper.
        self.type_ctx = bool(type_ctx)
        if self.type_ctx:
            self.ctx_q = nn.Linear(hidden_dim, hidden_dim)
            self.ctx_k = nn.Linear(hidden_dim, hidden_dim)
            self.ctx_v = nn.Linear(hidden_dim, hidden_dim)
            self.ctx_norm = nn.LayerNorm(hidden_dim)
            self.ctx_gate = nn.Parameter(torch.zeros(num_types))

        # 🔬 attention gate (TAO CUOI CUNG co y: khong dich RNG cua cac module truoc, giu duoc
        # so sanh voi baseline o phan CHUNG). alpha_i = ag_scale * <ag_query(q), ag_key(v_i)>/sqrt(d)
        self.attn_gate = bool(attn_gate)
        if self.attn_gate:
            self.ag_query = nn.Linear(hidden_dim, hidden_dim)
            self.ag_key = nn.Linear(hidden_dim, hidden_dim)
            # scale nho luc dau -> alpha ~ sigma(bias) = mo (giong khoi tao baseline b=1.0), roi hoc lon dan
            self.ag_scale = nn.Parameter(torch.tensor(0.1))

        # 🔬 refine gate (FiLM theo loai) — TAO CUOI CUNG (khong dich RNG). init identity:
        # film_gamma=0 -> γ=1+0=1, film_beta=0 -> β=0, refine_gate_scale=0 -> tanh=0 -> v̂=LN(v).
        self.refine_gate = bool(refine_gate)
        if self.refine_gate:
            self.film_gamma = nn.Embedding(num_types, hidden_dim)
            self.film_beta = nn.Embedding(num_types, hidden_dim)
            nn.init.zeros_(self.film_gamma.weight)
            nn.init.zeros_(self.film_beta.weight)
            self.refine_gate_scale = nn.Parameter(torch.zeros(num_types))

        # 🔬 type-prototype gate: query la prototype thuan cua loai (khong t_cls). TAO CUOI CUNG.
        self.proto_gate = bool(proto_gate)
        if self.proto_gate:
            self.gate_proto = nn.Embedding(num_types, hidden_dim)
            nn.init.normal_(self.gate_proto.weight, std=0.02)   # nho -> alpha bat dau gan hang

        # 🔬 LAYERSCALE PER-TYPE (muc tieu: NON-HARM moi loai). TAO CUOI CUNG (khong dich RNG).
        #   v_out = v + beta_type * (gated - v)     ; beta hoc rieng tung loai
        # beta=0 -> v_out = v (dung bang T0, khong gate = KHONG HAI theo cau truc)
        # beta=1 -> v_out = gated (dung bang T2 hien tai)
        # Init beta=1 (khoi dau tai T2, gate_net co gradient tu dau) roi phat L2 keo ve 0 theo
        # tung loai -> loai nao KHONG loi (vd LOCATION) bi keo ve identity, loai co loi (COLOR)
        # chong lai duoc. Khac areg (mot lambda chung, ep alpha->1 xoa luon COLOR): day la
        # scalar RIENG MOI LOAI nen tach duoc dung cho tot/xau.
        self.gate_layerscale_pertype = bool(gate_layerscale_pertype)
        if self.gate_layerscale_pertype:
            self.gate_ls = nn.Parameter(torch.full((num_types,), float(gate_layerscale_init)))

        # 🔬 BLEND TARGET HOC PER-PATCH (thay t̄ tinh chung): moi patch fall-back ve mot vector
        # rieng = MLP([v_proj_i ; t̄]) -> giu da dang khong gian (khong sup ve MOT diem chung).
        # Chi co nghia khi patch mang tin hieu localize (vd DINOv2). Init ~identity-ish qua residual.
        self.gate_no_type_emb = bool(gate_no_type_emb)   # 🔬 ablation: gate CHI dua tren t_cls, BO e_type
        # 🔬 ablation bu: gate CHI dua tren e_type, BO t_cls -> chon loc khong gian theo LOAI
        # nhung MU noi dung cau hoi. Khac han mot hang so theo loai, vi alpha van bien thien
        # theo tung patch qua v_i.
        self.gate_no_text_cls = bool(gate_no_text_cls)
        # 🔬 gate_blend_vorig: dich tron = noi suy giua text_pooled va v_orig (patch TRUOC Flamingo).
        # Van de do duoc: text_pooled la MOT vector dung chung cho ca 197 patch, nen alpha=0 khong
        # phai "bo qua patch" ma la "thay patch bang cung mot vector" — decoder nhan ve mot chuoi
        # day token trung nhau. Va oracle lai HA alpha de sua loi (-0.145 tren mau sai), tuc no
        # MUON nhieu cai hang so do hon.
        # Moi bien the dich tron da thu (l6b 73.34, l6f 70.18, blend_net, LocalPool3x3, box_fuse)
        # deu la HAM CUA v_proj -> nam trong span cua chinh token decoder da nhan.
        # v_orig la bieu dien TRUOC Flamingo: decoder khong co duong nao khac de thay no.
        # gamma khoi tao 0 -> blend_target = text_pooled y het hien tai o buoc 0; moi do lech
        # ve sau la HOC duoc, khong phai do khoi tao khac.
        self.gate_blend_vorig = bool(gate_blend_vorig)
        # 🔬 gate_alpha_budget: bien alpha thanh mot PHAN BO tren cac patch thay vi 197 sigmoid
        # DOC LAP. Van de cau truc do duoc: khong co canh tranh giua cac patch, nen "lam noi" mot
        # vung KHONG lam mo vung khac — model co the dat TAT CA alpha = 1 va no lam dung the
        # (SigLIP1: LOCATION alpha=0.9995 SD=0.0002, OBJECT 0.9989). Vi vay lam phang alpha chi
        # mat 0.03 tren SigLIP1: chua bao gio co phep phan bo nao de ma pha.
        #     alpha_i = clamp(P * m * softmax(score)_i, 0, 1),  m = sigmoid(budget_logit)
        # Tong khoi luong bi giu ~ P*m, nen sang cho nay BUOC phai toi cho kia.
        # KHAC han phat thua (--gate_sparsity_lambda, da thu: 37.69 va 71.61): phat thua day alpha
        # xuong TOAN CUC; ngan sach ep PHAN BO LAI ma giu nguyen tong.
        self.gate_alpha_budget = bool(gate_alpha_budget)
        if self.gate_alpha_budget:
            import math as _m
            _b0 = max(min(float(gate_budget_init), 0.999), 0.001)
            self.budget_logit = nn.Parameter(torch.tensor([_m.log(_b0 / (1 - _b0))]))
        if self.gate_blend_vorig:
            self.vorig_proj = nn.Linear(hidden_dim, hidden_dim)
            # sigmoid(-6) = 0.0025 ~ 0 -> blend_target = text_pooled tai init.
            # KHONG duoc dung zeros(1): sigmoid(0) = 0.5, tuc da tron nua-nua ngay tu epoch 0.
            # He so DU (residual scale) khoi tao 0, KHONG qua sigmoid:
            #     blend_target = text_pooled + gamma * (v_orig_proj - text_pooled)
            # gamma=0 -> DUNG BANG text_pooled tai init (non-harm tuyet doi), va dao ham theo
            # gamma la (v_orig_proj - text_pooled) — do lon day du, khong bi sigmoid bop.
            # Da thu ca hai bien the sigmoid va deu hong theo huong khac nhau:
            #   init -6: non-harm (lech 0.0056) nhung dao ham 0.00246 -> gamma dung yen 28 epoch
            #   init -2: dao ham 0.105 nhung khoi tao da lech 0.271 -> khong con non-harm
            self.blend_gamma = nn.Parameter(torch.zeros(1))
        self.gate_blend_learned = bool(gate_blend_learned)
        if self.gate_blend_learned:
            self.blend_net = nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim), nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim))
            self.ag_key = nn.Linear(hidden_dim, hidden_dim)
            self.ag_scale = nn.Parameter(torch.tensor(0.1))

        # 🔬 SPATIAL-PRESERVING BLEND (tong quat, khong hardcode danh sach loai cau hoi):
        # blend_target hien tai la MOT vector dung chung cho moi patch (text_pooled) -> dong
        # nhat hoa, xoa cau truc TUONG DOI giua cac vung anh khac nhau. Voi cau hoi ve thuoc
        # tinh toan cuc (vd mau sac) khong sao, nhung voi cau hoi quan he khong gian (vd vi
        # tri tuong doi) thi day chinh la thu can giu. Thay vi hardcode "loai nao dung nhanh
        # nao" (khong tong quat qua dataset khac), tron MEM giua 2 dich bang he so beta HOC
        # TU CHINH CAU HOI (query = W_q[t_cls;e_type], hoac t_cls thuan neu dataset khong co
        # type_ids):
        #   blend_target_i = beta*text_pooled + (1-beta)*LocalPool3x3(v_proj)_i
        # beta=sigmoid(Linear(query)), bias khoi tao lon (+3) -> beta~0.95 luc dau, tuc gan
        # nhu T2 goc (an toan, khong pha baseline), model tu hoc giam beta cho cau hoi nao
        # can giu cau truc khong gian. Local pool khong tham so (avg 3x3) de tranh them nang
        # luc thua ma khong co tin hieu de dung (bai hoc tu type_bias/type_ctx).
        #
        # 🔬 v2 (do duoc tren pilot v1): beta hoi tu ve HANG SO RIENG CHO MOI TYPE (within-type
        # std ~0.0000-0.0006, between-type std that 0.08-0.48), va TYPE NAO bi keo xuong doi
        # theo SEED (symmetry-breaking ngau nhien luc dau train, khong phai do du lieu) — tin
        # hieu type-level la "mon hoi re" nen gradient dong het vao 1 ma tran chung
        # (query_proj) roi khong con dong luc khai thac phan text-instance-level (kho hon).
        # Sua: TACH beta thanh 2 dau DOC LAP, khong dung chung trong so:
        #   beta = sigmoid( type_head(type_emb)  +  instance_head(text_cls) )
        # type_head CHI thay type_emb (nam phan coarse-theo-type, dung y da quan sat duoc).
        # instance_head CHI thay text_cls, KHONG thay type — khong co duong tat "muon" type,
        # buoc phai tu tim tin hieu rieng cau hoi neu co that. Neu instance_head hoi tu ve 0
        # (khong dong gop) day la bang chung sach nhat tu truoc den gio rang representation
        # nay khong co tin hieu instance-level (khop voi top-k~random moi encoder da thu).
        self.spatial_blend = bool(spatial_blend)
        if self.spatial_blend:
            # base bias rieng (KHONG nam trong 2 dau) -> init beta~0.95 dung nhu cu du
            # type_ids co hay khong (2 dau ben duoi la delta thuan tuy, zero-init).
            self.blend_bias = nn.Parameter(torch.tensor(3.0))
            self.blend_type_head = nn.Linear(hidden_dim, 1, bias=False)
            nn.init.zeros_(self.blend_type_head.weight)
            self.blend_instance_head = nn.Linear(hidden_dim, 1, bias=False)
            nn.init.zeros_(self.blend_instance_head.weight)

        if self.gate_l6_fuse:
            _bn = int(gate_l6_fuse_bottleneck)
            self.l6_fuse = nn.Sequential(
                nn.Linear(2 * hidden_dim, _bn), nn.GELU(),
                nn.Linear(_bn, hidden_dim))
            nn.init.zeros_(self.l6_fuse[-1].weight)  # zero-init -> blend_target = text_pooled tai init (non-harm)
            nn.init.zeros_(self.l6_fuse[-1].bias)

    def _ctx_target(self, v_proj, query, type_ids, base_target):
        """Dich tron co dieu kien theo loai: base_target + tanh(g_type) * c_type.

        c_type = softmax(q Wq (v Wk)^T / sqrt(D)) (v Wv)  — attention MOT dau, query la
        q_type (da chua text_cls VA type_emb), keys/values lay tu chinh patch thi giac.
        Tai khoi tao ctx_gate = 0 nen tra ve DUNG base_target (khoi phuc chinh xac paper)."""
        if not getattr(self, 'type_ctx', False):
            return base_target
        d = v_proj.size(-1)
        q = self.ctx_q(query).unsqueeze(1)                       # [B, 1, D]
        k = self.ctx_k(v_proj)                                   # [B, P, D]
        w = torch.softmax((q * k).sum(-1) / (d ** 0.5), dim=-1)  # [B, P]
        c = torch.bmm(w.unsqueeze(1), self.ctx_v(v_proj))        # [B, 1, D]
        c = self.ctx_norm(c)
        if type_ids is None:
            g = torch.tanh(self.ctx_gate.mean())
        else:
            g = torch.tanh(self.ctx_gate[type_ids]).view(-1, 1, 1)   # [B, 1, 1]
        return base_target + g * c

    def _minf(self, type_ids, pc=False):
        """min_alpha per-sample theo loai (gather tu min_alpha_pertype[type_ids]) hoac scalar.
        pc=True -> [B,1,1] cho per-channel; nguoc lai [B,1] cho [B,P]."""
        mp = getattr(self, 'min_alpha_pertype', None)
        if mp is None or type_ids is None:
            return self.min_alpha
        v = mp.to(type_ids.device)[type_ids]           # [B]
        return v.view(-1, 1, 1) if pc else v.view(-1, 1)

    def _maxf(self, type_ids, pc=False):
        """max_alpha per-sample theo loai (gather tu max_alpha_pertype[type_ids]) hoac scalar.
        pc=True -> [B,1,1] cho per-channel; nguoc lai [B,1] cho [B,P]."""
        mp = getattr(self, 'max_alpha_pertype', None)
        if mp is None or type_ids is None:
            return self.max_alpha
        v = mp.to(type_ids.device)[type_ids]           # [B]
        return v.view(-1, 1, 1) if pc else v.view(-1, 1)

    def _bias(self, type_ids):
        """b dung chung (scalar) hoac b theo tung loai -> [B, 1] de broadcast tren patch.
        type_ids=None (khong co nhan loai) thi lay trung binh cac b, giu duoc hanh vi hop ly."""
        if not getattr(self, 'type_bias', False):
            return self.vision_bias
        if type_ids is None:
            return self.vision_bias.mean()
        return self.vision_bias[type_ids].unsqueeze(-1)   # [B, 1]
    
    def compute_alpha(self, vision_features, text_features, type_ids=None, type_emb_override=None,
                      text_attention_mask=None):
        """Chi tinh alpha [B, P, 1], KHONG tron. Dung cho che do fusion-gate:
        alpha duoc ap vao residual cua GCA thay vi tron hau ky."""
        v_proj = self.vision_proj(vision_features)
        t_proj = self.text_proj(text_features)
        text_cls = t_proj[:, 0, :]
        if getattr(self, 'gate_type_blind', False):
            type_ids = None; type_emb_override = None
        if getattr(self, 'gate_no_text_cls', False):
            text_cls = torch.zeros_like(text_cls)   # 🔬 gate CHI thay e_type, mu noi dung cau hoi
        if type_ids is not None:
            type_emb = (type_emb_override if type_emb_override is not None
                        else self.type_embedding(type_ids))
            if getattr(self,'gate_no_type_emb',False): type_emb = torch.zeros_like(type_emb)
            if self.norm_type_emb:
                type_emb = F.normalize(type_emb, dim=-1) * self.type_scale
            query = self.query_proj(torch.cat([text_cls, type_emb], dim=-1))
        else:
            query = self.query_proj(torch.cat([text_cls, torch.zeros_like(text_cls)], dim=-1))
        num_patches = v_proj.size(1)
        gate_input = torch.cat([v_proj, query.unsqueeze(1).expand(-1, num_patches, -1)], dim=-1)
        raw = self.gate_net(gate_input)
        a = torch.sigmoid(raw.squeeze(-1) + self._bias(type_ids))
        _minv_ca = self._minf(type_ids); _maxv_ca = self._maxf(type_ids)
        a = _minv_ca + (_maxv_ca - _minv_ca) * a
        if getattr(self, 'flatten_alpha', False):
            a = a.mean(dim=1, keepdim=True).expand_as(a)
        self.last_alpha = a
        return a.unsqueeze(-1)

    def forward(self, vision_features, text_features, type_ids=None, text_attention_mask=None,
                detach_for_gate=False, vision_orig=None, type_emb_override=None, alpha_feats=None,
                region_map=None, peek_embedding=None, gca_attn_weights=None):
        """
        Args:
            vision_features: [B, P, D]  post-Flamingo fused vision (v_fused)
            alpha_feats: [B, P, D] optional — dac trung de TINH alpha (vd lop trung gian L4 giau
                         local structure). Neu None dung v_proj. BLEND van dung vision_features (output).
            text_features:   [B, L, D]  post-Flamingo text features
            type_ids:        [B]        question type IDs (0=OBJ,1=COUNT,2=COLOR,3=LOC)
            text_attention_mask: [B, L] 1=real token, 0=padding
            detach_for_gate: cut gradient from gate_net back into Flamingo/vision_orig
            vision_orig:     [B, P, D]  pre-Flamingo features (required for delta gate).
                             v_delta = vision_features − vision_orig encodes how much
                             Flamingo changed each patch for this specific question.

        Returns:
            gated_vision: [B, P, D]  instance-level gated vision
            gate_values:  [B, P]     α per patch for monitoring
        """
        batch_size, num_patches, hidden_dim = vision_features.shape

        # 🔬 gate_type_blind: chan NGAY TRONG module, khong chi o DeterministicVQA.forward —
        # generate/eval/script oracle deu goi thang VisionGating nen chan o tren se lot.
        if getattr(self, 'gate_type_blind', False):
            type_ids = None
            type_emb_override = None

        # 1. Project fused vision for the gated OUTPUT (always with grad → Flamingo signal)
        v_proj = self.vision_proj(vision_features)  # [B, P, D]
        t_proj = self.text_proj(text_features)       # [B, L, D]

        # 2. Type-aware query: q = W_q[t_cls; e_type]
        text_cls = t_proj[:, 0, :]  # [B, D] — BOS token acts as sentence summary
        if getattr(self, 'gate_no_text_cls', False):
            text_cls = torch.zeros_like(text_cls)   # 🔬 gate CHI thay e_type, mu noi dung cau hoi
        if type_ids is not None:
            type_emb = (type_emb_override if type_emb_override is not None
                        else self.type_embedding(type_ids))
            if getattr(self,'gate_no_type_emb',False): type_emb = torch.zeros_like(type_emb)   # [B, D]
            if self.norm_type_emb:
                type_emb = F.normalize(type_emb, dim=-1) * self.type_scale
            query = self.query_proj(
                torch.cat([text_cls, type_emb], dim=-1))   # [B, D]
        else:
            query = text_cls
        # 🔬 2026-08-09: TCVG DONG (khong con tinh alpha 1 lan tinh, dung cho ca cau tra loi).
        # peek_embedding = tin hieu "sap can gi" -- train: mean embedding cua dap an that (co
        # nhan). eval: mean embedding cua draft sinh o pass 1 (xem generate()). peek_proj
        # KHOI TAO 0 -> query khong doi neu khong dung peek_embedding (an toan, tuong thich
        # nguoc hoan toan voi hanh vi cu).
        if peek_embedding is not None and getattr(self, 'dynamic_peek', False):
            query = query + self.peek_proj(peek_embedding)
        query_expanded = query.unsqueeze(1).expand(-1, num_patches, -1)  # [B, P, D]

        # 3. Gate input — two modes:
        #
        #  [DELTA GATE] use_delta_gate=True and vision_orig provided:
        #    gate sees cat([orig_proj(v_orig), delta_proj(v_delta), q]) — 3D input
        #    v_orig:  raw spatial content before any language conditioning
        #    v_delta: per-patch Flamingo attention fingerprint for this question
        #    Together: gate knows WHERE things are (v_orig) and WHAT Flamingo
        #    attended to per question (v_delta) → true instance-level gating
        #
        #  [STANDARD GATE] fallback:
        #    gate sees cat([v_proj, q]) — same as pre-delta behavior
        if self.use_delta_gate and vision_orig is not None:
            v_delta = vision_features - vision_orig  # [B, P, D]: Flamingo's contribution

            # Detach both signals from gate gradient if requested — stops feedback
            # oscillation (gate_net gradient cannot corrupt Flamingo or vision_proj)
            v_orig_g  = vision_orig.detach()  if detach_for_gate else vision_orig
            v_delta_g = v_delta.detach()      if detach_for_gate else v_delta

            o_proj = self.orig_proj(v_orig_g)    # [B, P, D]
            d_proj = self.delta_proj(v_delta_g)  # [B, P, D]
            gate_input = torch.cat([o_proj, d_proj, query_expanded], dim=-1)  # [B, P, 3D]
        else:
            # Standard gate — detach v_proj to avoid Flamingo feedback (run11 behaviour)
            # 🔬 alpha_feats: tinh alpha tu dac trung lop trung gian (local structure) neu co.
            #   NHUNG neu gate_blend_l6: alpha_feats (L6) dung lam BLEND_TARGET, con alpha tinh tu
            #   v_proj (output L12, semantic) -> quyet dinh bang semantic (bai hoc glL6).
            _l6_blend = (getattr(self, 'gate_blend_l6', False) or getattr(self, 'gate_l6_fuse', False)) and alpha_feats is not None
            _src = v_proj if (_l6_blend or alpha_feats is None) else alpha_feats
            v_for_gate = _src.detach() if detach_for_gate else _src
            gate_input = torch.cat([v_for_gate, query_expanded], dim=-1)       # [B, P, 2D]

        # 🔬 box_feat: dac trung tu region_map COCO, bat bien voi thu tu id ca the.
        #   Tinh mot lan, dung cho ca alpha (diem tiem 2) va blend_target (diem tiem 1).
        box_feat = None
        # alpha_pre phai duoc khoi tao: nhanh _use_gca_alpha KHONG gan no, nen neu khong co dong
        # nay thi diem tiem 2 se NameError thay vi bo qua.
        alpha_pre = None
        if getattr(self, 'box_content', False) and region_map is not None:
            _rmb = region_map
            if _rmb.size(1) != num_patches:
                # region_map la 196 patch that; v_proj co them pooler token o DAU (cung quy uoc
                # offset voi spatial_blend: v_proj[:, offset:] moi la patch that) -> pad o dau.
                _bpad = num_patches - _rmb.size(1)
                if _bpad > 0:
                    _rmb = torch.cat([torch.zeros(_rmb.size(0), _bpad, dtype=_rmb.dtype,
                                                  device=_rmb.device), _rmb], dim=1)
                else:
                    _rmb = _rmb[:, -num_patches:]
            _isobj = (_rmb > 0).to(v_proj.dtype)                                  # [B,P]
            _same = ((_rmb.unsqueeze(2) == _rmb.unsqueeze(1)) & (_rmb.unsqueeze(2) > 0))
            _regsz = _same.to(v_proj.dtype).sum(dim=2) / float(num_patches)       # [B,P]
            # so ca the phan biet: dem id khac 0 duy nhat moi anh (bat bien thu tu)
            _nreg = torch.tensor(
                [torch.unique(_r[_r > 0]).numel() for _r in _rmb],
                device=v_proj.device, dtype=v_proj.dtype) / float(self.box_max_inst)
            _nreg = _nreg.view(-1, 1).expand(-1, num_patches)                     # [B,P]
            box_feat = self.box_proj(torch.stack([_isobj, _regsz, _nreg], dim=-1))  # [B,P,D]
            if getattr(self, 'box_class_emb', None) is not None:
                # cong nhung lop: gio box_feat chua CA hinh hoc VA danh tinh lop
                box_feat = box_feat + self.box_class_emb(_rmb.clamp(min=0, max=self.box_class_emb.num_embeddings - 1))

        # 4. α per patch: scaled sigmoid so α ∈ [min_alpha, max_alpha]
        #
        # Formula: α = min_alpha + (max_alpha - min_alpha) · σ(gate_net + vision_bias)
        #
        # Why scaled sigmoid instead of hard clamp:
        #   Hard clamp: gradient = 0 when α hits boundary → gate can stop learning
        #   Scaled sigmoid: gradient = (max-min)·σ'(·) > 0 always → gate always learns
        #
        # With min=0.0, max=1.0 (default): α = σ(·) — identical to original formula.
        # With min=0.0, max=0.85: α ∈ [0, 0.85] → prevents saturation to 1.0,
        #   keeps α(1-α) ≥ α·0.15 > 0 → gradient of (v_proj - text_pooled) never dies.
        # 🔬 HYPOTHESIS #1 (2026-08-10): thay vi hoc gate_net/query_proj/type_embedding RIENG
        # (bo chon loc moi, tham so moi), tai su dung TRONG SO ATTENTION cua chinh GCA —
        # da duoc tinh, da duoc train cho muc dich fusion — lam alpha thang. Cau hoi can tra
        # loi: gia tri that cua TCVG nam o CONG THUC TRON (alpha*v + (1-alpha)*text_pooled,
        # VAN GIU NGUYEN o day) hay o bo chon loc rieng do hoc THEM tren cung mot tin hieu ma
        # GCA da nam? gca_attn_weights: [B,P,L] (query=patch thi giac, key=token van ban),
        # DA la xac suat hop le (softmax tren L, tu nn.MultiheadAttention text2vision) nen
        # KHONG di qua sigmoid/gate_net/bias nua — chi lay max qua cac vi tri van ban HOP LE
        # (bo padding) lam do "patch nay duoc TU NAO trong cau hoi 'hoi' manh nhat", tu nhien
        # nam trong [0,1]. KHONG THAM SO MOI trong duong tinh alpha (gate_net/type_embedding/
        # query_proj van ton tai de tuong thich state_dict nhung KHONG duoc goi toi trong nhanh nay).
        _use_gca_alpha = getattr(self, 'alpha_from_gca', False) and gca_attn_weights is not None
        if _use_gca_alpha:
            if text_attention_mask is not None:
                _pad = (text_attention_mask == 0).unsqueeze(1)               # [B,1,L]
                _attn_masked = gca_attn_weights.masked_fill(_pad, 0.0)
            else:
                _attn_masked = gca_attn_weights
            raw_sigmoid = _attn_masked.max(dim=-1).values                    # [B, P] in [0,1]
        elif getattr(self, 'global_scalar_gate', False):
            # 🔬 alpha_pre = 0 luon -> khong phu thuoc v_i, q, type -> alpha = sigma(vision_bias)
            # MOT so DUY NHAT dung chung moi patch/moi sample. Xem docstring o __init__.
            alpha_pre = vision_features.new_zeros(batch_size, num_patches)  # [B, P]
        elif getattr(self, 'proto_gate', False):
            # 🔬 TYPE-PROTOTYPE GATE (y tuong: TCVG hoc CAI CHUNG cua ca LOAI, khong phai instance).
            #   alpha_i = scale * <p_type, k(v_i)> / sqrt(d)
            # query = p_type: MOT prototype hoc duoc cho moi loai, KHONG chua t_cls.
            # Khac attn_gate (dung W_q[t_cls;e_type] = instance+type). Bo t_cls vi t_cls chinh la
            # cai GCA da dung (instance-level) -> dua vao gate la DU THUA. Prototype thuan la thu
            # DUY NHAT khong trung GCA: qua moi cau hoi mau, CUNG prototype do tim patch sac mau
            # trong tung anh. Tin hieu it nhieu hon (trung binh hoa qua instance).
            if type_ids is not None:
                p = self.gate_proto(type_ids)                  # [B, D] prototype thuan cua loai
            else:
                p = self.gate_proto.weight.mean(0, keepdim=True).expand(batch_size, -1)
            _k = self.ag_key(v_for_gate)                       # [B, P, D]
            _q = p.unsqueeze(1)                                # [B, 1, D]
            _logit = (_q * _k).sum(-1) / (v_for_gate.size(-1) ** 0.5)  # [B, P]
            alpha_pre = self.ag_scale * _logit                 # [B, P]
        elif getattr(self, 'attn_gate', False):
            # 🔬 ATTENTION GATE: alpha_i = scale * <q_type, k(v_i)> / sqrt(d).
            # Khac concat-MLP o cho type NHAN voi noi dung patch thay vi CONG vao. Voi tich vo
            # huong, doi q_type BAT BUOC doi thu tu alpha giua cac patch -> per-patch, per-type
            # la tinh chat CAU TRUC, khong phai may rui. Day la co che ma §3.3 mo ta ("type-aware
            # query cham diem patch") — query trong attention la de TRUY VAN, khong phai de cong.
            #   q_type = query (da chua W_q[t_cls; e_type])
            #   scale hoc duoc, init nho de alpha bat dau gan hang (giong baseline mo)
            _k = self.ag_key(v_for_gate)                       # [B, P, D]
            _q = self.ag_query(query).unsqueeze(1)             # [B, 1, D]
            _logit = (_q * _k).sum(-1) / (v_for_gate.size(-1) ** 0.5)  # [B, P]
            alpha_pre = self.ag_scale * _logit                 # [B, P]
        elif not _use_gca_alpha:
            if getattr(self, 'gate_nets', None) is not None and type_ids is not None:
                # dinh tuyen CUNG theo loai du doan (type head chinh xac 99.63%).
                # Chay tung loai co mat trong batch roi ghep lai — batch=12 nen re.
                alpha_pre = torch.zeros(gate_input.size(0), gate_input.size(1),
                                        self.gate_nets[0][-1].out_features,
                                        device=gate_input.device, dtype=gate_input.dtype)
                for _t in type_ids.unique():
                    _m = (type_ids == _t)
                    alpha_pre[_m] = self.gate_nets[int(_t)](gate_input[_m])
            else:
                alpha_pre = self.gate_net(gate_input)  # [B,P,1] (scalar) or [B,P,D] (per-channel)
            # 🔬 gate_spatial_pertype: TAT rieng phan dieu chinh THEO TUNG PATCH o nhung loai ma
            #   viec nen khong gian la SAI, nhung VAN GIU b_type (muc nen theo loai).
            #   Ly do: alpha = sigma(b_type + MLP([v_i;query])). b_type chiem 97% bien thien va chi
            #   dat muc tin cay thi giac cho moi loai — vo hai. Phan MLP theo patch moi la phan nen
            #   khong gian, va do la phan HAI COUNT (dem can GIU DU moi ca the, nen la mat).
            #   Khac --gate_layerscale_pertype (scale CA hieu ung gate): do da chay (ls1) va beta
            #   khong phan hoa duoc ([0.814,0.700,0.748,0.786]) vi ham muc tieu phang theo cac vo huong do.
            _sp = getattr(self, 'spatial_pertype', None)
            if _sp is not None and type_ids is not None:
                _m = _sp.to(alpha_pre.device)[type_ids].view(-1, 1, 1)   # [B,1,1]
                alpha_pre = alpha_pre * _m

        # 🔬 DIEM TIEM 2: cong noi dung box vao logit cua alpha (zero-init -> khong doi gi tai init).
        #   Cho gate biet "patch nay co thuoc mot ca the khong, vat the do to bao nhieu, anh co
        #   may ca the" — thong tin nam ngoai span cua patch SigLIP dong bang.
        if box_feat is not None and alpha_pre is not None:
            _bump = self.box_alpha_head(box_feat)   # [B,P,1] — zero-init, KHONG nhan them tanh(g)
            alpha_pre = alpha_pre + (_bump if alpha_pre.dim() == 3 else _bump.squeeze(-1))
        # 🔬 per-channel: alpha [B,P,D]; else [B,P]. proto/attn/gca-alpha khong dung per-channel.
        _use_pc = (getattr(self, 'gate_per_channel', False)
                   and not getattr(self, 'proto_gate', False)
                   and not getattr(self, 'attn_gate', False)
                   and not _use_gca_alpha)
        if _use_pc:
            _b = self._bias(type_ids)
            if torch.is_tensor(_b) and _b.numel() > 1:
                _b = _b.view(-1, 1, 1)                                          # [B,1,1] per-type
            raw_sigmoid = torch.sigmoid(alpha_pre + _b)                         # [B,P,D]
            _minv = self._minf(type_ids, pc=True)
            _maxv = self._maxf(type_ids, pc=True)
            alpha_full = _minv + (_maxv - _minv) * raw_sigmoid
            alpha = alpha_full.mean(-1)                                         # [B,P] monitor/penalty/return
            alpha_expanded = alpha_full                                        # [B,P,D] cho blend
        elif _use_gca_alpha:
            # raw_sigmoid da duoc tinh truc tiep tu gca_attn_weights o tren — KHONG qua
            # gate_net/alpha_pre, chi con ap min/max/type nhu moi nhanh khac de dong bo he thong clamp.
            _minv = self._minf(type_ids)
            _maxv = self._maxf(type_ids)
            alpha = _minv + (_maxv - _minv) * raw_sigmoid          # [B, P] ∈ [min, max]
        else:
            if alpha_pre.dim() == 3:
                alpha_pre = alpha_pre.squeeze(-1)                              # [B,P]
            if getattr(self, 'gate_alpha_budget', False):
                # 🔬 alpha la mot PHAN BO tren cac patch: sang cho nay BUOC phai toi cho kia.
                # softmax OVER PATCHES (dim=1) -> canh tranh. Tong ~ P*m duoc giu boi ngan sach m.
                _w = torch.softmax(alpha_pre + self._bias(type_ids), dim=1)    # [B, P], tong = 1
                _m = torch.sigmoid(self.budget_logit)                          # ngan sach trung binh
                alpha = (alpha_pre.size(1) * _m * _w).clamp(0.0, 1.0)          # [B, P]
            else:
                raw_sigmoid = torch.sigmoid(alpha_pre + self._bias(type_ids))  # [B, P] ∈ (0, 1)
                _minv = self._minf(type_ids)                                   # scalar hoac [B,1] per-type
                _maxv = self._maxf(type_ids)                                   # scalar hoac [B,1] per-type
                alpha = _minv + (_maxv - _minv) * raw_sigmoid          # [B, P] ∈ [min, max]

        # 🔬 flatten_alpha (CHI DUNG LUC SUY LUAN, chan doan): thay alpha bang TRUNG BINH
        # cua chinh no tren cac patch. Pha TINH CHON LOC (moi patch cung alpha) nhung GIU
        # NGUYEN BIEN DO trung binh -> khong gay dich chuyen phan phoi nhu ep alpha=1.
        # Tach bach: neu sut manh -> thiet hai den tu tinh chon loc;
        #            neu gan nhu khong sut -> thiet hai cua ep alpha=1 chi la artifact bien do.
        if getattr(self, 'flatten_alpha', False) and not _use_pc:
            alpha = alpha.mean(dim=1, keepdim=True).expand_as(alpha)

        # 🔬 ALPHA_OVERRIDE (CHI DUNG LUC SUY LUAN, chan doan): thay alpha bang gia tri do ben
        # ngoai chi dinh, [B,P] hoac [B,1] (broadcast) trong [0,1]. Van di qua he clamp
        # min/max cua chinh model nen KHONG lech phan bo so voi luc train.
        # Dung cho ORACLE ALPHA (eval.py --oracle_alpha): toi uu alpha per-sample de cuc dai
        # likelihood dap an GOLD -> TRAN TREN cua MOI thiet ke gate co the co trong cong thuc
        # blend nay. Neu tran do thap -> ho gate khong chua loi giai, moi no luc hoc alpha vo ich.
        # Neu tran do cao -> loi giai CO trong ho, cai thieu la TIN HIEU HOC (giam sat alpha).
        # Mac dinh None -> hoan toan tro, khong doi bat ky hanh vi train/eval nao.
        _ov = getattr(self, 'alpha_override', None)
        if _ov is not None and not _use_pc:
            _ov = _ov.to(device=alpha.device, dtype=alpha.dtype)
            if _ov.dim() == 2 and _ov.size(1) == 1:
                _ov = _ov.expand(-1, num_patches)
            if _ov.size(1) == num_patches - 1:
                # nhan 196 patch that, v_proj co pooler token o dau -> giu nguyen alpha cua model
                # o vi tri pooler (NaN = "khong ep")
                _ov = torch.cat([torch.full((_ov.size(0), 1), float('nan'),
                                            device=_ov.device, dtype=_ov.dtype), _ov], dim=1)
            # NaN = "giu alpha cua model o o do". Cho phep ep alpha CHI tren nhung hang/patch
            # xac dinh duoc nhan, phan con lai khong bi nhieu -> phep do sach hon.
            alpha = torch.where(torch.isnan(_ov), alpha, _minv + (_maxv - _minv) * _ov)

        if not _use_pc:
            alpha_expanded = alpha.unsqueeze(-1)  # [B, P, 1] for broadcasting
        # per-channel: alpha_expanded [B,P,D] da set o tren

        # 7. Gated combination — masked mean of text features (exclude padding)
        if text_attention_mask is not None:
            # [B, L, 1] mask; clamp denom to avoid div-by-zero on degenerate inputs
            mask = text_attention_mask.float().unsqueeze(-1)  # [B, L, 1]
            text_pooled = (t_proj * mask).sum(dim=1, keepdim=True) / mask.sum(dim=1, keepdim=True).clamp(min=1)
        else:
            text_pooled = t_proj.mean(dim=1, keepdim=True)
        text_pooled = text_pooled.expand(-1, num_patches, -1)  # [B, P, D]

        # α close to 1 → use vision features (important patches)
        # α close to 0 → use text context (suppress noise)
        if getattr(self, 'refine_gate', False):
            # 🔬 TCVG NHU BO TINH CHINH THEO LOAI (khong phai gate/nen):
            #   r_i  = γ_type ⊙ v_proj(v_i) + β_type       (FiLM: affine kenh, rieng moi loai)
            #   v̂_i = LN( v_i + tanh(g_type) · r_i )        (residual, g_type init 0 = identity)
            # Khac gate blend: KHONG tron ve t̄ (khong the chi tru), ma THEM bien doi rieng loai.
            # Song sot qua LayerNorm: FiLM doi HINH DANG kenh = doi HUONG vector, LN chi xoa
            # mean+scale nen huong giu duoc — dung cho gate cu chet (kenh bien do bi LN xoa).
            if type_ids is not None:
                _g = 1.0 + self.film_gamma(type_ids).unsqueeze(1)   # [B,1,D] init 1
                _b = self.film_beta(type_ids).unsqueeze(1)          # [B,1,D] init 0
                _gt = torch.tanh(self.refine_gate_scale[type_ids]).view(-1, 1, 1)  # init 0
            else:
                _g = 1.0 + self.film_gamma.weight.mean(0).view(1, 1, -1)
                _b = self.film_beta.weight.mean(0).view(1, 1, -1)
                _gt = torch.tanh(self.refine_gate_scale.mean())
            r = _g * v_proj + _b
            gated_vision = self.layer_norm(vision_features + _gt * r)
            self.last_alpha = alpha  # giu de bao cao, du refine khong dung alpha de tron
        elif self.gate_mode == 'multiply':
            # 🔥 Suppress THUC SU: patch bi nen tien ve vector 0, attention phia sau bo qua
            # mot cach tu nhien. Phai layer_norm TRUOC roi moi nhan alpha — neu nhan truoc
            # rooi layer_norm thi LayerNorm chuan hoa tung token ve mean 0 var 1 va XOA SACH
            # he so alpha, khien gating thanh no-op.
            gated_vision = alpha_expanded * self.layer_norm(v_proj)
        else:
            # Mac dinh (goc): tron ve text_pooled — MOT vector dung chung cho moi patch.
            # alpha=0 khong phai "bo qua patch" ma la "thay patch bang cung mot vector text".
            if getattr(self, 'gate_l6_fuse', False) and alpha_feats is not None:
                # 🔬 blend_target = text_pooled + l6_fuse([v_L12 ; proj(L6)]): L6 spatial DUOC HOC
                # bien sang khong gian semantic roi CONG vao (zero-init -> = text_pooled tai init).
                blend_target = text_pooled + self.l6_fuse(torch.cat([v_proj, alpha_feats], dim=-1))
            elif getattr(self, 'gate_blend_l6', False) and alpha_feats is not None:
                # 🔬 blend_target = proj(L6): patch suppress roi ve ban SPATIAL cua chinh no (L6),
                # mang dinh vi ma output L12 da mat -> "nen" THEM info thay vi tru ve vector rong.
                blend_target = alpha_feats
            elif getattr(self, 'gate_blend_learned', False):
                # 🔬 blend target hoc per-patch: MLP([v_proj_i ; t̄]) -> moi patch fall-back rieng
                blend_target = self.blend_net(torch.cat([v_proj, text_pooled], dim=-1))
            elif getattr(self, 'spatial_blend', False):
                # 🔬 tron mem giua text_pooled (toan cuc) va local-pool 3x3 (giu cau truc tuong
                # doi giua cac patch) — he so beta hoc tu query, KHONG hardcode theo type_ids.
                # BUG (phat hien 2026-08-08): use_siglip_pooler chen THEM 1 token toan cuc o dau
                # (num_patches=197=196+1) -> grid*grid != num_patches LUON DUNG trong moi run
                # thuc te (recipe chuan deu bat use_siglip_pooler) -> nhanh nay roi thang vao
                # else, blend_mix_head KHONG BAO GIO duoc goi trong forward -> dung yen tuyet
                # doi tai init suot qua trinh train (da xac nhan: 3/3 seed, weight=0, bias=3.0
                # sau nhieu epoch that). Sua: tach rieng 1 token toan cuc dau (neu co) khoi luoi
                # vuong truoc khi reshape, token do giu nguyen blend_target=text_pooled (dung
                # ban chat "toan cuc" cua no, khong can local-pool).
                grid = int(round(num_patches ** 0.5))
                offset = 0
                if grid * grid != num_patches:
                    grid2 = int(round((num_patches - 1) ** 0.5))
                    if grid2 * grid2 == num_patches - 1:
                        grid, offset = grid2, 1
                if grid * grid == num_patches - offset:
                    # 🔬 2026-08-09: neu co region_map (COCO instance annotation that, [B,P] chi so
                    # region tung patch), gop patch THEO RANH GIOI VAT THE THAT thay vi cua so 3x3
                    # co dinh tuy tien. CHI hoat dong khi region_map duoc truyen vao -- mac dinh
                    # None -> giu nguyen HANH VI CU (avg_pool2d 3x3), khong doi gi neu khong dung.
                    if region_map is not None:
                        _rm = region_map[:, offset:] if region_map.size(1) == num_patches else region_map
                        # same[b,i,j] = 1 neu patch i,j cung region trong sample b -- pool trung binh
                        # trong nhom, KHONG dung conv (region khong deu/khong hinh chu nhat).
                        _same = (_rm.unsqueeze(2) == _rm.unsqueeze(1)).to(v_proj.dtype)  # [B,P,P]
                        _cnt = _same.sum(dim=2, keepdim=True).clamp(min=1)
                        local_target = torch.bmm(_same, v_proj[:, offset:, :]) / _cnt  # [B,P,D]
                        self.last_region_local_target = local_target.detach()  # 🔬 debug/verify only
                    else:
                        v_grid = v_proj[:, offset:, :].transpose(1, 2).reshape(batch_size, hidden_dim, grid, grid)
                        local_target = F.avg_pool2d(v_grid, kernel_size=3, stride=1, padding=1,
                                                     count_include_pad=False)
                        local_target = local_target.reshape(batch_size, hidden_dim, grid * grid).transpose(1, 2)
                    if offset:
                        local_target = torch.cat([text_pooled[:, :offset, :], local_target], dim=1)
                    # v2: beta = sigmoid(base + type_delta + instance_delta) — 2 dau DOC LAP,
                    # khong chia se trong so, de instance_delta khong bi type_delta "an cuop"
                    # ngan sach gradient (xem docstring o __init__).
                    _instance_delta = self.blend_instance_head(text_cls)  # [B,1] — luon active
                    if type_ids is not None:
                        _type_delta = self.blend_type_head(type_emb)      # [B,1]
                    else:
                        _type_delta = torch.zeros_like(_instance_delta)
                    beta = torch.sigmoid(self.blend_bias + _type_delta + _instance_delta).unsqueeze(1)  # [B,1,1]
                    self.last_beta = beta.detach().view(-1)  # [B] — de do bien theo type, khong doi hanh vi
                    self.last_beta_instance_delta = _instance_delta.detach().view(-1)  # [B] — phan RIENG cau hoi
                    blend_target = beta * text_pooled + (1 - beta) * local_target
                else:
                    blend_target = text_pooled  # luoi patch khong vuong -> fallback an toan
            else:
                blend_target = text_pooled
            # 🔬 gate_blend_vorig: noi suy dich tron ve phia v_orig (patch TRUOC Flamingo).
            if getattr(self, 'gate_blend_vorig', False) and vision_orig is not None:
                _vo = self.vorig_proj(vision_orig)            # [B, P, D] — truoc Flamingo
                if _vo.size(1) == num_patches:
                    _bt = blend_target if blend_target.dim() == 3 else blend_target.unsqueeze(1)
                    blend_target = _bt + self.blend_gamma * (_vo - _bt)
            # 🔬 DIEM TIEM 1: nap noi dung box vao dich tron. Truoc day (1-alpha) tron ve
            #   `text_pooled` — MOT vector dung chung cho ca 197 patch, nen alpha=0 khong phai
            #   "bo qua patch" ma la "thay patch bang cung mot vector", khong chen noi dung moi.
            #   Gio dich tron mang thong tin ca the theo TUNG patch, la noi dung NGOAI span cua
            #   v_proj -> alpha tro thanh lua chon thuc: chi tiet patch <-> cau truc vat the.
            #   zero-init lop cuoi + tanh(box_g) voi g=0 -> tai init blend_target = text_pooled,
            #   tuc bang TCVG goc, khong gay hai (quy uoc repo: type_experts, l6_fuse, blend_head).
            if box_feat is not None:
                # KHONG nhan tanh(box_g): box_fuse[-1] da zero-init nen tai init da = 0 (non-harm),
                # ma gradient theo box_fuse[-1].weight VAN khac 0 nen nhanh bootstrap duoc.
                # Nhan them tanh(g) voi g=0 la zero-init THU HAI -> tich hai so 0 -> gradient theo
                # ca hai deu 0 -> nhanh chet han (do duoc 2026-08-11: box_g giu dung 0.0000 sau 40
                # epoch, moi |grad| = 0.000e+00). Quy uoc repo la zero-init DUNG MOT thua so.
                blend_target = blend_target + self.box_fuse(
                    torch.cat([v_proj, box_feat], dim=-1))
            if self.use_type_null and type_ids is not None:
                blend_target = blend_target + self.type_null(type_ids).unsqueeze(1)
            # 🔬 dich tron co dieu kien theo loai lay tu chinh anh (xem _ctx_target)
            blend_target = self._ctx_target(v_proj, query, type_ids, blend_target)
            _lm = getattr(self, 'ln_mode', 'post')
            _vc = v_proj
            if getattr(self, 'gate_gca_residual', False) and vision_orig is not None:
                _vo = self.gcares_proj(vision_orig)
                if _vo.size(1) == v_proj.size(1):
                    # beta rieng, KHONG dung lai alpha. gamma init 0 -> beta = 1 -> _vc = v_proj.
                    _bl = self.gcares_head(torch.cat([v_proj, query_expanded], dim=-1))
                    _beta = 1.0 + torch.tanh(_bl)                               # [B,P,1], init = 1.0
                    _vc = _vo + _beta * (v_proj - _vo)
            if _lm == 'pre':
                gated_vision = (alpha_expanded * self.layer_norm(_vc)
                                + (1 - alpha_expanded) * self.layer_norm(blend_target))
            elif _lm == 'none':
                gated_vision = alpha_expanded * _vc + (1 - alpha_expanded) * blend_target
            else:
                gated_vision = alpha_expanded * _vc + (1 - alpha_expanded) * blend_target
                gated_vision = self.layer_norm(gated_vision)

        # 🔥 Giu lai alpha (khong detach) de co the cong phat thua thot vao loss.
        # Ly do: cross-attention phia sau von da tu bo qua duoc patch khong lien quan,
        # nen giu tat ca patch KHONG TON GI -> alpha = 1 la nghiem toi uu hop le va
        # gating tro thanh du thua (do duoc: T0 = 71.59 vs T2 = 71.83).
        # Phat len mean(alpha) tao ra DANH DOI that: giu mot patch phai "tra gia" lambda,
        # nen mo hinh buoc phai chon patch nao dang giu — va vi moi loai cau hoi can
        # patch khac nhau, type-conditioning moi tro nen CO ICH thay vi du thua.
        # 🔬 LAYERSCALE PER-TYPE: noi tuyen tinh gate <-> identity theo tung loai.
        # beta=0 -> tra ve vision_features (ungated = T0); beta=1 -> gated (T2).
        if getattr(self, 'gate_layerscale_pertype', False):
            if type_ids is not None:
                beta = self.gate_ls[type_ids].view(-1, 1, 1)   # [B,1,1]
            else:
                beta = self.gate_ls.mean()
            gated_vision = vision_features + beta * (gated_vision - vision_features)

        self.last_alpha = alpha
        return gated_vision, alpha  # Return alpha for monitoring


# ============================================================================
# TYPE-AWARE LOGITS BIASING (Soft Vocabulary Conditioning)
# ============================================================================

class TypeAwareLogitsBias(nn.Module):
    """
    Soft logits biasing based on question type
    
    Key idea: Different question types should prefer different answer tokens
        - COLOR → boost color words (đỏ, xanh, vàng, ...)
        - COUNT → boost numbers (một, hai, ba, 1, 2, 3, ...)
        - LOCATION → boost spatial words (trên, dưới, trái, phải, ...)
        - OBJECT → no bias (all objects equally likely)
    
    Implementation: Learn type-specific bias vectors
        final_logits = base_logits + type_bias[type_id]
    
    ⚠️ This is SOFT - tokens outside preferred vocab still have probability!
    """
    def __init__(self, vocab_size, num_types=4, init_scale=0.1):
        super().__init__()
        
        # Learnable bias per type: [num_types, vocab_size]
        # Initialize small to avoid dominating base logits
        self.type_biases = nn.Parameter(
            torch.randn(num_types, vocab_size) * init_scale
        )
    
    def forward(self, logits, type_ids):
        """
        Args:
            logits: [B, seq_len, vocab_size] - base answer logits
            type_ids: [B] - question type IDs
        
        Returns:
            biased_logits: [B, seq_len, vocab_size] - type-conditioned logits
        """
        batch_size, seq_len, vocab_size = logits.shape
        
        # Get bias for each sample's type
        bias = self.type_biases[type_ids]  # [B, vocab_size]
        
        # Broadcast to match logits shape
        bias = bias.unsqueeze(1).expand(-1, seq_len, -1)  # [B, seq_len, vocab_size]
        
        # Add bias (soft reweighting)
        return logits + bias


# ============================================================================
# TYPE-SPECIFIC TEXT ADAPTER (breaks OBJECT-COUNT gradient interference)
# ============================================================================

class TypeSpecificTextAdapter(nn.Module):
    """
    Bottleneck adapter applied to the BOS/CLS position [B, D] AFTER Flamingo fusion,
    before VisionGating. Each question type gets a separate up-projection.

    Applied point: text_for_concat[:, 0, :] (post-Flamingo BOS token).
    VisionGating internally uses t_proj[:, 0, :] as its text summary for gate query,
    so modifying this position makes gating type-aware with independent type gradients.

    IMPORTANT: Never apply to full text_features [B, L, D] — disrupts decoder
    cross-attention and causes catastrophic convergence failure (run83: COUNT=0% for 8 epochs).

    Architecture: x + up[type](GELU(down(x)))  where x is [B, D]
    - Shared down: bottleneck compression
    - Type-specific up (4 × hidden_dim): zero-init → identity at start
    """
    def __init__(self, hidden_dim: int, num_types: int = 4, bottleneck: int = 64):
        super().__init__()
        self.num_types = num_types
        self.down = nn.Linear(hidden_dim, bottleneck, bias=False)
        self.act = nn.GELU()
        self.up = nn.ModuleList([
            nn.Linear(bottleneck, hidden_dim, bias=False) for _ in range(num_types)
        ])
        nn.init.normal_(self.down.weight, std=0.02)
        for up in self.up:
            nn.init.zeros_(up.weight)  # zero init → pure residual at start

    def forward(self, x: torch.Tensor, type_ids: torch.Tensor) -> torch.Tensor:
        # x: [B, L, D]  type_ids: [B]
        shared = self.act(self.down(x))          # [B, L, bottleneck]
        delta = torch.zeros_like(x)
        for t in range(self.num_types):
            mask = (type_ids == t)               # [B]
            if mask.any():
                delta[mask] = self.up[t](shared[mask]).to(x.dtype)
        return x + delta                         # residual


# ============================================================================
# ATTENTION POOLING (learned sentence summary)
# ============================================================================

class AttentionPooling(nn.Module):
    """
    Replaces BOS token or mean-pool as text_cls for VisionGating.

    Learns a scoring vector over encoder hidden states so the model can
    attend to question-relevant tokens (object nouns, color adjectives, etc.)
    instead of relying on BOS (no context) or mean-pool (dilutes content).

    Params: D (1024 for BARTpho) — negligible overhead.
    """
    def __init__(self, hidden_size: int):
        super().__init__()
        self.score = nn.Linear(hidden_size, 1, bias=False)
        nn.init.normal_(self.score.weight, std=0.02)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # hidden_states: [B, L, D]  attention_mask: [B, L] (1=real, 0=pad)
        scores = self.score(hidden_states).squeeze(-1)          # [B, L]
        if attention_mask is not None:
            scores = scores + (1.0 - attention_mask.float()) * -1e4   # mask padding
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)  # [B, L, 1]
        return (hidden_states * weights).sum(1)                 # [B, D]


# ============================================================================
# OUTPUT DATACLASS (simplified)
# ============================================================================

@dataclass
class DeterministicVQAOutput:
    """Output for deterministic VQA (no KL)"""
    answer_logits: torch.Tensor
    answer_loss: Optional[torch.Tensor] = None
    type_loss: Optional[torch.Tensor] = None  # 🔥 NEW: Auxiliary type classification loss
    total_loss: Optional[torch.Tensor] = None
    type_logits: Optional[torch.Tensor] = None  # 🔥 NEW: Type predictions [B, num_types]
    answer_cls_logits: Optional[torch.Tensor] = None  # 🔬 phan loai tren TAP DAP AN (328 lop)
    attention_weights: Optional[torch.Tensor] = None
    gate_stats: Optional[dict] = None  # Vision gate statistics
    vision_kd_loss: Optional[torch.Tensor] = None  # kept for compat (unused)
    text_kd_loss: Optional[torch.Tensor] = None    # kept for compat (unused)
    contrastive_loss: Optional[torch.Tensor] = None  # 🔥 Cross-modal contrastive loss
    divergence_loss: Optional[torch.Tensor] = None   # Inter-type gate divergence loss
    kl_pretrained_loss: Optional[torch.Tensor] = None  # 🔬 KL ve decoder BARTpho pretrained


# ============================================================================
# MAIN MODEL: DETERMINISTIC VQA (NO LATENT)
# ============================================================================

class DeterministicVQA(nn.Module):
    """
    Deterministic VQA without latent reasoning bottleneck.
    
    Architecture:
    1. Vision encoder (SigLIP) - frozen or LoRA adapted
    2. Text encoder (BART) - frozen/partially unfrozen or LoRA adapted
    3. Vision-text fusion (Flamingo gated cross-attn)
    4. Decoder cross-attn directly to fused features
    5. Answer generation
    
    NO VAE, NO KL, NO free bits!
    """
    
    def __init__(
        self,
        vision_model_name: str = 'google/siglip-base-patch16-224',  # 🔥 CHANGED: DINOv2 → SigLIP
        bartpho_model_name: str = 'vinai/bartpho-syllable',
        bartpho_revision: str = None,  # Pin commit hash to reproduce exact results
        num_fusion_layers: int = 4,  # 🔥 INCREASED: 2→4 for deeper vision-text reasoning
        gca_strength: float = 1.0,   # 🔬 he so lam yeu GCA (1.0=day du, 0.0=GCA tat)
        gca_dropout: float = 0.0,    # 🔬 xac suat tat GCA moi batch khi TRAIN (suy luan luon bat)
        gca_dropout_types: str = '',  # 🔬 '' = toan batch; '0,3' = chi OBJECT va LOCATION
        num_heads: int = 8,
        dropout: float = 0.1,
        fusion_type: str = 'text2vision',  # 🔥 NEW: 'text2vision', 'vision2text', 'bidirectional'
        gradient_checkpointing: bool = True,
        use_vision_lora: bool = False,  # 🔥 Use LoRA for vision encoder
        vision_lora_r: int = 8,  # 🔥 LoRA rank (8 recommended for ~10K samples)
        vision_lora_alpha: int = 16,  # 🔥 LoRA alpha scaling
        vision_lora_dropout: float = 0.1,  # 🔥 LoRA dropout
        use_decoder_lora: bool = False,  # 🔬 LoRA cho BARTpho DECODER (243.6M = 40.1% model)
        decoder_lora_r: int = 16,
        decoder_lora_alpha: int = 32,
        decoder_lora_dropout: float = 0.1,
        use_text_lora: bool = False,  # 🔥 NEW: Use LoRA for text encoder
        text_lora_r: int = 16,  # 🔥 Text LoRA rank (higher than vision)
        text_lora_alpha: int = 32,  # 🔥 Text LoRA alpha
        text_lora_dropout: float = 0.1,  # 🔥 Text LoRA dropout
        use_type_task: bool = False,  # 🔥 Enable type prediction head (auxiliary loss only)
        use_logits_bias: bool = False,  # 🔥 Enable TypeAwareLogitsBias (risky - use separately)
        use_vision_gate: bool = False,  # 🔥 NEW: Use vision gating
        vision_gate_init: float = 1.5,  # 🔥 Initial vision boost (>1.0 = prefer vision)
        tcvg_type_emb_std: float = None,  # 🔥 std init cho type_embedding (None = mac dinh N(0,1))
        tcvg_type_emb_init=None,  # 🔬 tensor [num_types, hidden_dim] prototype ngu nghia that (xem compute_type_prototypes.py)
        tcvg_gate_mode: str = 'blend',    # 🔥 'blend' (goc) | 'multiply' (suppress ve 0)
        tcvg_two_layer: bool = False,      # 🔬 ap TCVG sau MOI lop GCA (thay vi 1 lan sau ca hai)
        tcvg_norm_type_emb: bool = False, # 🔥 B: chuan hoa type_embedding
        tcvg_type_null: bool = False,     # 🔥 A: dich tron rieng theo loai
        tcvg_type_bias: bool = False,     # 🔬 b (vision_bias) theo tung loai thay vi scalar chung
        tcvg_type_ctx: bool = False,      # 🔬 dich tron co dieu kien theo loai, lay tu anh
        tcvg_ln_mode: str = 'post',       # 🔬 vi tri LayerNorm: post (goc) | pre | none
        tcvg_attn_gate: bool = False,     # 🔬 alpha bang tich vo huong q.k thay vi concat-MLP
        tcvg_refine_gate: bool = False,   # 🔬 TCVG nhu bo tinh chinh FiLM theo loai (khong gate)
        tcvg_proto_gate: bool = False,    # 🔬 gate bang prototype thuan cua loai (khong t_cls)
        tcvg_global_scalar_gate: bool = False,  # 🔬 alpha = sigma(bias) MOT so DUY NHAT, khong
                                                 #    phu thuoc patch/cau hoi/loai — kiem dinh
                                                 #    gia thuyet TCVG=shrinkage khong phai selector
        gca_box_tokens: bool = False,     # 🔬 token box lam key/value phu cho GCA (vao TRUOC gate)
        box_class_n: int = 0,             # 🔬 so lop cho dau phu du doan lop COCO tung patch (0 = tat)
        box_ground: bool = False,         # 🔬 box lam NHAN dau ra (multi-task), khong phai dau vao
        num_answer_classes: int = 0,      # 🔬 >0: bat dau phan loai tren tap dap an train
        gate_spatial_pertype = None,      # 🔬 [4] mask 0/1: bat phan dieu chinh theo patch cho tung loai
        gate_box_content: bool = False,   # 🔬 nap noi dung box COCO vao blend_target + logit alpha
        box_max_inst: int = 32,           # 🔬 chuan hoa so ca the (n_reg / box_max_inst)
        box_class_vocab: int = 0,         # 🔬 >0: nhung ID lop COCO cua patch vao box_feat
        tcvg_spatial_blend: bool = False,  # 🔬 blend_target = tron mem(text_pooled, local-pool
                                            #    3x3 cua v_proj) theo beta hoc tu query — giu
                                            #    cau truc tuong doi giua patch, tong quat (khong
                                            #    hardcode type taxonomy)
        tcvg_dynamic_peek: bool = False,  # 🔬 TCVG dong: alpha tinh lai theo tin hieu "sap can
                                          #    gi" (nhan luc train, draft luc eval), khong tinh
                                          #    1 lan tinh dung chung ca cau
        tcvg_alpha_from_gca: bool = False,  # 🔬 HYP #1: alpha = max attention weight cua GCA
                                             #    (da tinh cho fusion) thay vi hoc gate_net rieng
        use_type_codebook: bool = False,  # 🔬 phat hien loai KHONG GIAM SAT (VQ codebook)
        codebook_size: int = 4,           # so prototype
        codebook_beta: float = 0.25,      # he so commitment cua VQ-VAE
        codebook_lambda: float = 0.1,     # trong so cua vq_loss trong tong loss
        decoder_vision_only: bool = False,# 🔥 decoder chi cross-attend vao vision
        text_path_dropout: float = 0.0,   # 🔬 H1: xac suat che text tokens khoi decoder khi train
        concat_fusion: bool = False,      # 🔬 thay GCA bang concat cau hoi+anh re, de TCVG ganh
        summary_token: bool = False,      # 🔬 TCVG dang token-cong-them (patch giu nguyen, hai none)
        slot_attn: bool = False,          # 🔬 tien hoa: slot attention gom instance theo loai
        num_slots: int = 4,
        slot_init_std: float = 0.02,      # 🔬 do phan hoa slot luc init (0.02 = ban cu -> K ban sao)
        slot_tanh_gate: bool = False,      # 🔬 tanh-gate init 0 -> slot khong gay hai luc init
        slot_stage: str = 'post',         # 🔬 'post' = slot doc feature SAU gate (cu); 'pre' = TRUOC gate
        slot_no_type: bool = False,       # 🔬 tat dieu kien hoa theo loai cua slot (ablation "co xung ten TCVG")
        decoder_pool_vision: int = 0,     # 🔬 >0: pool 197 token thi giac ve K token truoc decoder
        tcvg_topk: int = 0,               # 🔥 >0: chi giu top-k patch theo alpha, XOA phan con lai
        tcvg_fusion_gate: bool = False,   # 🔥 alpha dieu khien cuong do hop nhat GCA theo tung patch
        tcvg_fg_2pass: bool = False,      # 🔥 2 luot: alpha tinh tu v^(L) (dung Eq.4 paper)
        tcvg_topk_random: bool = False,   # 🔥 doi chung: chon k patch ngau nhien thay vi top-alpha
        tcvg_topk_bottom: bool = False,   # 🔬 doi chung 2: chon k patch alpha THAP NHAT (nguoc top)
        vision_gate_min_alpha: float = 0.0,  # 🔥 Floor on α
        vision_gate_min_alpha_pertype=None,  # 🔬 floor α theo loai [OBJECT,COUNT,COLOR,LOCATION]: bao ve COUNT (distributed view) & OBJECT (bao hoa)
        vision_gate_max_alpha_pertype=None,  # 🔬 tran α theo loai [OBJECT,COUNT,COLOR,LOCATION]: mien COUNT khoi tran chung (can alpha gan 1)
        alpha_reg_lambda: float = 0.0,      # 🔬 (A) phat (1-alpha)^2 de gate chi gate khi giup loss
        gate_layerscale_pertype: bool = False,  # 🔬 non-harm: beta_type * (gated - v), noi gate<->identity theo loai
        gate_layerscale_init: float = 1.0,      # 🔬 init beta (1.0 = khoi dau tai T2)
        gate_layerscale_l2: float = 0.0,        # 🔬 phat L2 keo beta_type ve 0 (identity) tung loai
        gate_blend_learned: bool = False,       # 🔬 blend target hoc per-patch thay t̄ tinh chung
        gate_no_type_emb: bool = False,         # 🔬 ablation: gate dua tren t_cls, BO e_type
        gate_no_text_cls: bool = False,         # 🔬 ablation bu: gate dua tren e_type, BO t_cls
        gate_blend_vorig: bool = False,         # 🔬 dich tron = noi suy ve v_orig (truoc Flamingo)
        patch_self_attn: bool = False,          # 🔬 MODULE MOI: cho cac patch attend LAN NHAU
        psa_heads: int = 8,
        gate_alpha_budget: bool = False,        # 🔬 alpha la phan bo tren patch (co canh tranh)
        gate_budget_init: float = 0.72,
        gate_pertype_net: bool = False,         # 🔬 moi loai MOT mang gate rieng (y ban dau)
        gate_type_blind: bool = False,          # 🔬 ablation: gate KHONG biet loai (train VA test)
        type_from_gate_lambda: float = 0.0,     # 🔬 type_loss doc THI GIAC SAU GATE (grad -> gate)
        kl_pretrained_lambda: float = 0.0,      # 🔬 phat do lech khoi BARTpho pretrained (0 = TAT hoan toan)
        gate_per_channel: bool = False,         # 🔬 alpha [B,P,D]: gate per patch×channel (chon feature theo loai)
        gate_gca_residual: bool = False,        # 🔬 TCVG dieu khien luong GCA thay vi lam lai viec GCA
        gate_blend_l6: bool = False,            # 🔬 blend_target = proj(L6) spatial (can gate_vision_layer>=0)
        gate_l6_fuse: bool = False,             # 🔬 blend_target = text_pooled + l6_fuse([v_L12;L6]) (L6 -> semantic space)
        gate_l6_fuse_bottleneck: int = 256,
        vision_l6_enrich: bool = False,         # 🔬 PROBE H1/H2: v_dec = v_L12 + enrich([v_L12;L6]), KHONG gate, uncond. Test L6 co signal task khong.
        gate_harm_lambda: float = 0.0,          # 🔬 do-no-harm: phat relu(loss_gate_on - loss_gate_off) per-sample -> gate rut ve identity o sample no lam TE (giam break, giu fix)
        gate_harm_protect: bool = False,        # 🔬 EM-aligned harm: chi bao ve token gate-off argmax DUNG (chong right->wrong, dac tin hieu hon)
        gate_answer_contrastive_lambda: float = 0.0,  # 🔬 GAC: giam sat CHINH gate bang InfoNCE gated-vision <-> ANSWER embedding (khong phai question -> pha vo attractor alpha->0). Ep gate chon patch du doan dap an, phan biet per-question.
        gate_answer_contrastive_temp: float = 0.07,   # 🔬 nhiet do GAC (SimCLR default)
        gate_diversity_lambda: float = 0.0,   # 🔬 thuong do lech chuan alpha THEO PATCH trong
                                               # tung mau -- chong alpha "phang" ve gan-hang-so
                                               # ngay ca trong khoang [min,max] da kep.
        gate_vision_layer: int = -1,            # 🔬 gate tinh alpha tu hidden layer L (local structure); -1=output
        vision_backbone_layer: int = -1,        # 🔬 dung L INTERMEDIATE lam feature CHINH cho GCA+TCVG+decoder (it language-aligned, giu local/mau); -1=last_hidden
        type_moe: bool = False,                 # 🔬 type-routed MoE: moi loai 1 expert FFN (decouple types)
        type_moe_bottleneck: int = 256,
        type_moe_soft: bool = False,             # 🔬 soft-routing: tron theo softmax(type_logits) thay vi argmax
        vision_gate_max_alpha: float = 1.0,  # 🔥 Ceiling on α (0.85 prevents saturation to 1.0)
        use_type_adapter: bool = False,  # 🔥 NEW: Type-conditioned vision adapter
        type_adapter_rank: int = 64,  # 🔥 Adapter bottleneck rank
        type_adapter_bias: float = 2.0,  # 🔥 Type supervision strength
        # 🔥🔥🔥 ONLINE DISTILLATION (deprecated — removed, kept for CLI compat)
        use_distillation: bool = False,
        distill_vision: bool = True,
        distill_text: bool = True,
        vision_teacher_name: str = 'google/siglip-so400m-patch14-384',
        text_teacher_name: str = 'vinai/phobert-large',
        distill_alpha: float = 0.5,
        # 🔥 Cross-Modal Contrastive Alignment Loss
        use_contrastive: bool = False,   # Enable InfoNCE vision↔text alignment
        contrastive_lambda: float = 0.1, # λ_c weight (recommended: 0.05–0.15)
        contrastive_temp: float = 0.07,  # Temperature τ (0.07 = SimCLR default)
    use_gate_divergence: bool = False,    # Inter-type gate divergence loss
    gate_divergence_lambda: float = 0.05, # λ_div weight (recommended: 0.03–0.1)
    type_loss_weight: float = 0.2,        # Weight for auxiliary type loss
    type_branch_detach: bool = False,     # 🔬 type_loss qua nhanh rieng tren stopgrad(text_cls):
                                          # lam giau dieu kien gate (type_vec) MA KHONG nhieu generation
        label_smoothing: float = 0.1,
        focal_gamma: float = 0.0,         # Focal loss γ (0=standard CE, 2=standard focal)
        use_delta_gate: bool = False,     # Delta gate: gate_input=cat([v_orig, v_delta, q])
        use_mean_pool_cls: bool = False,   # Mean-pool valid tokens as text_cls instead of BOS
        use_attn_pool_cls: bool = False,   # Learned attention pool over encoder tokens as text_cls
        use_siglip_pooler: bool = False,   # Prepend SigLIP pooler_output as extra global vision token
        use_type_text_adapter: bool = False,  # Type-specific bottleneck adapter on encoder output
        type_text_adapter_bottleneck: int = 64,  # Bottleneck dim (64 = ~330K params)
    ):
        super().__init__()
        
        print("[DETERMINISTIC VQA] Initializing without latent reasoning...")
        print("  ✅ No VAE/KL regularization")
        print("  ✅ Direct cross-attention fusion")
        print("  ✅ Optimized for accuracy & stability")
        print(f"  🔥 Fusion type: {fusion_type}")
        print(f"  🔥 Vision Encoder: {vision_model_name}")
        
        # Store distillation config (deprecated — stubs kept for CLI compat)
        self.use_distillation = False   # KD removed — proven ineffective
        self.distill_vision = False
        self.distill_text = False
        self.distill_alpha = distill_alpha

        # 🔥 Cross-modal contrastive alignment config
        self.use_contrastive = use_contrastive
        self.contrastive_lambda = contrastive_lambda
        self.contrastive_temp = contrastive_temp
        # Inter-type gate divergence config
        self.use_gate_divergence = use_gate_divergence
        self.gate_divergence_lambda = gate_divergence_lambda
        self.type_loss_weight = type_loss_weight
        # 🔬 KL-to-pretrained: chong SUP DO phan bo dau ra do fine-tune.
        #   Do duoc (probe_kl_magnitude.py): sau khi train, entropy dau ra hep di 35.0% va lech
        #   8.41 nat so voi BARTpho pretrained; mo hinh KHONG BAO GIO phat ra dap an ngoai tu vung
        #   train (0/3001 duoi sinh tu do). Nguyen nhan la sup do do fine-tune, KHONG phai tri giac
        #   (truy hoi SigLIP xep dap an chua thay o rank 14 vs 13 cho dap an da thay).
        #   Khac PMI da that bai: PMI sua LUC SUY LUAN tren phan bo da sup; day ngan no sup LUC TRAIN.
        #   lambda = 0.0 -> KHONG nap gi, KHONG doi bat ky run cu nao.
        self.kl_pretrained_lambda = float(kl_pretrained_lambda)
        self._ref_decoder = None
        if self.kl_pretrained_lambda > 0:
            from transformers import AutoModelForSeq2SeqLM as _M
            _ref = _M.from_pretrained(bartpho_model_name)
            self._ref_decoder = _ref.model.decoder
            self._ref_lm_head = _ref.lm_head
            for _p in self._ref_decoder.parameters(): _p.requires_grad_(False)
            for _p in self._ref_lm_head.parameters(): _p.requires_grad_(False)
            self._ref_decoder.eval(); self._ref_lm_head.eval()
            print(f"  🔬 KL-to-pretrained: lambda={self.kl_pretrained_lambda} "
                  f"(moc BARTpho dong bang, chuan hoa theo CE)")
        self.type_branch_detach = type_branch_detach
        self.label_smoothing = label_smoothing
        self.focal_gamma = focal_gamma
        # Per-type label smoothing: {type_id(int) -> epsilon}
        # type 0=OBJECT, 1=COUNT, 2=COLOR, 3=LOCATION
        # None = disabled (use uniform self.label_smoothing)
        self.type_label_smoothing: Optional[dict] = None
        self.use_mean_pool_cls = use_mean_pool_cls
        self.use_attn_pool_cls = use_attn_pool_cls
        self.use_siglip_pooler = use_siglip_pooler
        self.use_type_text_adapter = use_type_text_adapter
        self.type_text_adapter_bottleneck = type_text_adapter_bottleneck
        
        self.use_type_task = use_type_task  # 🔥 Type prediction head (auxiliary loss)
        self.use_logits_bias = use_logits_bias  # 🔥 Type-aware logits biasing (optional, risky)
        
        self.use_vision_lora = use_vision_lora
        self.vision_lora_r = vision_lora_r
        self.vision_lora_alpha = vision_lora_alpha
        self.vision_lora_dropout = vision_lora_dropout
        
        # Type adapter settings
        self.use_type_adapter = use_type_adapter
        self.type_adapter_rank = type_adapter_rank
        self.type_adapter_bias = type_adapter_bias
        
        self.use_decoder_lora = use_decoder_lora
        self.decoder_lora_r = decoder_lora_r
        self.decoder_lora_alpha = decoder_lora_alpha
        self.decoder_lora_dropout = decoder_lora_dropout
        self.use_text_lora = use_text_lora  # 🔥 NEW
        self.text_lora_r = text_lora_r  # 🔥 NEW
        self.text_lora_alpha = text_lora_alpha  # 🔥 NEW
        self.text_lora_dropout = text_lora_dropout  # 🔥 NEW
        
        # 🔥 Vision gating (will be initialized after knowing bart_hidden_dim)
        self.use_vision_gate = use_vision_gate
        self.vision_gate_init = vision_gate_init
        self.tcvg_type_emb_std = tcvg_type_emb_std
        self.tcvg_type_emb_init = tcvg_type_emb_init
        self.tcvg_gate_mode = tcvg_gate_mode
        self.tcvg_norm_type_emb = tcvg_norm_type_emb
        self.tcvg_type_null = tcvg_type_null
        self.tcvg_type_bias = tcvg_type_bias
        self.tcvg_type_ctx = tcvg_type_ctx
        self.tcvg_ln_mode = tcvg_ln_mode
        self.tcvg_attn_gate = tcvg_attn_gate
        self.tcvg_refine_gate = tcvg_refine_gate
        self.tcvg_proto_gate = tcvg_proto_gate
        self.tcvg_global_scalar_gate = tcvg_global_scalar_gate
        self.use_type_codebook = use_type_codebook
        self.codebook_size = codebook_size
        self.codebook_beta = codebook_beta
        self.codebook_lambda = codebook_lambda
        self.decoder_vision_only = decoder_vision_only
        self.text_path_dropout = text_path_dropout
        self.decoder_pool_vision = decoder_pool_vision
        self.concat_fusion = bool(concat_fusion)
        self.use_summary_token = bool(summary_token)
        self.use_slot_attn = bool(slot_attn)
        self.num_slots = num_slots
        self.slot_stage = str(slot_stage)
        self.slot_no_type = bool(slot_no_type)
        self.tcvg_topk = tcvg_topk
        self.tcvg_fusion_gate = tcvg_fusion_gate
        self.tcvg_fg_2pass = tcvg_fg_2pass
        self.tcvg_topk_random = tcvg_topk_random
        self.tcvg_topk_bottom = tcvg_topk_bottom
        self.vision_gate_min_alpha = vision_gate_min_alpha
        self.vision_gate_min_alpha_pertype = vision_gate_min_alpha_pertype
        self.vision_gate_max_alpha_pertype = vision_gate_max_alpha_pertype
        self.alpha_reg_lambda = alpha_reg_lambda
        self.gate_layerscale_pertype = gate_layerscale_pertype
        self.gate_layerscale_init = gate_layerscale_init
        self.gate_layerscale_l2 = gate_layerscale_l2
        self.gate_blend_learned = gate_blend_learned
        self.gca_box_tokens = gca_box_tokens
        self.box_class_n = box_class_n
        self.box_ground = box_ground
        self.num_answer_classes = num_answer_classes
        self.gate_spatial_pertype = gate_spatial_pertype
        self.gate_box_content = gate_box_content
        self.box_max_inst = box_max_inst
        self.box_class_vocab = box_class_vocab
        self.tcvg_spatial_blend = tcvg_spatial_blend
        self.tcvg_dynamic_peek = tcvg_dynamic_peek
        self.tcvg_alpha_from_gca = tcvg_alpha_from_gca
        self.gate_no_type_emb = gate_no_type_emb
        self.gate_no_text_cls = gate_no_text_cls
        self.gate_blend_vorig = gate_blend_vorig
        # 🔬 patch_self_attn — MODULE THEM VAO, khong sua TCVG, khong doi thu tu module.
        # SU THAT CAU TRUC: fusion_type='text2vision' nghia la query=vision, key/value=text,
        # tuc PATCH ATTEND SANG TEXT. Decoder thi attend tu text SANG vision. Nen sau khi ra khoi
        # SigLIP dong bang, cac patch KHONG BAO GIO attend lan nhau o bat ky dau trong phan hoc
        # duoc cua model. Model khong co cho nao de hoc "hai patch nay thuoc cung mot vat".
        # Ma DEM can dung dieu do, va COUNT la loai co du dia lon nhat (+23.64) va te nhat (66.22).
        # Dat GIUA Flamingo va TCVG -> TCVG van SAU Flamingo (khong vi pham rang buoc bai KES),
        # va alpha duoc tinh tren feature DA BIET cau truc nhom.
        # out_proj zero-init -> tai init module la ANH XA DONG NHAT, non-harm tuyet doi.
        self.patch_self_attn = patch_self_attn
        self.psa = None
        self.gate_alpha_budget = gate_alpha_budget
        self.gate_budget_init = gate_budget_init
        self.gate_pertype_net = gate_pertype_net
        self.gate_type_blind = bool(gate_type_blind)
        self.type_from_gate_lambda = float(type_from_gate_lambda)
        self.gate_per_channel = gate_per_channel
        self.gate_gca_residual = gate_gca_residual
        self.gate_blend_l6 = gate_blend_l6
        self.gate_l6_fuse = gate_l6_fuse
        self.gate_l6_fuse_bottleneck = gate_l6_fuse_bottleneck
        self.vision_l6_enrich = vision_l6_enrich
        self.gate_harm_lambda = gate_harm_lambda
        self.gate_harm_protect = gate_harm_protect
        self.gate_answer_contrastive_lambda = gate_answer_contrastive_lambda
        self.gate_answer_contrastive_temp = gate_answer_contrastive_temp
        self.gate_diversity_lambda = gate_diversity_lambda
        self.gate_vision_layer = gate_vision_layer
        self.vision_backbone_layer = vision_backbone_layer
        if vision_backbone_layer >= 0:
            print(f"  🔬 Vision BACKBONE dùng L{vision_backbone_layer} (intermediate, ít language-align) cho GCA+TCVG+decoder")
        self.type_moe = type_moe
        self.type_moe_bottleneck = type_moe_bottleneck
        self.type_moe_soft = type_moe_soft
        self.vision_gate_max_alpha = vision_gate_max_alpha
        self.gate_detach_input = False    # set by train.py via --gate_detach_input
        self.use_delta_gate = use_delta_gate
        
        # Vision encoder (SigLIP or DINOv2)
        # For SigLIP, load full model first, then extract vision_model
        full_vision_model = AutoModel.from_pretrained(vision_model_name, attn_implementation='eager')
        
        # Extract vision-only component if it's a multi-modal model (like SigLIP)
        if hasattr(full_vision_model, 'vision_model'):
            # SigLIP has separate vision_model and text_model
            self.vision_encoder = full_vision_model.vision_model
            vision_hidden_dim = full_vision_model.config.vision_config.hidden_size
            self.is_siglip = True
            print(f"  📊 Detected SigLIP - using vision_model component only")
        else:
            # DINOv2 is vision-only already
            self.vision_encoder = full_vision_model
            vision_hidden_dim = full_vision_model.config.hidden_size
            self.is_siglip = False
            print(f"  📊 Detected DINOv2 - using full model")
        
        print(f"  📊 Vision encoder: {vision_model_name}")
        print(f"  📊 Vision hidden_dim: {vision_hidden_dim}")
        
        # 🔥 Enable gradient checkpointing BEFORE LoRA (if requested)
        # NOTE: SigLIP vision_model has compatibility issues with gradient checkpointing + LoRA
        # Since vision encoder is frozen (only LoRA adapters train), we can skip it safely
        self.gradient_checkpointing = gradient_checkpointing
        if gradient_checkpointing:
            if self.is_siglip:
                # SigLIP: Skip gradient checkpointing (frozen + LoRA = minimal memory anyway)
                print(f"  ⚠️  Vision Gradient Checkpointing: SKIPPED for SigLIP")
                print(f"      (SigLIP vision_model has implementation conflicts with PEFT)")
                print(f"      (Vision encoder is frozen, only ~1M LoRA params train - memory OK)")
            else:
                # DINOv2: Safe to enable
                if hasattr(self.vision_encoder, 'gradient_checkpointing_enable'):
                    self.vision_encoder.gradient_checkpointing_enable()
                    print(f"  🔥 Vision Gradient Checkpointing: ENABLED (DINOv2)")
                elif hasattr(self.vision_encoder, 'config'):
                    self.vision_encoder.config.gradient_checkpointing = True
                    print(f"  🔥 Vision Gradient Checkpointing: ENABLED (config-based)")
        
        # 🔥 Add LoRA to vision encoder if requested (AFTER gradient checkpointing setup)
        if use_vision_lora:
            self._inject_lora_to_vision_encoder()
            print(f"  🔥 Vision LoRA: r={vision_lora_r}, alpha={vision_lora_alpha}, dropout={vision_lora_dropout}")
        
        # Language model
        # attn_implementation='eager': forces pre-4.45 manual bmm+softmax attention,
        # matching the numerical behavior that produced the 70.38% baseline.
        # transformers 4.45+ switched MBart to attention_interface (SDPA/Flash) by default.
        #
        # tie_word_embeddings NOT overridden (default=True): lm_head.weight is tied to
        # model.shared/decoder.embed_tokens — loaded from pre-trained checkpoint.
        # BARTpho checkpoint stores the embedding as model.shared; with tying, lm_head
        # reuses this pre-trained matrix as the output projection (524M params).
        # Previous flag tie_word_embeddings=False caused lm_head to be randomly initialized
        # (checkpoint has no lm_head.weight key → MISSING → random init, 41M extra random
        # params). That regression was added in commit 300e058 after the 72.26% result.
        bartpho_full = MBartForConditionalGeneration.from_pretrained(
            bartpho_model_name, revision=bartpho_revision,
            attn_implementation='eager',
        )
        bartpho_full.config.use_cache = False

        self.tokenizer = BartphoTokenizer.from_pretrained(
            bartpho_model_name, revision=bartpho_revision
        )
        bart_hidden_dim = bartpho_full.config.d_model
        print(f"  📊 BARTpho d_model: {bart_hidden_dim}")

        self.encoder = bartpho_full.model.encoder
        self.decoder = bartpho_full.model.decoder
        self.lm_head = bartpho_full.lm_head
        
        self.config = self.encoder.config
        self.config.decoder_start_token_id = self.tokenizer.bos_token_id
        self.config.pad_token_id = self.tokenizer.pad_token_id
        self.config.eos_token_id = self.tokenizer.eos_token_id
        
        del bartpho_full
        
        # 🔥 Add LoRA to text encoder if requested
        if use_decoder_lora:
            self._inject_lora_to_decoder()
            print(f"  🔬 Decoder LoRA: r={decoder_lora_r}, alpha={decoder_lora_alpha}")
        if use_text_lora:
            self._inject_lora_to_text_encoder()
            print(f"  🔥 Text LoRA: r={text_lora_r}, alpha={text_lora_alpha}, dropout={text_lora_dropout}")
        
        # Vision position embeddings (calculate dynamically based on model)
        # SigLIP & DINOv2: 224x224 image with patch_size=16 → 14x14 = 196 patches
        # Note: Models return [batch, num_patches+1, hidden] where +1 is CLS token
        # We'll initialize for 196 patches (after removing CLS)
        # 🔬 SUY RA tu config, KHONG hard-code. Truoc day cung 196 nen doi sang backbone
        # do phan giai khac (vd siglip-base-patch16-384 -> 24x24 = 576 patch) se lech
        # vision_pos_embed va hong am tham.
        #
        # Ly do can do phan giai cao hon (do duoc): OBJECT (41.6% test) va LOCATION (22.8%)
        # cong lai dong gop RONG -0.046 vao Overall — gate tu chon alpha=1 (khong lam gi) voi
        # hai loai nay o phan lon seed. Va top-k cho thay chon patch theo alpha ngang chon
        # ngau nhien. Gia thuyet: luoi 14x14 voi dac trung dong bang qua THO de "chon vung"
        # mang thong tin. 384px -> 24x24 = 576 patch, gap 4 lan do phan giai khong gian.
        # DEM PATCH BANG DUMMY FORWARD THAT (khong dua config): DINOv2 config ghi image_size=518
        # nhung processor crop ve 224 -> config sai. Dummy forward 224x224 cho so patch THUC.
        try:
            with torch.no_grad():
                _d = next(self.vision_encoder.parameters()).device
                _dummy = torch.zeros(1, 3, 224, 224, device=_d)
                _o = self.vision_encoder(pixel_values=_dummy).last_hidden_state
                _seq = _o.shape[1]
            # tru CLS token neu la so le/co CLS (SigLIP khong CLS -> chan; DINOv2 co CLS)
            import math
            _g = int(round(math.sqrt(_seq)))
            self.num_patches = _g * _g if _g * _g == _seq else _seq - 1   # bo CLS neu khong phai so chinh phuong
        except Exception as _e:
            _vc = getattr(getattr(full_vision_model, 'config', None), 'vision_config', None) or getattr(full_vision_model, 'config', None)
            self.num_patches = (int(getattr(_vc,'image_size',224)) // int(getattr(_vc,'patch_size',16))) ** 2
        if self.num_patches != 196:
            print(f"  ⚠️  num_patches = {self.num_patches} (dummy-forward 224) — KHAC 196 mac dinh")
        self.vision_pos_embed = nn.Parameter(
            torch.randn(1, self.num_patches, vision_hidden_dim) * 0.02
        )
        print(f"  📊 Vision position embeddings: {self.num_patches} patches")
        
        # Vision projection
        self.vision_proj = nn.Sequential(
            nn.Linear(vision_hidden_dim, bart_hidden_dim),
            nn.LayerNorm(bart_hidden_dim),
            nn.Dropout(dropout)
        )
        print(f"  ✅ Vision projection: {vision_hidden_dim} → {bart_hidden_dim}")
        if self.gate_vision_layer >= 0:
            self.gate_layer_proj = nn.Linear(vision_hidden_dim, bart_hidden_dim)
            print(f"  🔬 Gate reads intermediate vision layer L{self.gate_vision_layer} (local structure)")
        if self.vision_l6_enrich:
            # PROBE H1/H2: v_dec = v_L12 + enrich([v_L12 ; proj(L6)]), UNCONDITIONAL (khong gate).
            # zero-init lop cuoi -> khoi dau = T0 chinh xac. Test L6 co mang signal task decoder dung duoc.
            self.l6_enrich = nn.Sequential(
                nn.Linear(2 * bart_hidden_dim, 256), nn.GELU(), nn.Linear(256, bart_hidden_dim))
            nn.init.zeros_(self.l6_enrich[-1].weight); nn.init.zeros_(self.l6_enrich[-1].bias)
            print(f"  🔬 PROBE vision_l6_enrich: decoder nhan v_L12 + learned(L6), no gate (test H1/H2)")
        if self.type_moe:
            _bn=self.type_moe_bottleneck
            self.type_experts = nn.ModuleList([nn.Sequential(nn.Linear(bart_hidden_dim,_bn), nn.GELU(), nn.Linear(_bn,bart_hidden_dim)) for _ in range(4)])
            for _e in self.type_experts: nn.init.zeros_(_e[-1].weight); nn.init.zeros_(_e[-1].bias)  # identity init (residual)
            print(f"  🔬 Type-routed MoE: 4 experts, bottleneck={_bn} (decouple types)")

        # SigLIP pooler_output global token (optional)
        if use_siglip_pooler:
            self.siglip_global_proj = nn.Linear(vision_hidden_dim, bart_hidden_dim)
            nn.init.eye_(self.siglip_global_proj.weight[:min(vision_hidden_dim, bart_hidden_dim), :min(vision_hidden_dim, bart_hidden_dim)])
            nn.init.zeros_(self.siglip_global_proj.bias)
            print(f"  ✅ SigLIP global token: pooler_output {vision_hidden_dim} → {bart_hidden_dim} (prepended to patches)")

        # Attention pooling for text_cls (replaces BOS or mean-pool)
        if use_attn_pool_cls:
            self.attn_pool = AttentionPooling(bart_hidden_dim)
            print(f"  ✅ Attention pooling text_cls: learned scoring over {bart_hidden_dim}D encoder tokens")

        # Type-specific text adapter (breaks OBJECT-COUNT gradient interference)
        if use_type_text_adapter:
            self.type_text_adapter = TypeSpecificTextAdapter(
                hidden_dim=bart_hidden_dim,
                num_types=4,
                bottleneck=type_text_adapter_bottleneck,
            )
            print(f"  🔥 TypeSpecificTextAdapter: bottleneck={type_text_adapter_bottleneck}, "
                  f"params={4 * type_text_adapter_bottleneck * bart_hidden_dim + type_text_adapter_bottleneck * bart_hidden_dim:,}")
        else:
            self.type_text_adapter = None
        
        # 🔥 Cross-Modal Contrastive Alignment: projection heads
        # Both heads project into a shared contrastive space (128D follows SimCLR/MoCo convention:
        # small enough to prevent dimensional collapse, large enough for InfoNCE discrimination).
        # They are applied AFTER fusion so the contrastive objective directly supervises the
        # fused representations, not the raw encoder outputs.
        _contrastive_dim = 128
        if use_contrastive:
            self.vision_contrastive_head = nn.Sequential(
                nn.Linear(bart_hidden_dim, bart_hidden_dim),
                nn.GELU(),
                nn.Linear(bart_hidden_dim, _contrastive_dim)
            )
            self.text_contrastive_head = nn.Sequential(
                nn.Linear(bart_hidden_dim, bart_hidden_dim),
                nn.GELU(),
                nn.Linear(bart_hidden_dim, _contrastive_dim)
            )
            print(f"  🔥 Cross-Modal Contrastive: λ={contrastive_lambda}, τ={contrastive_temp}, proj_dim={_contrastive_dim}")
        else:
            self.vision_contrastive_head = None
            self.text_contrastive_head = None

        # 🔬 GATE-ANSWER CONTRASTIVE (GAC): giam sat CHINH gate.
        # Anchor = mean(GATED vision) (co grad ve gate_net), positive = ANSWER embedding.
        # Dung ANSWER (khong phai question) nen attractor alpha->0 (gated -> question-text) KHONG
        # thoa man -> gate buoc phai chon patch vision du doan dap an -> phan biet per-question.
        if gate_answer_contrastive_lambda and gate_answer_contrastive_lambda > 0:
            self.gac_vision_head = nn.Sequential(
                nn.Linear(bart_hidden_dim, bart_hidden_dim),
                nn.GELU(),
                nn.Linear(bart_hidden_dim, _contrastive_dim)
            )
            self.gac_answer_head = nn.Sequential(
                nn.Linear(bart_hidden_dim, bart_hidden_dim),
                nn.GELU(),
                nn.Linear(bart_hidden_dim, _contrastive_dim)
            )
            print(f"  🔬 Gate-Answer Contrastive (GAC): λ={gate_answer_contrastive_lambda}, τ={gate_answer_contrastive_temp} (supervise gate via gated-vision <-> ANSWER)")
        else:
            self.gac_vision_head = None
            self.gac_answer_head = None

        # Flamingo-style fusion with configurable direction
        self.gca_strength = gca_strength
        # 🔬 gca_dropout: XAC SUAT tat GCA cho MOT BATCH trong luc TRAIN (suy luan luon bat day du).
        #   Van de do duoc: huan luyen chung KHONG sinh ra bo tro. Ham mat mat chi la CE tren dap an,
        #   khong co gi thuong cho viec CHIA VIEC, nen khi GCA da lam xong viec thi nghiem toi uu cua
        #   TCVG la ANH XA DONG NHAT — va no tim ra dung the: LOCATION alpha=0.9995 SD=0.0002.
        #   Ma TCVG CO nang luc that: tat GCA thi no cho +1.64 (duong 3/3 seed).
        #   Tat ngau nhien khien TCVG KHONG THE lui ve dong nhat, vi co nhung batch no phai tu ganh.
        #   Day la stochastic depth ap cho dung mot dư thua da do duoc, khong phai regularizer chung.
        self.gca_dropout = gca_dropout
        self.gca_dropout_types = ([int(x) for x in gca_dropout_types.split(',') if x.strip()]
                                  or None) if gca_dropout_types else None
        self.fusion_type = fusion_type
        self.flamingo_fusion = nn.ModuleList([
            FlamingoGatedCrossAttention(bart_hidden_dim, num_heads, dropout, fusion_type=fusion_type)
            for _ in range(num_fusion_layers)
        ])
        print(f"  ✅ Fusion: {num_fusion_layers} Flamingo layers ({fusion_type} mode)")
        # 🔬 gca_strength cho thi nghiem lam yeu GCA (mac dinh 1.0 = khong doi gi)
        _gs = getattr(self, 'gca_strength', 1.0)
        if _gs != 1.0:
            for _fl in self.flamingo_fusion:
                _fl.gca_strength = _gs
            print(f"  🔬 GCA strength = {_gs} (LAM YEU GCA co chu dich)")
        
        # 🔥 NEW: Type-Conditioned Vision Adapter (AFTER vision projection)
        if self.use_type_adapter:
            from type_conditioned_adapter import TypeConditionedVisionAdapter
            
            self.vision_adapter = TypeConditionedVisionAdapter(
                hidden_dim=vision_hidden_dim,  # Apply to vision features (768 for SigLIP)
                num_types=4,
                rank=self.type_adapter_rank,
                dropout=dropout,
                use_type_supervision=True,
                type_bias_strength=self.type_adapter_bias
            )
            print(f"  🔥 Type-Conditioned Vision Adapter: rank={self.type_adapter_rank}, bias={self.type_adapter_bias}")
        else:
            self.vision_adapter = None
        
        # 🔥 Initialize VisionGating (applied AFTER Flamingo — gate on question-conditioned features)
        # IMPORTANT: gate operates on raw projected SigLIP features, not Flamingo-fused features.
        # When gating is after Flamingo, v_proj ≈ text_pooled (both text-fused in same space)
        # → (v_proj - text_pooled) ≈ 0 → gate_net gets near-zero gradient → flat heatmap.
        # Raw SigLIP features are in a different semantic space from BARTpho text_pooled
        # → strong gradient signal → gate learns real content/spatial discrimination.
        if self.patch_self_attn:
            self.psa = nn.MultiheadAttention(bart_hidden_dim, psa_heads, dropout=dropout,
                                             batch_first=True)
            self.psa_ln = nn.LayerNorm(bart_hidden_dim)
            self.psa_out = nn.Linear(bart_hidden_dim, bart_hidden_dim)
            nn.init.zeros_(self.psa_out.weight); nn.init.zeros_(self.psa_out.bias)  # -> anh xa dong nhat
            print(f"  🔬 Patch self-attention: {psa_heads} heads, out_proj zero-init (non-harm)")

        if self.use_vision_gate:
            # So "loai" ma gate phai lap chi so: voi codebook khong giam sat, chi so la CUM
            # (0..codebook_size-1), khong phai nhan loai. Hard-code 4 se IndexError khi K > 4.
            _gate_num_types = self.codebook_size if self.use_type_codebook else 4
            self.vision_gating = VisionGating(
                hidden_dim=bart_hidden_dim,
                num_types=_gate_num_types,
                init_bias=self.vision_gate_init,
                type_emb_std=getattr(self, 'tcvg_type_emb_std', None),
                type_emb_init=getattr(self, 'tcvg_type_emb_init', None),
                gate_mode=getattr(self, 'tcvg_gate_mode', 'blend'),
                norm_type_emb=getattr(self, 'tcvg_norm_type_emb', False),
                type_null=getattr(self, 'tcvg_type_null', False),
                type_bias=getattr(self, 'tcvg_type_bias', False),
                type_ctx=getattr(self, 'tcvg_type_ctx', False),
                ln_mode=getattr(self, 'tcvg_ln_mode', 'post'),
                attn_gate=getattr(self, 'tcvg_attn_gate', False),
                refine_gate=getattr(self, 'tcvg_refine_gate', False),
                proto_gate=getattr(self, 'tcvg_proto_gate', False),
                global_scalar_gate=getattr(self, 'tcvg_global_scalar_gate', False),
                gate_layerscale_pertype=getattr(self, 'gate_layerscale_pertype', False),
                gate_layerscale_init=getattr(self, 'gate_layerscale_init', 1.0),
                gate_blend_learned=getattr(self, 'gate_blend_learned', False),
                spatial_pertype=getattr(self, 'gate_spatial_pertype', None),
                box_content=getattr(self, 'gate_box_content', False),
                box_max_inst=getattr(self, 'box_max_inst', 32),
                box_class_vocab=getattr(self, 'box_class_vocab', 0),
                spatial_blend=getattr(self, 'tcvg_spatial_blend', False),
                dynamic_peek=getattr(self, 'tcvg_dynamic_peek', False),
                alpha_from_gca=getattr(self, 'tcvg_alpha_from_gca', False),
                gate_no_type_emb=getattr(self, 'gate_no_type_emb', False),
                gate_no_text_cls=getattr(self, 'gate_no_text_cls', False),
                gate_blend_vorig=getattr(self, 'gate_blend_vorig', False),
                gate_alpha_budget=getattr(self, 'gate_alpha_budget', False),
                gate_budget_init=getattr(self, 'gate_budget_init', 0.72),
                gate_pertype_net=getattr(self, 'gate_pertype_net', False),
                gate_per_channel=getattr(self, 'gate_per_channel', False),
                gate_gca_residual=getattr(self, 'gate_gca_residual', False),
                gate_blend_l6=getattr(self, 'gate_blend_l6', False),
                gate_l6_fuse=getattr(self, 'gate_l6_fuse', False),
                gate_l6_fuse_bottleneck=getattr(self, 'gate_l6_fuse_bottleneck', 256),
                min_alpha=self.vision_gate_min_alpha,
                min_alpha_pertype=getattr(self, 'vision_gate_min_alpha_pertype', None),
                max_alpha_pertype=getattr(self, 'vision_gate_max_alpha_pertype', None),
                max_alpha=self.vision_gate_max_alpha,
                use_delta_gate=self.use_delta_gate,
            )
            # 🔬 dat co len chinh module: moi noi goi thang VisionGating (generate, eval,
            # script oracle) deu duoc chan, khong chi nhanh DeterministicVQA.forward.
            self.vision_gating.gate_type_blind = bool(getattr(self, 'gate_type_blind', False))
            gate_mode = ("global_scalar(no content/query)" if getattr(self, 'tcvg_global_scalar_gate', False)
                         else "delta(v_orig+v_delta+q)" if self.use_delta_gate else "standard(v_fused+q)")
            alpha_range = f"α∈[{self.vision_gate_min_alpha:.2f}, {self.vision_gate_max_alpha:.2f}]"
            print(f"  🔥 Type-Conditioned Vision Gating (after GCA): 4 types, init_bias={self.vision_gate_init:.2f}, {alpha_range}, gate={gate_mode}")
            if getattr(self, 'tcvg_refine_gate', False):
                print(f"  🔬 REFINE_GATE ACTIVE: FiLM per-type (v̂=LN(v+tanh(g)·(γ⊙v+β))), KHONG tron ve t̄")
            if getattr(self, 'decoder_vision_only', False):
                print(f"  🔬 DECODER_VISION_ONLY ACTIVE: decoder chi thay vision (bo text) -> TCVG la kenh DUY NHAT")
        
        # 🔥 NEW: Type prediction head (auxiliary task)
        if self.use_type_task:
            self.type_head = TypePredictionHead(
                hidden_dim=bart_hidden_dim,
                num_types=4,
                dropout=dropout
            )
            print(f"  🔥 Type Prediction Head: 4 types (OBJECT/COUNT/COLOR/LOCATION)")
            # 🔬 --type_from_gate_lambda: mot dau THU HAI doc THI GIAC SAU GATE.
            # Ly do: type_head thuong doc text_cls, ma loai cau hoi da nam san tren mat chu
            # (luat viet tay khong hoc dat 97.67%, COLOR 625/625, COUNT 444/444) -> CE do gan nhu
            # khong con gradient huu ich, va no KHONG chay qua gate (tinh o dong ~2944, truoc fusion).
            # Doc tu gated_vision thi gradient buoc phai di qua alpha -> type_loss thanh muc tieu
            # CUA GATE thay vi cua encoder van ban.
            self.type_head_gate = None
            if self.type_from_gate_lambda > 0:
                self.type_head_gate = TypePredictionHead(
                    hidden_dim=bart_hidden_dim, num_types=4, dropout=dropout)
                print(f"  🔬 Type-from-GATE head: lambda={self.type_from_gate_lambda} "
                      f"(gradient di qua alpha)")
        else:
            self.type_head = None
            self.type_head_gate = None
            print(f"  ℹ️  Type Prediction Head: DISABLED")

        # 🔬 BOX-GROUNDED MULTI-TASK: box lam NHAN DAU RA, khong phai dau vao.
        #   Khac han --gate_box_content (nap box_feat vao gate): o day box KHONG bao gio la input,
        #   nen luc suy luan KHONG can annotation nao -> con so so duoc voi bai doi thu.
        #   Hai dau phu doc tu chinh feature ma decoder doc (post-gate):
        #     ground_head : moi patch -> logit "patch nay co thuoc mot ca the khong"  (BCE)
        #     count_head  : pool toan anh -> so ca the phan biet trong anh            (L1)
        #   Nhan lay tu patch_region_map.pkl. Muc dich: ep bieu dien thi giac phai
        #   ca-the-hoa duoc, dung nang luc ma SigLIP dong bang khong co san.
        if getattr(self, 'gca_box_tokens', False):
            # 5 dac trung hinh hoc moi ca the, doc tu region_map tren luoi 14x14:
            #   cx, cy (tam), extent_y, extent_x (be rong), npatch/196 (dien tich)
            # Bat bien voi thu tu id: token duoc sinh theo tung ca the roi dua vao attention,
            # ma attention thi bat bien hoan vi tren tap key/value.
            self.box_token_proj = nn.Sequential(nn.Linear(5, 128), nn.GELU(),
                                                nn.Linear(128, bart_hidden_dim))
            print(f"  🔬 GCA box tokens: toi da {getattr(self,'box_max_inst',32)} token/anh "
                  f"lam key/value phu (vao TRUOC TCVG)")
        if getattr(self, 'num_answer_classes', 0) > 0:
            # 🔬 DAU PHAN LOAI TREN TAP DAP AN.
            #   Ly do (do duoc 2026-08-12 tren checkpoint 73.31):
            #     801 loi, 759 (94.8%) la CHON NHAM MOT DAP AN HOP LE KHAC.
            #     Chi 36 mau co gold ngoai trie, 6 mau sinh ra rac. 98.2% gold nam trong tap train.
            #     beam 1/3/5/10 + rep_penalty + trie mo rong => TAT CA 73.31 -> tim kiem KHONG phai nut that.
            #   => bai toan thuc chat la PHAN LOAI ~328 lop, ma minh dang giai bang SINH CHUOI
            #      va toi uu bang CE tren TUNG TOKEN trong khi cham bang EM tren CA CHUOI.
            #   Dau nay toi uu THANG quyet dinh can: mot softmax tren tap dap an.
            #   Doc CUNG bieu dien ma decoder doc (vision sau gate + text) de khong phai kenh khac.
            self.answer_head = nn.Sequential(
                nn.Linear(bart_hidden_dim * 2, bart_hidden_dim), nn.GELU(), nn.Dropout(0.1),
                nn.Linear(bart_hidden_dim, int(self.num_answer_classes)))
            print(f"  🔬 Answer classification head: {int(self.num_answer_classes)} lop dap an")
        if getattr(self, 'box_class_n', 0) > 0:
            # 🔬 Dau phu du doan LOP COCO cua ca the phu len tung patch (81 lop, 0 = nen).
            #   Giau hon han objectness: 6.3 bit/patch thay vi 1 bit, va 6.3 bit/patch thay vi
            #   MOT so vo huong toan cuc nhu box_feat cu (is_obj/region_size da giai ma duoc
            #   tuyen tinh tu patch dong bang o AUC 0.87 -> gan nhu khong co gi moi).
            #   De model TU HOC doi chieu danh tu trong cau hoi voi lop cua patch, khong can
            #   bang anh xa tieng Viet -> COCO viet tay.
            self.box_class_head = nn.Linear(bart_hidden_dim, int(self.box_class_n))
            print(f"  🔬 Box class head: du doan {int(self.box_class_n)} lop COCO tung patch")
        if getattr(self, 'box_ground', False):
            self.box_ground_head = nn.Linear(bart_hidden_dim, 1)
            self.box_count_head = nn.Sequential(nn.Linear(bart_hidden_dim, 128), nn.GELU(),
                                                nn.Linear(128, 1))
            print(f"  🔬 Box-Grounded multi-task heads: objectness (per-patch BCE) + count (L1)")

        # 🔬 TYPE BRANCH (detach): type_vec = MLP(stopgrad(text_cls)). type_loss chi train nhanh nay
        # + type_head -> KHONG chay gradient vao encoder chung (het nhieu generation). Gate dung
        # type_vec (qua type_emb_override) -> van duoc lam giau tin hieu loai.
        self.type_branch = None
        if self.type_branch_detach and self.use_type_task:
            self.type_branch = nn.Sequential(
                nn.Linear(bart_hidden_dim, bart_hidden_dim), nn.GELU(),
                nn.Linear(bart_hidden_dim, bart_hidden_dim))
            print(f"  🔬 Type Branch (detach): type_loss cach ly khoi encoder, feed gate qua type_vec")

        # 🔥 NEW: Type-aware logits biasing (separate flag - risky!)
        if self.use_logits_bias:
            vocab_size = self.lm_head.out_features
            self.logits_bias = TypeAwareLogitsBias(
                vocab_size=vocab_size,
                num_types=4,
                init_scale=0.1  # Small initialization to not dominate base logits
            )
            print(f"  🔥 Type-Aware Logits Bias: vocab_size={vocab_size}, 4 types")
        else:
            self.logits_bias = None
            print(f"  ℹ️  Type-Aware Logits Bias: DISABLED")
        
        # Gradient checkpointing for text encoder (vision already enabled earlier)
        if self.gradient_checkpointing:
            # Text encoder: Always supports gradient checkpointing via method
            if hasattr(self.encoder, 'gradient_checkpointing_enable'):
                self.encoder.gradient_checkpointing_enable()
                print(f"  🔥 Text Gradient Checkpointing: ENABLED")
        
        # 🔥🔥🔥 ONLINE KNOWLEDGE DISTILLATION 🔥🔥🔥
        if self.use_distillation:
            print("\n" + "="*80)
            print("🔥🔥🔥 ONLINE KNOWLEDGE DISTILLATION ENABLED 🔥🔥🔥")
            print(f"  Vision KD: {'ON' if distill_vision else 'OFF'}  |  Text KD: {'ON' if distill_text else 'OFF'}")
            print("="*80)
            
            # ── Vision Teacher (SigLIP-SO400M) ──────────────────────────────
            if distill_vision:
                print(f"  📚 Loading Vision Teacher: {vision_teacher_name}")
                vision_teacher_full = AutoModel.from_pretrained(
                    vision_teacher_name,
                    torch_dtype=torch.float16  # FP16 to save VRAM
                )
                if hasattr(vision_teacher_full, 'vision_model'):
                    self.vision_teacher = vision_teacher_full.vision_model
                else:
                    self.vision_teacher = vision_teacher_full
                for param in self.vision_teacher.parameters():
                    param.requires_grad = False
                self.vision_teacher.eval()
                self.vision_teacher_processor = AutoImageProcessor.from_pretrained(vision_teacher_name, use_fast=False)
                teacher_vision_hidden = self.vision_teacher.config.hidden_size
                print(f"     ✅ Vision Teacher loaded: {teacher_vision_hidden}D, FP16, frozen")
                
                # Projection: student RAW encoder dim → teacher vision dim.
                # We distill from raw patch_tokens (vision_hidden, e.g. 768D for SigLIP-base)
                # BEFORE vision_proj to avoid double-projection interference.
                # vision_hidden_dim is captured in outer scope.
                self.vision_distill_proj = nn.Linear(vision_hidden_dim, teacher_vision_hidden)
                print(f"  🎯 Vision KD proj: {vision_hidden_dim}D (raw encoder) → {teacher_vision_hidden}D teacher")
            else:
                self.vision_teacher = None
                self.vision_teacher_processor = None
                self.vision_distill_proj = None
                teacher_vision_hidden = None
                print(f"  ⏭️  Vision KD: SKIPPED")
            
            # ── Text Teacher (PhoBERT-large) ─────────────────────────────────
            if distill_text:
                print(f"  📚 Loading Text Teacher: {text_teacher_name}")
                self.text_teacher = AutoModel.from_pretrained(
                    text_teacher_name,
                    torch_dtype=torch.float16  # FP16 to save VRAM
                )
                for param in self.text_teacher.parameters():
                    param.requires_grad = False
                self.text_teacher.eval()
                self.text_teacher_tokenizer = AutoTokenizer.from_pretrained(text_teacher_name)
                teacher_text_hidden = self.text_teacher.config.hidden_size
                print(f"     ✅ Text Teacher loaded: {teacher_text_hidden}D, FP16, frozen")
                
                # Projection: student bart_hidden → teacher text dim
                self.text_distill_proj = nn.Linear(bart_hidden_dim, teacher_text_hidden)
                print(f"  🎯 Text proj: {bart_hidden_dim} → {teacher_text_hidden}")
            else:
                self.text_teacher = None
                self.text_teacher_tokenizer = None
                self.text_distill_proj = None
                teacher_text_hidden = None
                print(f"  ⏭️  Text KD: SKIPPED")
            
            print(f"  🎯 Distillation α={distill_alpha:.2f}")
            print("="*80 + "\n")
        else:
            self.vision_teacher = None
            self.text_teacher = None
            self.vision_teacher_processor = None
            self.text_teacher_tokenizer = None
            self.vision_distill_proj = None
            self.text_distill_proj = None
        
        # 🔬 concat-fusion module (thay GCA) — tao sau khi co bart_hidden_dim
        if self.concat_fusion:
            self.cf_proj = nn.Linear(bart_hidden_dim * 2, bart_hidden_dim)
            self.cf_gate = nn.Parameter(torch.zeros(1))

        # 🔬 TypeSlotAttention — tien hoa TCVG (tao truoc codebook)
        self.slot_attn = TypeSlotAttention(
            bart_hidden_dim, num_slots=self.num_slots, num_types=4,
            init_std=slot_init_std, tanh_gate=slot_tanh_gate) if self.use_slot_attn else None
        if self.slot_attn is not None:
            self.slot_attn.no_type = self.slot_no_type
            print(f"  🔬 TypeSlotAttention: K={self.num_slots} stage={self.slot_stage} "
                  f"no_type={self.slot_no_type} init_std={slot_init_std} tanh_gate={slot_tanh_gate}")

        # 🔬 SummaryToken (TCVG dang token-cong-them) — tao truoc codebook
        self.summary_token = SummaryToken(bart_hidden_dim, num_types=4) if self.use_summary_token else None

        # 🔬 TypeCodebook — TAO CUOI CUNG co y: module nay tieu thu RNG, neu tao som hon thi
        # moi module sau no bi doi khoi tao va bien the khong con ghep cap duoc voi baseline.
        self.type_codebook = None
        if self.use_type_codebook:
            self.type_codebook = TypeCodebook(bart_hidden_dim, num_codes=self.codebook_size,
                                              beta=self.codebook_beta)
            print(f"  🔬 TypeCodebook: {self.codebook_size} prototype, KHONG dung nhan loai")
        self.last_codebook_idx = None
        self.last_codebook_ppl = None

        # 🔬 TCVG lop thu hai (ap sau lop GCA dau tien). Tao bang deepcopy cua gate goc:
        # deepcopy KHONG tieu RNG, nen moi module khac giu nguyen khoi tao -> run nay van
        # ghep cap chinh xac voi baseline cung seed. Hai gate khoi tao giong het nhau roi
        # tach ra khi hoc.
        self.tcvg_two_layer = bool(tcvg_two_layer)
        self.vision_gating_mid = None
        if self.tcvg_two_layer and getattr(self, 'vision_gating', None) is not None:
            import copy as _copy
            self.vision_gating_mid = _copy.deepcopy(self.vision_gating)
            print("  🔬 TCVG 2 lop: gate phu ap sau lop GCA #1 (deepcopy, khong dich RNG)")

        print("[DETERMINISTIC VQA] ✓ Multi-task type-conditioned model initialized!")
    
    def _inject_lora_to_vision_encoder(self):
        """
        Inject LoRA into vision encoder using HuggingFace PEFT library.
        
        CRITICAL WARNING: SigLIP vision_model has compatibility issues with PEFT LoRA!
        - Bug: SigLIP encoder forward() conflicts with PEFT wrapper
        - Error: "got multiple values for keyword argument 'inputs_embeds'"
        - Root cause: SigLIP internal implementation incompatible with PEFT hooks
        
        RECOMMENDED SOLUTIONS:
        1. Use DINOv2 instead: --vision_model facebook/dinov2-base
        2. OR disable vision LoRA for SigLIP (freeze vision encoder completely)
        
        This method will RAISE ERROR if SigLIP + LoRA detected!
        """
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError:
            raise RuntimeError(
                "\n"
                "❌ PEFT library is REQUIRED for LoRA!\n"
                "   Install with: pip install peft\n"
                "   Then retry training.\n"
            )

        lora_config = LoraConfig(
            r=self.vision_lora_r,
            lora_alpha=self.vision_lora_alpha,
            lora_dropout=self.vision_lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj"],
            bias="none",
            task_type="FEATURE_EXTRACTION"
        )

        if self.is_siglip:
            # SigLIP: wrapping the outer SiglipVisionTransformer with PEFT causes
            # "got multiple values for keyword argument inputs_embeds" because PEFT
            # intercepts forward() args. Fix: wrap only the inner SiglipEncoder
            # (pure transformer stack, no pixel embedding logic) — forward conflict avoided.
            target = self.vision_encoder.encoder
            self.vision_encoder.encoder = get_peft_model(target, lora_config)
            trainable_params = sum(p.numel() for p in self.vision_encoder.encoder.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.vision_encoder.parameters())
            print(f"  [LoRA] Vision (SigLIP inner encoder) - Trainable: {trainable_params:,} ({trainable_params/total_params*100:.2f}%) | Total: {total_params:,}")
        else:
            # DINOv2 / other: safe to wrap full model
            self.vision_encoder = get_peft_model(self.vision_encoder, lora_config)
            trainable_params = sum(p.numel() for p in self.vision_encoder.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.vision_encoder.parameters())
            print(f"  [LoRA] Vision - Trainable: {trainable_params:,} ({trainable_params/total_params*100:.2f}%) | Total: {total_params:,}")
    
    def set_qgnd_vocab(self, token_ids, lam=0.0, temp=0.07):
        """Nap tu vung NEO cho QGND: cac token NOI DUNG lay tu cau hoi train.

        Goi tu train.py sau khi dung model. Dang buffer nen di theo .to(device) va nam trong
        state_dict -> eval tai lai duoc dung tap token, khong lech am tham.
        lam = 0 -> hoan toan tro, khong doi hanh vi cu mot chut nao.
        """
        import torch as _t
        ids = _t.as_tensor(sorted(set(int(i) for i in token_ids)), dtype=_t.long)
        if hasattr(self, 'qgnd_ids'):
            delattr(self, 'qgnd_ids')
        self.register_buffer('qgnd_ids', ids, persistent=True)
        self.qgnd_lambda = float(lam)
        self.qgnd_temp = float(temp)
        print(f"  🔬 QGND: neo cau noi bang {ids.numel()} token noi dung tu cau hoi "
              f"(lambda={lam}, temp={temp}) — 0 tham so moi, dung lm_head.weight lam bo phan loai")

    def _inject_lora_to_decoder(self):
        """LoRA cho BARTpho decoder — nham thang vao khoang cach TONG QUAT HOA.

        Do duoc: decoder 243.6M (40.1% model) + lm_head 41.0M ~= toan bo 279.8M trainable,
        tren 10.800 mau train = 26.000 tham so moi vi du. train 99.12% vs test 73.31 (chenh 25.8).
        Dong bang han (--freeze_decoder) co the qua manh vi decoder phai thich nghi voi bieu dien
        fused; LoRA la muc trung gian: giu kha nang thich nghi, cat dung luong ~200 lan.
        Cung target_modules voi encoder LoRA (q/k/v_proj) de nhat quan; BART decoder co ca
        self_attn lan encoder_attn nen PEFT se ap vao ca hai.
        """
        try:
            from peft import LoraConfig, get_peft_model
            print(f"  [LoRA] Injecting into BARTpho DECODER (r={self.decoder_lora_r})...")
            cfg = LoraConfig(
                r=self.decoder_lora_r,
                lora_alpha=self.decoder_lora_alpha,
                lora_dropout=self.decoder_lora_dropout,
                target_modules=["q_proj", "k_proj", "v_proj"],
                bias="none",
                task_type="FEATURE_EXTRACTION",
            )
            self.decoder = get_peft_model(self.decoder, cfg)
            tr = sum(p.numel() for p in self.decoder.parameters() if p.requires_grad)
            to = sum(p.numel() for p in self.decoder.parameters())
            print(f"  [LoRA] Decoder - Trainable: {tr:,} ({tr/to*100:.2f}%) | Total: {to:,}")
        except ImportError:
            print(f"  ⚠️  PEFT library not found!")

    def _inject_lora_to_text_encoder(self):
        """
        Inject LoRA into BARTpho text encoder using PEFT library
        
        This adapts the encoder with low-rank matrices instead of unfreezing layers.
        Benefits:
        - 10x fewer parameters than unfreezing 3 layers (~1.5M vs ~18M)
        - Adapts ALL 12 layers instead of just last 3
        - More stable training (no gradient mismatch between frozen/unfrozen layers)
        - Better generalization on low-resource datasets (~10K samples)
        
        Proven effective in LoRA paper: https://arxiv.org/abs/2106.09685
        """
        try:
            from peft import LoraConfig, get_peft_model
            
            print(f"  [LoRA] Injecting into BARTpho encoder (r={self.text_lora_r})...")
            
            # LoRA config for text encoder (BARTpho)
            # Note: Use FEATURE_EXTRACTION not SEQ_2_SEQ_LM because encoder doesn't generate
            lora_config = LoraConfig(
                r=self.text_lora_r,
                lora_alpha=self.text_lora_alpha,
                lora_dropout=self.text_lora_dropout,
                target_modules=["q_proj", "k_proj", "v_proj"],  # BART uses *_proj naming
                bias="none",
                task_type="FEATURE_EXTRACTION"  # Encoder is feature extractor, not generator
            )
            
            # Apply LoRA to encoder (PEFT handles everything!)
            self.encoder = get_peft_model(self.encoder, lora_config)
            
            # Print trainable parameters
            trainable_params = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.encoder.parameters())
            print(f"  [LoRA] Text Encoder - Trainable: {trainable_params:,} ({trainable_params/total_params*100:.2f}%) | Total: {total_params:,}")
            
        except ImportError:
            print(f"  ⚠️  PEFT library not found!")
            print(f"      Install with: pip install peft")
            print(f"      Text encoder will remain frozen unless you unfreeze layers manually")
            raise RuntimeError("PEFT required for text LoRA. Install: pip install peft")
    
    def freeze_pretrained(
        self,
        unfreeze_encoder_layers: int = 3,
        unfreeze_decoder: bool = True,
        freeze_lm_head: bool = False
    ):
        """
        Freeze pretrained components (vision frozen, optionally with LoRA)
        
        Args:
            unfreeze_encoder_layers: Number of text encoder layers to unfreeze (from end)
            unfreeze_decoder: Whether to unfreeze decoder
        
        Note: Vision encoder is ALWAYS frozen except for LoRA adapters (if enabled)
        """
        # 🔥 Vision Encoder: PEFT LoRA handles freezing automatically
        if self.use_vision_lora:
            try:
                from peft import PeftModel
                if isinstance(self.vision_encoder, PeftModel):
                    # DINOv2: full model wrapped by PEFT — base frozen, LoRA trainable
                    trainable = sum(p.numel() for p in self.vision_encoder.parameters() if p.requires_grad)
                    print(f"[Freeze] Vision encoder: FROZEN (base) + PEFT LoRA ({trainable/1e6:.2f}M params)")
                elif hasattr(self.vision_encoder, 'encoder') and isinstance(self.vision_encoder.encoder, PeftModel):
                    # SigLIP: inner encoder wrapped — freeze outer, re-enable LoRA params
                    for param in self.vision_encoder.parameters():
                        param.requires_grad = False
                    for name, param in self.vision_encoder.named_parameters():
                        if 'lora_' in name:
                            param.requires_grad = True
                    trainable = sum(p.numel() for p in self.vision_encoder.parameters() if p.requires_grad)
                    print(f"[Freeze] Vision encoder: FROZEN (base) + SigLIP inner LoRA ({trainable/1e6:.2f}M params)")
                else:
                    print(f"[Freeze] Vision encoder: WARNING - LoRA requested but not applied!")
            except ImportError:
                raise RuntimeError("PEFT not installed but vision LoRA requested!")
        else:
            # Manually freeze if no LoRA
            for param in self.vision_encoder.parameters():
                param.requires_grad = False
            print(f"[Freeze] Vision encoder: FULLY FROZEN")
        
        # 🔥 Text Encoder: Freeze base, then restore LoRA OR last N layers
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        # Handle text LoRA (RECOMMENDED for low-resource)
        if self.use_text_lora:
            try:
                from peft import PeftModel
                if isinstance(self.encoder, PeftModel):
                    # Freeze overwrite above, need to re-enable LoRA params explicitly
                    for name, param in self.encoder.named_parameters():
                        if 'lora_' in name:
                            param.requires_grad = True
                    trainable = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
                    print(f"[Freeze] Text encoder: FROZEN (base) + PEFT LoRA TRAINABLE ({trainable/1e6:.2f}M params)")
                    print(f"         ✅ Adapting ALL 12 layers with low-rank matrices")
                else:
                    print(f"[Freeze] Text encoder: LoRA requested but PEFT not applied")
            except ImportError:
                print(f"[Freeze] Text encoder: LoRA requested but PEFT not installed")
        
        # Fallback: Unfreeze last N layers (OLD METHOD - less efficient)
        elif unfreeze_encoder_layers > 0:
            for layer in self.encoder.layers[-unfreeze_encoder_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True
            trainable = sum(p.numel() for layer in self.encoder.layers[-unfreeze_encoder_layers:] 
                          for p in layer.parameters() if p.requires_grad)
            print(f"[Freeze] Text encoder: Last {unfreeze_encoder_layers} layers UNFROZEN ({trainable/1e6:.2f}M params)")
            print(f"         ⚠️  Consider using --use_text_lora for better efficiency!")
        else:
            print(f"[Freeze] Text encoder: FULLY FROZEN")
        
        # 🔬 freeze_lm_head (2026-08-29): lm_head tied voi embedding BARTpho pretrain, va o nhanh
        # LoRA duoi day no van duoc mo khoa (41M tham so). Do duoc tren checkpoints_g_s1_s1:
        #   Spearman(log tan suat dap an trong train, ||W||)      = -0.052 TRUOC train
        #   Spearman(log tan suat dap an trong train, ||W||-||W0||) = +0.507 SAU train
        # tuc chinh qua trinh train NAN prior tan suat vao ma tran output; va 9 token chi thuoc
        # lop bi giu lai bi CO lai -0.1374 so voi -0.0095 cua token da train (~14x).
        # Prior do la thu de bep lop chua tung train (xem analysis/openvocab/): lop chua train
        # van NHIN THAY duoc (40.2% vs chance 11.1% trong 9 lop) nhung thua moi lop da train
        # cung loai. Dong bang lm_head = bo mot kenh hoc prior do, GIU NGUYEN kien truc.
        # Mac dinh False -> khong doi bat ky run cu nao.
        _lm_head_frozen_by_flag = bool(freeze_lm_head)

        # Decoder
        if unfreeze_decoder:
            if getattr(self, 'use_decoder_lora', False):
                # 🔬 LoRA decoder: khoi freeze nay chay SAU khi PEFT tiem LoRA, va neu bat
                #   requires_grad=True cho TOAN BO decoder thi no HUY dung tac dung cua LoRA
                #   (do duoc 2026-08-11: tong trainable 279.8M -> 282.2M, tuc TANG chu khong giam).
                #   Voi LoRA, chi cac tham so lora_* duoc train; base decoder giu dong bang.
                for n, param in self.decoder.named_parameters():
                    param.requires_grad = ('lora_' in n)
                for param in self.lm_head.parameters():
                    param.requires_grad = True
                _tr = sum(p.numel() for p in self.decoder.parameters() if p.requires_grad)
                print(f"[Freeze] Decoder: chi LoRA trainable ({_tr:,}) | LM head: UNFROZEN")
            else:
                for param in self.decoder.parameters():
                    param.requires_grad = True
                for param in self.lm_head.parameters():
                    param.requires_grad = True
                print(f"[Freeze] Decoder + LM head: UNFROZEN")
        else:
            for param in self.decoder.parameters():
                param.requires_grad = False
            for param in self.lm_head.parameters():
                param.requires_grad = False
            print(f"[Freeze] Decoder + LM head: FROZEN")

        if _lm_head_frozen_by_flag:
            for param in self.lm_head.parameters():
                param.requires_grad = False
            _n = sum(p.numel() for p in self.lm_head.parameters())
            print(f"[Freeze] 🔬 LM head DONG BANG theo --freeze_lm_head ({_n:,} tham so giu nguyen "
                  f"BARTpho pretrain) — chan kenh hoc prior tan suat dap an")
    
    def _extract_teacher_vision_features(self, images_384):
        """
        Extract vision teacher features (SigLIP-SO400M at 384px)
        
        Args:
            images_384: [B, 3, 384, 384] - Images preprocessed for teacher
            
        Returns:
            teacher_vision_features: [B, 729, 1152] - Teacher patch embeddings
        """
        with torch.no_grad():
            teacher_outputs = self.vision_teacher(pixel_values=images_384.half())
            teacher_patches = teacher_outputs.last_hidden_state  # [B, 730, 1152] with CLS
            
            # Remove CLS token
            if teacher_patches.size(1) > 729:  # 729 patches + 1 CLS (or any extra token)
                teacher_patches = teacher_patches[:, 1:, :]  # [B, 729, 1152]
            
            return teacher_patches.float()  # Convert back to FP32 for loss
    
    def _extract_teacher_text_features(self, raw_questions):
        """
        Extract text teacher features (PhoBERT-large)
        
        Args:
            raw_questions: List[str] - Raw Vietnamese questions
            
        Returns:
            teacher_text_features: [B, 1024] - Teacher question embeddings (CLS)
        """
        with torch.no_grad():
            # Tokenize with teacher's tokenizer
            teacher_inputs = self.text_teacher_tokenizer(
                raw_questions,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors='pt'
            ).to(self.text_teacher.device)
            
            teacher_outputs = self.text_teacher(**teacher_inputs)
            teacher_cls = teacher_outputs.last_hidden_state[:, 0, :]  # [B, 1024]
            
            return teacher_cls.float()  # Cast FP16 → FP32 for loss computation
    
    def compute_distillation_loss(
        self,
        student_vision_patches,  # [B, 196, bart_hidden] — post vision_proj, PRE-fusion
        student_text_features,   # [B, bart_hidden] — CLS token from text encoder
        teacher_vision_patches,  # [B, 729, teacher_vision_hidden]
        teacher_text_features    # [B, teacher_text_hidden]
    ):
        """
        Compute feature-level knowledge distillation losses.

        Design rationale
        ----------------
        The student vision encoder (SigLIP-base, 224px, 768D) must learn to produce
        features that are *semantically aligned* with the teacher (SigLIP-SO400M,
        384px, 1152D).  Two complementary objectives are used:

        (1) Vision KD — patch-level feature mimicry (Romero et al., FitNets 2015):
            • The teacher has 729 patches (384/14²); the student has 196 (224/16²).
            • A fixed average-pool reduces teacher → 196 slots so every student
              patch has a 1-to-1 positional target.  No student features enter the
              target computation, eliminating the circular-gradient problem that
              existed in the previous cross-attention formulation.
            • Loss = cosine distance (scale-invariant, more robust to norm mismatch
              across architectures than raw MSE) — Park et al., RKD CVPR 2019.

        (2) Text KD — sentence-level semantic alignment (Hinton et al. 2015):
            • CLS-pooled student (bart_hidden) → projected → teacher dim.
            • Same cosine distance loss.

        Total loss formula (Hinton 2015):
            L = (1 - α) * L_CE  +  α * L_KD

        Returns
        -------
        vision_kd_loss : scalar tensor
        text_kd_loss   : scalar tensor
        """
        zero = torch.tensor(0.0, device=student_vision_patches.device)

        # ── Vision KD ──────────────────────────────────────────────────────
        if self.distill_vision and teacher_vision_patches is not None:
            # Step 1: Reduce teacher 729 patches → 196 student slots via
            #         adaptive average pooling along the sequence dimension.
            #         This is a FIXED deterministic mapping — no learnable params,
            #         no student gradient in the target.
            # teacher_vision_patches: [B, 729, D_t]
            # Reshape → [B, D_t, 729] → AdaptiveAvgPool1d(196) → [B, D_t, 196]
            # → transpose → [B, 196, D_t]
            with torch.no_grad():
                teacher_pooled = F.adaptive_avg_pool1d(
                    teacher_vision_patches.transpose(1, 2),  # [B, D_t, 729]
                    output_size=student_vision_patches.size(1)  # 196
                ).transpose(1, 2)  # [B, 196, D_t]

            # Step 2: Project student features into teacher embedding space
            student_proj = self.vision_distill_proj(student_vision_patches)  # [B, 196, D_t]

            # Step 3: Cosine distance loss averaged over all patch positions
            # Flatten patches for vectorised cosine computation
            s = student_proj.reshape(-1, student_proj.size(-1))       # [B*196, D_t]
            t = teacher_pooled.reshape(-1, teacher_pooled.size(-1))   # [B*196, D_t]
            vision_kd_loss = (1.0 - F.cosine_similarity(s, t, dim=-1)).mean()
        else:
            vision_kd_loss = zero

        # ── Text KD ────────────────────────────────────────────────────────
        if self.distill_text and teacher_text_features is not None:
            student_text_proj = self.text_distill_proj(student_text_features)  # [B, D_t]
            text_kd_loss = (1.0 - F.cosine_similarity(
                student_text_proj,
                teacher_text_features.detach(),
                dim=-1
            )).mean()
        else:
            text_kd_loss = zero

        return vision_kd_loss, text_kd_loss

    def compute_contrastive_loss(
        self,
        fused_vision: torch.Tensor,  # [B, num_patches, D] — fused vision after Flamingo
        text_cls: torch.Tensor,       # [B, D]              — BARTpho CLS (LoRA-adapted)
        labels: Optional[torch.Tensor] = None,  # [B, seq_len] — for false-negative masking
    ) -> torch.Tensor:
        """
        Cross-Modal Contrastive Alignment Loss (InfoNCE / NT-Xent style).

        Goal
        ----
        Force fused_vision and text_cls to be semantically aligned:
          - Same sample (i, i) → high cosine similarity   (positive pair)
          - Different samples (i, j≠i) → low similarity   (negatives)

        This directly addresses the English-vision / Vietnamese-text alignment gap
        that cross-entropy alone cannot close.

        Design choices
        ---------------
        • Align FUSED vision (post-Flamingo) not raw patches:
            After Flamingo cross-attention, vision has attended to Vietnamese text
            → the Flamingo gates are directly responsible for alignment quality.

        • Mean-pool vision → single vector:
            [B, 196, D] → [B, D]. More stable than CLS-only after fusion layers.

        • False-negative masking (critical for PK Sampling + Vietnamese VQA):
            PK Sampling puts K samples of the same type in every batch.
            Many of them share the same answer (e.g. 8 COUNT samples all answer "hai").
            Without masking, these are penalised as negatives → noisy gradient.
            → We detect same-answer pairs via token overlap (labels) and mask them
              out of the denominator, treating them as "neither positive nor negative".

        • Temperature τ=0.07 (SimCLR default).

        Args
        ----
        fused_vision : [B, P, D]       — vision features post-Flamingo fusion
        text_cls     : [B, D]          — BARTpho encoder CLS token
        labels       : [B, seq_len]    — token ids of answers (-100 = padding)
                        If provided, enables false-negative masking.

        Returns
        -------
        loss : scalar — symmetric InfoNCE loss (image→text + text→image) / 2
        """
        B = text_cls.size(0)
        device = text_cls.device

        # 1. Mean-pool vision patches → [B, D]
        v = fused_vision.mean(dim=1)  # [B, D]
        t = text_cls                   # [B, D]

        # 2. Project into shared contrastive space [B, 128]
        v = self.vision_contrastive_head(v)   # [B, 128]
        t = self.text_contrastive_head(t)     # [B, 128]

        # 3. L2-normalise (cosine similarity becomes dot product)
        v = F.normalize(v, dim=-1)
        t = F.normalize(t, dim=-1)

        # 4. Similarity matrix [B, B] scaled by temperature
        logits = torch.matmul(v, t.T) / self.contrastive_temp  # [B, B]

        # Clamp to avoid extreme values (BF16/TF32 can overflow faster on H100)
        logits = torch.clamp(logits, min=-100.0, max=100.0)

        # 5. False-negative mask: samples sharing the same answer should NOT
        #    be treated as negatives — mask them out of the CE denominator.
        #    mask[i, j] = True  → pair (i,j) is a false negative → ignore
        #    mask[i, i] = False → diagonal is always the positive (kept)
        fn_mask = None
        if labels is not None:
            # Exact-match false-negative masking: only mask pairs whose full
            # answer token sequences are identical. Any-token-overlap is too
            # aggressive for BartPho syllable tokenisation (common tokens like
            # "có", "một", "hai" appear in most answers and would collapse the
            # effective batch to size 1, making ctr → 0).
            valid_tokens = []
            for lab in labels:
                toks = tuple(lab[lab != -100].cpu().tolist())
                valid_tokens.append(toks)

            # [B, B] bool: True where i≠j AND full answer sequence is identical
            fn_mask = torch.zeros(B, B, dtype=torch.bool, device=device)
            for i in range(B):
                for j in range(B):
                    if i != j and valid_tokens[i] == valid_tokens[j]:
                        fn_mask[i, j] = True

        # 6. Apply false-negative mask by setting those logits to a large negative
        #    value (avoid -inf to prevent NaNs in softmax when a row is all-masked)
        if fn_mask is not None and fn_mask.any():
            # Determine which rows still have at least one valid negative
            row_has_valid = ~fn_mask
            row_has_valid.fill_diagonal_(False)
            valid_rows = row_has_valid.any(dim=1)

            logits = logits.masked_fill(fn_mask, -1e4)

            # 7. Positive targets = diagonal (i, i)
            targets = torch.arange(B, device=device)

            # 8. Symmetric InfoNCE: vision→text + text→image (valid rows only)
            if valid_rows.any():
                loss_v2t = F.cross_entropy(logits[valid_rows], targets[valid_rows])
                loss_t2v = F.cross_entropy(logits.T[valid_rows], targets[valid_rows])
            else:
                return torch.tensor(0.0, device=device)
        else:
            # 7. Positive targets = diagonal (i, i)
            targets = torch.arange(B, device=device)

            # 8. Symmetric InfoNCE: vision→text + text→image
            loss_v2t = F.cross_entropy(logits,   targets)
            loss_t2v = F.cross_entropy(logits.T, targets)

        loss = (loss_v2t + loss_t2v) / 2.0

        # Guard NaN/Inf
        if torch.isnan(loss) or torch.isinf(loss):
            return torch.tensor(0.0, device=device)

        return loss

    def compute_answer_gate_contrastive(
        self,
        gated_vision: torch.Tensor,  # [B, P, D] — POST-gate vision (grad flows to gate_net)
        labels: torch.Tensor,        # [B, seq_len] — answer token ids (-100 = pad)
        alpha: torch.Tensor = None,  # 🔬 [B, P] gate values — neu co, pool CO TRONG SO theo alpha
    ) -> torch.Tensor:
        """
        🔬 GATE-ANSWER CONTRASTIVE (GAC). Giam sat CHINH cai gate.

        Khac contrastive cu (align PRE-gate fused-vision <-> QUESTION, khong cham gate):
          - Anchor = mean(GATED vision), CO gradient ve gate_net.
          - Positive = ANSWER embedding (mean decoder embed cua label tokens).
        Vi target la DAP AN chu khong phai cau hoi, attractor alpha->0 (gated -> question-text)
        KHONG con thoa man contrastive -> gate buoc phai chon patch vision du doan dap an,
        tao dac trung phan biet per-question ma LM loss (da bao hoa tren SigLIP) khong ep duoc.
        Mot chieu vision->answer (cho anh, chon dung dap an trong batch).

        🔬 v2 (2026-08-08): pool trung binh DEU truoc day la tin hieu qua mo -- model co the
        "lach" bang cach lam MOI patch hoi lien quan dap an (giai phap lan toa), khong patch
        nao noi bat, van thoa man contrastive ma khong hoc chon loc. Neu co alpha, pool CO
        TRONG SO theo chinh alpha (Σ alpha_i·v_i / Σ alpha_i) -- gradient contrastive chay
        THANG toi dung alpha dang gan cho tung patch, khong con di vong qua mean deu.
        """
        B = gated_vision.size(0)
        device = gated_vision.device
        # Anchor: pool gated vision (theo alpha neu co, khong thi mean deu) -> [B, D] -> proj -> normalize
        if alpha is not None:
            _a = alpha.unsqueeze(-1) if alpha.dim() == 2 else alpha  # [B,P,1]
            pooled = (gated_vision * _a).sum(dim=1) / _a.sum(dim=1).clamp(min=1e-6)
        else:
            pooled = gated_vision.mean(dim=1)
        v = self.gac_vision_head(pooled)
        v = F.normalize(v, dim=-1)
        # Positive: mean decoder-embedding of answer tokens (detach: chi keo gate/vision ve answer,
        # khong sua bang embedding decoder). valid mask loai -100.
        with torch.no_grad():
            _ids = labels.clamp(min=0)                          # [B, L] (pad/-100 -> 0)
            _emb = self.decoder.embed_tokens(_ids)              # [B, L, D]
            _m = (labels != -100).unsqueeze(-1).to(_emb.dtype)  # [B, L, 1]
            _denom = _m.sum(dim=1).clamp(min=1.0)
            a_raw = (_emb * _m).sum(dim=1) / _denom             # [B, D]
        a = self.gac_answer_head(a_raw.detach())
        a = F.normalize(a, dim=-1)
        logits = torch.matmul(v, a.T) / self.gate_answer_contrastive_temp
        logits = torch.clamp(logits, min=-100.0, max=100.0)
        # False-negative mask: cac sample CUNG dap an khong phai negative (giong contrastive cu)
        valid_tokens = [tuple(lab[lab != -100].cpu().tolist()) for lab in labels]
        fn_mask = torch.zeros(B, B, dtype=torch.bool, device=device)
        for i in range(B):
            for j in range(B):
                if i != j and valid_tokens[i] == valid_tokens[j]:
                    fn_mask[i, j] = True
        targets = torch.arange(B, device=device)
        if fn_mask.any():
            row_has_valid = ~fn_mask
            row_has_valid.fill_diagonal_(False)
            valid_rows = row_has_valid.any(dim=1)
            logits = logits.masked_fill(fn_mask, -1e4)
            if not valid_rows.any():
                return torch.tensor(0.0, device=device)
            loss = F.cross_entropy(logits[valid_rows], targets[valid_rows])
        else:
            loss = F.cross_entropy(logits, targets)
        if torch.isnan(loss) or torch.isinf(loss):
            return torch.tensor(0.0, device=device)
        return loss

    def compute_gate_divergence_loss(
        self,
        gate_values: torch.Tensor,  # [B, P] — alpha per patch from VisionGating
        type_ids: torch.Tensor,     # [B]    — question type per sample
        num_types: int = 4,
    ) -> torch.Tensor:
        """
        Inter-type gate divergence loss.

        Forces VisionGating to produce different alpha distributions for different
        question types. Requires no human assumptions about which type should be
        sparse or dense — the CE loss guides WHAT each type attends to, this loss
        only enforces that types DO attend differently.

        Method: compute per-type centroid of alpha [K, num_patches] → mean → [num_patches],
        then maximize variance across type centroids (= minimize negative variance).

        Needs at least 2 types present in the batch. With PK sampling (P=4), this
        is always satisfied.

        Returns scalar loss (negative inter-type variance).
        """
        centroids = []
        for t in range(num_types):
            mask = (type_ids == t)
            if mask.sum() > 0:
                centroids.append(gate_values[mask].mean(dim=0))  # [num_patches]

        if len(centroids) < 2:
            return torch.tensor(0.0, device=gate_values.device)

        stacked = torch.stack(centroids)           # [T, num_patches]
        inter_type_var = stacked.var(dim=0).mean() # scalar — variance across types per patch
        return -inter_type_var                     # minimize → maximize variance

    def _concat_fuse(self, fused_vision, fused_text, attention_mask):
        """🔬 Fusion RE thay GCA: nhoi cau hoi pooled vao tung patch bang concat + proj.
        v'_i = v_i + tanh(g_cf) * W_cf([v_i ; text_pooled]).  O(P*D^2), re hon GCA attention.
        Muc dich (gca_sweep: TCVG keo lai +3.06 khi GCA=0) -> thay GCA dat bang concat re, de TCVG
        phai GANH -> TCVG load-bearing thay vi du thua. text KHONG doi."""
        m = attention_mask.float().unsqueeze(-1)
        tp = (fused_text * m).sum(1, keepdim=True) / m.sum(1, keepdim=True).clamp(min=1)
        tp = tp.expand(-1, fused_vision.size(1), -1)
        cf = self.cf_proj(torch.cat([fused_vision, tp], dim=-1))
        fused_vision = fused_vision + torch.tanh(self.cf_gate) * cf
        return fused_vision, fused_text

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        stage: int = 3,  # Kept for compatibility, but ignored
        answer_weights: Optional[torch.Tensor] = None,  # Token-level weights for balanced loss
        question_types: Optional[torch.Tensor] = None,  # Question type (0=object_id, 1=counting, 2=color, 3=location)
        images_384: Optional[torch.Tensor] = None,  # For vision teacher (384px)
        raw_questions: Optional[list] = None,  # For text teacher (raw strings)
        sample_weights: Optional[torch.Tensor] = None,  # Per-sample weights [batch_size] for type-conditional weighting
        region_map: Optional[torch.Tensor] = None,  # 🔬 [B, num_patches] chi so COCO-region tung patch (tcvg_spatial_blend), None = hanh vi cu (3x3 window)
        token_weights: Optional[torch.Tensor] = None,  # 🔬 GGE: trong so [B, T] tren tung vi tri nhan. None = hanh vi cu.
    ):
        """
        Forward pass - deterministic fusion
        
        NO sampling, NO KL, just pure cross-attention!
        
        Args:
            answer_weights: [vocab_size] tensor with per-token loss weights (inverse freq)
            question_types: [batch] tensor with question type:
                0 = object identification (Đây là gì?)
                1 = counting (Có bao nhiêu?)
                2 = color (Màu gì?)
                3 = location (Ở đâu? Trên bàn?)
        """
        batch_size = pixel_values.size(0)
        # H3: text-only warmup mode — decoder learns answer patterns before seeing vision
        text_only = getattr(self, 'text_only_mode', False)

        # 1. Vision encoding (skipped during text-only warmup)
        if not text_only:
            # Note: self.vision_encoder is already vision_model component for SigLIP
            # or full DINOv2 model. Both take pixel_values directly.
            _bbl = getattr(self, 'vision_backbone_layer', -1)
            _need_hs = (getattr(self, 'gate_vision_layer', -1) >= 0) or (_bbl >= 0)
            vision_outputs = self.vision_encoder(pixel_values=pixel_values, output_hidden_states=_need_hs)
            # 🔬 vision_backbone_layer: dung L INTERMEDIATE (it language-aligned, giu local/mau) lam
            # feature CHINH cho CA GCA + TCVG + decoder (khac cac L6-flag cu chi nhet L6 vao gate).
            if _bbl >= 0:
                patch_tokens = vision_outputs.hidden_states[_bbl]  # [B, seq, hidden] tai lop L
            else:
                patch_tokens = vision_outputs.last_hidden_state  # [batch, seq_len, hidden]
            # 🔬 dac trung lop trung gian cho gate (local structure). Lay [L], bo CLS, chieu.
            self._gate_alpha_feats = None
            if getattr(self, 'gate_vision_layer', -1) >= 0:
                _hl = vision_outputs.hidden_states[self.gate_vision_layer]
                if _hl.size(1) > self.num_patches: _hl = _hl[:, 1:, :]
                self._gate_alpha_feats = self.gate_layer_proj(_hl)  # [B, P, bart_hidden]

            # Remove CLS token if present
            # SigLIP vision_model: [batch, 197, hidden_dim] → 196 patches + 1 CLS
            # DINOv2: [batch, 197, hidden_dim] → 196 patches + 1 CLS
            # We only need patch tokens for cross-attention fusion
            original_seq_len = patch_tokens.size(1)
            if original_seq_len > self.num_patches:  # Has CLS token
                patch_tokens = patch_tokens[:, 1:, :]  # Remove first token (CLS)
                assert patch_tokens.size(1) == self.num_patches, \
                    f"Shape mismatch after CLS removal: got {patch_tokens.size(1)} patches, expected {self.num_patches}"

            # 🔥 NEW: Apply type-conditioned adapter (BEFORE position embeddings)
            if self.vision_adapter is not None:
                patch_tokens = self.vision_adapter(
                    patch_tokens,
                    type_ids=question_types
                )

            # Add position embeddings
            patch_tokens = patch_tokens + self.vision_pos_embed.expand(batch_size, -1, -1)
            raw_patch_tokens = patch_tokens  # [B, 196, vision_hidden] — used for KD only
            vision_features = self.vision_proj(patch_tokens)  # [batch, 196, bart_hidden]
            vision_features = self._enrich_l6(vision_features)  # 🔬 PROBE: +learned(L6), no-op neu tat

            # Optionally prepend SigLIP pooler_output as a global vision token
            if self.use_siglip_pooler and hasattr(vision_outputs, 'pooler_output') and vision_outputs.pooler_output is not None:
                global_feat = self.siglip_global_proj(vision_outputs.pooler_output)  # [B, bart_hidden]
                vision_features = torch.cat([global_feat.unsqueeze(1), vision_features], dim=1)  # [B, 197, bart_hidden]
                # align _gate_alpha_feats voi patch count (them global token dau)
                if self._gate_alpha_feats is not None:
                    self._gate_alpha_feats = torch.cat([global_feat.unsqueeze(1), self._gate_alpha_feats], dim=1)
        else:
            raw_patch_tokens = None
            vision_features = None  # set after text encoding (need bart_hidden_dim)

        # 2. Text encoding
        text_encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        text_features = text_encoder_outputs.last_hidden_state

        if text_only:
            # Zero vision: decoder learns to generate answers from question context alone
            vision_features = torch.zeros(
                batch_size, self.num_patches, text_features.size(-1),
                device=text_features.device, dtype=text_features.dtype
            )

        # text_cls: attention pool (learned) > BOS (default) > mean-pool (worst)
        if self.use_attn_pool_cls:
            text_cls = self.attn_pool(text_features, attention_mask)  # [B, D]
        elif self.use_mean_pool_cls:
            _mask = attention_mask.float().unsqueeze(-1)  # [B, L, 1]
            text_cls = (text_features * _mask).sum(1) / _mask.sum(1).clamp(min=1)  # [B, D]
        else:
            text_cls = text_features[:, 0, :]  # [B, D]
        type_logits = None
        type_loss = None

        # 🔬 CODEBOOK KHONG GIAM SAT: cum duoc phat hien tu t_cls thay cho nhan loai.
        # cb_idx dong vai tro type_ids (cho type_bias / ctx_gate / logits_bias),
        # cb_emb dong vai tro e_type trong q = W_q[t_cls; e_type] (dung §5 paper).
        cb_emb = cb_idx = None
        vq_loss = None
        if self.type_codebook is not None:
            cb_emb, cb_idx, vq_loss, _ppl = self.type_codebook(text_cls)
            self.last_codebook_idx = cb_idx.detach()
            self.last_codebook_ppl = float(_ppl)

        self._type_vec = None
        if self.use_type_task:
            if getattr(self, 'type_branch', None) is not None:
                # 🔬 type_loss qua nhanh detach -> khong nhieu encoder; type_vec feed gate
                self._type_vec = self.type_branch(text_cls.detach())  # [B, D]
                type_logits = self.type_head(self._type_vec)          # [B, 4]
            else:
                type_logits = self.type_head(text_cls)  # [B, 4]
            if question_types is not None:
                type_weights = getattr(self, 'type_class_weights', None)
                type_loss = F.cross_entropy(type_logits, question_types, weight=type_weights)

        # 3. Vision-text fusion (Flamingo GCA, text-to-vision) — skipped in text-only warmup
        if not text_only:
            # GCA injects question semantics into each patch → v(L)_i becomes question-conditioned.
            vision_features_pre_flamingo = vision_features  # [B, 196, D] — pre-GCA, for delta gate
            fused_vision = vision_features
            fused_text = text_features

            # 🔥 tcvg_fusion_gate: TCVG dieu khien CUONG DO HOP NHAT cua GCA theo tung patch
            # thay vi tron hau ky sau GCA.
            #
            # 2 LUOT (tcvg_fg_2pass, khuyen dung): giu DUNG Eq.4 cua paper — alpha van duoc tinh
            # tu v^(L) da dieu kien hoa theo cau hoi, nen gate van chon loc theo TUNG CAU HOI
            # (khong tut ve gating theo loai).
            #   Luot 1: GCA binh thuong -> v^(L)   ->  alpha = sigma(g([v^(L); W_q[t_cls; e_type]]))
            #   Luot 2: GCA lai TU VISION THO, residual nhan alpha
            # Luot 1 chi de QUYET DINH nhin dau, luot 2 moi thuc su hop nhat.
            #
            # 1 LUOT: alpha tinh tu vision THO — re hon nhung gate phai tu khop patch voi cau hoi
            # bang MLP 2 tang thay vi huong san ket qua khop cua multi-head attention.
            _fuse_alpha = None
            if getattr(self, 'tcvg_fusion_gate', False) and self.use_vision_gate:
                if question_types is not None:
                    _tid = question_types
                elif type_logits is not None:
                    _tid = torch.argmax(type_logits, dim=-1)
                else:
                    _tid = None
                if getattr(self, 'tcvg_fg_2pass', False):
                    _v1, _t1 = vision_features, text_features
                    for _fl in self.flamingo_fusion:
                        _v1, _t1 = _fl(_v1, _t1, attention_mask)
                    _fuse_alpha = self.vision_gating.compute_alpha(
                        _v1, _t1, type_ids=_tid, text_attention_mask=attention_mask)
                else:
                    _fuse_alpha = self.vision_gating.compute_alpha(
                        vision_features, text_features, type_ids=_tid,
                        text_attention_mask=attention_mask)

            _gca_attn_weights = None  # 🔬 HYP #1: trong so attention cua lop fusion CUOI CUNG
            if getattr(self, 'concat_fusion', False):
                fused_vision, fused_text = self._concat_fuse(fused_vision, fused_text, attention_mask)
            else:
                _bt = _btm = None
                if getattr(self, 'gca_box_tokens', False) and region_map is not None:
                    _bt, _btm = self._box_tokens_from_region_map(region_map, fused_vision.dtype)
                # LUON dat lai moi forward. Neu chi dat trong nhanh train thi gia tri 0.0 cua
                # batch train CUOI se con nguyen luc eval -> GCA tat khi suy luan. Dung lop loi
                # train/eval phan ky da tung lam hong tcvg_topk_random.
                _gsb = self.gca_strength
                _dt = getattr(self, 'gca_dropout_types', None)
                if self.training and getattr(self, 'gca_dropout', 0.0) > 0:
                    if _dt is None:
                        # che do TOAN BATCH (nhu lan chay dau)
                        if torch.rand(()).item() < self.gca_dropout:
                            _gsb = 0.0
                    elif question_types is not None:
                        # 🔬 THEO LOAI: chi tat GCA cho mau thuoc cac loai duoc chi dinh.
                        #   Do duoc o lan chay toan-batch (seed 0): bat cong o ba loai von TAT
                        #   (OBJECT/LOCATION alpha 0.999 -> 0.52/0.74) cho COUNT +4.05,
                        #   LOCATION +2.63, OBJECT +0.00 — nhung COLOR, loai DUY NHAT von da
                        #   hoat dong (alpha 0.4417), mat -4.48 va an sach phan tang.
                        #   => chi ap cho loai co alpha ghim o 1, tha COLOR ra.
                        _B = fused_vision.size(0)
                        _sel = torch.zeros(_B, device=fused_vision.device, dtype=torch.bool)
                        for _t in _dt:
                            _sel |= (question_types == _t)
                        _drop = _sel & (torch.rand(_B, device=fused_vision.device) < self.gca_dropout)
                        _gsb = torch.where(_drop,
                                           torch.zeros(_B, device=fused_vision.device),
                                           torch.full((_B,), float(self.gca_strength),
                                                      device=fused_vision.device)).view(-1, 1, 1)
                for _fl in self.flamingo_fusion:
                    _fl.gca_strength = _gsb
                self._last_gsb = _gsb
                for _li, fusion_layer in enumerate(self.flamingo_fusion):
                    fused_vision, fused_text = fusion_layer(
                        fused_vision, fused_text, attention_mask, residual_scale=_fuse_alpha,
                        extra_kv=_bt, extra_kv_mask=_btm)
                    # 🔬 TCVG lop giua: chon loc ngay sau lop GCA dau, truoc khi lop GCA sau
                    # doc lai. Dung question_types (co san truoc vong lap); gate xu ly None duoc.
                    if (self.vision_gating_mid is not None
                            and _li == 0 and len(self.flamingo_fusion) > 1):
                        fused_vision, _ = self.vision_gating_mid(
                            fused_vision, fused_text,
                            type_ids=question_types,
                            text_attention_mask=attention_mask)
                if getattr(self, 'tcvg_alpha_from_gca', False) and self.fusion_type == 'text2vision':
                    _gca_attn_weights = getattr(self.flamingo_fusion[-1], 'last_attn_weights', None)

            if self.fusion_type in ('text2vision', 'vision2text', 'bidirectional'):
                vision_for_decoder = self._apply_psa(fused_vision)   # 🔬 patch <-> patch
                text_for_concat = fused_text
            else:
                raise ValueError(f"Unknown fusion_type: {self.fusion_type}")

            # Type-specific adapter: modify ONLY BOS/CLS position [B,D] post-Flamingo.
            # VisionGating uses t_proj[:,0,:] as text summary for gate query → this makes gating type-aware.
            # Decoder cross-attends to all positions; touching only pos-0 is far less disruptive than full sequence.
            # WARNING: never apply this to full text_features — disrupts decoder cross-attention (run83 failure).
            if self.type_text_adapter is not None and question_types is not None:
                _cls = self.type_text_adapter(text_for_concat[:, 0, :], question_types)
                text_for_concat = torch.cat([_cls.unsqueeze(1), text_for_concat[:, 1:, :]], dim=1)
        else:
            vision_features_pre_flamingo = vision_features
            vision_for_decoder = vision_features  # zeros — decoder trains on text context only
            text_for_concat = text_features

        # 4. Type-Conditioned Vision Gating (AFTER Flamingo — question-conditioned features)
        # Gate input v(L)_i already encodes "patch i as understood in context of this question".
        # Combined with q = Wq[t_cls; e_type], gate achieves instance-level type conditioning:
        # same type but different questions → different patches selected.
        # vision_prefgate used for contrastive: pre-gating Flamingo output has no α-collapse
        # attractor (gated_vision = α·v + (1-α)·t̄ would trivially satisfy contrastive at α→0).
        vision_prefgate = vision_for_decoder if not text_only else None
        self._gac_gated = None   # 🔬 reset moi forward; chi set khi gate branch chay
        self._gac_alpha = None   # 🔬 [B,P] alpha tuong ung, dung cho weighted pooling trong GAC
        # 🔬 dynamic_peek v2 (2026-08-10): SUA lech train/eval cua v1 (v1 dung nhan THAT luc
        # train -- tin hieu gan hoan hao -- nhung dung DRAFT tu sinh (co the sai) luc eval ->
        # gate hoc qua tin tin hieu hoan hao, lech canh chinh khi gap tin hieu nhieu that. v2:
        # luc train CUNG TU SINH DRAFT that (voi trong so HIEN TAI, khong gradient) giong het
        # dieu kien eval -- khong con "an gian" bang nhan. Tон phi: 1 luot generate greedy them
        # moi batch (cham hon dang ke, nhung dung dieu kien that, va draft tu nhien noisy luc
        # dau train roi tot dan theo model -- curriculum tu nhien, khong can lich trinh rieng).
        peek_embedding = None
        if getattr(self, 'tcvg_dynamic_peek', False) and self.training and labels is not None:
            with torch.no_grad():
                self.eval()
                try:
                    _draft_texts = self.generate(
                        pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask,
                        max_length=labels.size(1), num_beams=1,
                        peek_embedding=torch.zeros(pixel_values.size(0), self.decoder.config.d_model,
                                                     device=pixel_values.device, dtype=self.decoder.embed_tokens.weight.dtype),
                    )
                finally:
                    self.train()
                _draft_enc = self.tokenizer(_draft_texts, truncation=True, padding=True,
                                             max_length=max(labels.size(1), 1), return_tensors='pt').to(pixel_values.device)
                _draft_emb = self.decoder.embed_tokens(_draft_enc['input_ids'])
                _draft_mask = _draft_enc['attention_mask'].unsqueeze(-1).float()
                peek_embedding = (_draft_emb * _draft_mask).sum(dim=1) / _draft_mask.sum(dim=1).clamp(min=1.0)
        elif getattr(self, 'tcvg_dynamic_peek', False) and not self.training and labels is not None:
            # (khong dung o day -- forward() luc eval thuong khong duoc goi voi mucdich generate;
            # giu nhanh nay chi de tuong thich neu co noi khac goi forward() luc eval co labels)
            with torch.no_grad():
                _lbl_ids = labels.clamp(min=0)
                _lbl_emb = self.decoder.embed_tokens(_lbl_ids)
                _lbl_mask = (labels != -100).unsqueeze(-1).to(_lbl_emb.dtype)
                peek_embedding = (_lbl_emb * _lbl_mask).sum(dim=1) / _lbl_mask.sum(dim=1).clamp(min=1.0)
        gate_stats = None
        if getattr(self, 'tcvg_fusion_gate', False) and _fuse_alpha is not None:
            # Da dung alpha o residual GCA -> KHONG tron hau ky nua (tranh gating hai lan)
            gate_stats = {'mean': float(_fuse_alpha.mean()), 'std': float(_fuse_alpha.std()),
                          'min': float(_fuse_alpha.min()), 'max': float(_fuse_alpha.max())}
        elif self.use_vision_gate and not text_only:
            if cb_idx is not None:
                # codebook -> KHONG dung nhan loai, ke ca khi dataloader co cung cap
                type_ids_for_gating = cb_idx
            elif question_types is not None:
                type_ids_for_gating = question_types
            elif type_logits is not None:
                type_ids_for_gating = torch.argmax(type_logits, dim=-1)
            else:
                type_ids_for_gating = None
            # 🔬 gate_type_blind: ablation "TCVG KHONG dieu kien hoa theo loai".
            # Bat buoc phai ep o DAY, vi nhanh train lay question_types tu dataloader bat ke
            # use_type_task bat hay tat, con nhanh generate lai chi co predicted_types khi CO
            # type head -> bo cờ --use_type_loss mot minh se tao LECH train/test (train co dieu
            # kien theo loai, test dung gia tri trung binh) va cho ra mot con so xau gia tao.
            if getattr(self, 'gate_type_blind', False):
                type_ids_for_gating = None
            # 🔬 type_branch: dung type_vec (detach) lam e_type cho gate; neu khong thi cb_emb
            _gate_type_ovr = self._type_vec if getattr(self, '_type_vec', None) is not None else cb_emb
            # 🔬 KIEN TRUC HAI TANG (slot_stage='pre_gated'): slot TONG HOP ung vien tu feature
            # NGUYEN VEN (truoc gate), noi vao tap token, roi TCVG CHON tren ca 197+K token.
            #   tang 1 (cong)  : slot sinh K token dai dien "co may thu" -- phep bien doi KHONG LOI,
            #                    nam ngoai span cua 196 patch, la thu decoder-1-luot khong lam duoc.
            #   tang 2 (chon)  : TCVG quyet dinh theo LOAI nen giu gi -- gio no co noi dung NGOAI SPAN
            #                    de chon, thay vi chi chon lai trong dung tap token decoder da attend.
            # Day la ly do duy nhat khien gate co the khong con du thua: khong phai vi alpha khon hon,
            # ma vi TAP UNG VIEN da khac. (Canh bao: attention tren 197+K VAN la to hop loi, nen ve
            # ly thuyet gate van bi bao trum -- prior thap. Thi nghiem nay de KIEM, khong de gia dinh.)
            if (self.slot_attn is not None and getattr(self, 'slot_stage', 'post') == 'pre_gated'):
                _sltid_pg = type_ids_for_gating
                _slots_pg = self.slot_attn(vision_for_decoder, text_for_concat[:, 0, :], _sltid_pg)
                self._slots_in_gate = _slots_pg.size(1)   # co: da noi slot -> KHONG noi lai o buoc sau
                vision_for_decoder = torch.cat([vision_for_decoder, _slots_pg], dim=1)
            gated_vision, gate_values = self.vision_gating(
                vision_for_decoder,
                text_for_concat,
                type_ids=type_ids_for_gating,
                type_emb_override=_gate_type_ovr,   # 🔬 type_vec (detach branch) hoac codebook e_type
                alpha_feats=getattr(self,'_gate_alpha_feats',None),
                text_attention_mask=attention_mask,
                detach_for_gate=self.gate_detach_input,
                peek_embedding=peek_embedding,
                vision_orig=(vision_features_pre_flamingo
                             if (self.use_delta_gate or getattr(self, 'gate_blend_vorig', False)
                                 or getattr(self, 'gate_gca_residual', False)) else None),
                region_map=region_map,
                gca_attn_weights=_gca_attn_weights,
            )
            gate_stats = {
                'mean': gate_values.mean().item(),
                'std': gate_values.std().item(),
                'min': gate_values.min().item(),
                'max': gate_values.max().item()
            }
            vision_for_decoder = gated_vision
            # 🔬 --type_from_gate_lambda: CE loai cau hoi doc tu THI GIAC SAU GATE.
            # Gradient di qua gated_vision -> alpha -> gate_net/query_proj, tuc type_loss lan nay
            # ep GATE phai tao ra bieu dien phan biet duoc loai, thay vi chi ep text_cls (noi nhan
            # da nam san). Cong vao type_loss de train.py khong can sua gi.
            if (getattr(self, 'type_head_gate', None) is not None
                    and question_types is not None):
                _tg_logits = self.type_head_gate(gated_vision.mean(dim=1))   # [B, 4]
                self.last_type_gate_logits = _tg_logits.detach()
                _tg_loss = F.cross_entropy(_tg_logits, question_types)
                type_loss = (_tg_loss * self.type_from_gate_lambda if type_loss is None
                             else type_loss + self.type_from_gate_lambda * _tg_loss)
            # 🔬 slot_stage=pre: GIU feature TRUOC gate de slot tong hop tu ban chua bi nen.
            # Ly do: gate nen 35.8% patch o COUNT / 30.0% o COLOR (do tu alpha_oracle), tuc dung
            # nhung patch co the can de TACH vat the bi tron ve text_pooled TRUOC khi slot kip
            # tong hop. Thu tu cu (slot doc feature da nen) lam slot phai ca the hoa tu ban suy giam,
            # o dung hai loai ta muon no hoat dong. KHONG vi pham rang buoc "TCVG sau Flamingo".
            self._slot_src_pre = fused_vision
            self._gac_gated = gated_vision   # 🔬 post-gate feature for gate-answer contrastive (grad -> gate)
            self._gac_alpha = gate_values    # 🔬 [B,P] KHONG detach -- GAC pool co trong so theo alpha,
                                              #    de gradient noi thang toi tung patch alpha thay vi loang
                                              #    qua mean deu (xem compute_answer_gate_contrastive)
            _moe_probs = F.softmax(type_logits, dim=-1) if type_logits is not None else None
            vision_for_decoder = self._apply_type_moe(vision_for_decoder, type_ids_for_gating, type_probs=_moe_probs)

        # 🔬 dau phan loai dap an: pool vision (sau gate) + text_cls -> softmax tren tap dap an
        answer_cls_logits = None
        if getattr(self, 'num_answer_classes', 0) > 0:
            _av = vision_for_decoder.mean(dim=1)                      # [B, D]
            answer_cls_logits = self.answer_head(torch.cat([_av, text_cls], dim=-1))

        # 🔬 dau phu box-grounded: doc feature decoder THAT SU doc, de gradient nan dung cho do.
        #   Luon tinh khi bat co (khong phu thuoc region_map) — loss ben train.py moi can nhan.
        if getattr(self, 'box_class_n', 0) > 0:
            self.last_class_logit = self.box_class_head(vision_for_decoder)   # [B,P,n_class]
        if getattr(self, 'box_ground', False):
            self.last_ground_logit = self.box_ground_head(vision_for_decoder).squeeze(-1)  # [B,P]
            self.last_count_pred = self.box_count_head(vision_for_decoder.mean(dim=1)).squeeze(-1)  # [B]
        
        # 4. Prepare decoder inputs
        if labels is not None:
            decoder_input_ids = shift_tokens_right(
                labels, 
                self.config.pad_token_id, 
                self.config.decoder_start_token_id
            )
        else:
            decoder_input_ids = torch.full(
                (batch_size, 1),
                self.config.decoder_start_token_id,
                dtype=torch.long,
                device=pixel_values.device
            )
        
        # 5. Decoder: Cross-attend to fused features
        # 🔥 TCVG top-k: XOA HAN cac patch co alpha thap khoi chuoi dua vao decoder.
        #
        # Ly do: alpha chi DANH TRONG SO thi trung chuc nang voi cross-attention cua decoder
        # (do duoc: T2 - T0 = +0.24, khong y nghia; moi can thiep len module deu trung tinh).
        # Khi XOA token, attention KHONG THE lay lai -> gating co nang luc nam NGOAI khong
        # gian ham cua attention, va hai vai tro tach bach:
        #   GCA  = nhet noi dung cau hoi vao patch
        #   TCVG = quyet dinh patch nao duoc di tiep
        # Khac voi gate_mode=multiply (da that bai): o day KHONG doi bien do token, van giu
        # blend + layer_norm, chi cat so luong token.
        _topk = getattr(self, 'tcvg_topk', 0)
        if _topk and _topk > 0 and self.use_vision_gate and 'gate_values' in dir() and gate_values is not None:
            _k = min(int(_topk), vision_for_decoder.size(1))
            _a = gate_values
            if _a.dim() == 3:
                _a = _a.squeeze(-1)                      # [B, P]
            _npad = vision_for_decoder.size(1) - _a.size(1)
            if _npad > 0:                                # token global (siglip pooler) luon giu
                _a = torch.cat([_a.new_full((_a.size(0), _npad), 1e9), _a], dim=1)
            if getattr(self, 'tcvg_topk_random', False):
                # DOI CHUNG: chon k patch NGAU NHIEN thay vi top-alpha. Cung kien truc, cung
                # ngan sach token, khac duy nhat o viec gate co duoc dung de chon hay khong.
                # Neu gate > random co y nghia -> TCVG THUC SU biet nhin dau.
                _rnd = torch.rand_like(_a)
                if _npad > 0:
                    _rnd[:, :_npad] = 1e9      # token global luon giu
                _idx = _rnd.topk(_k, dim=1).indices.sort(dim=1).values
            else:
                _idx = _a.topk(_k, dim=1).indices.sort(dim=1).values   # giu thu tu khong gian
            vision_for_decoder = torch.gather(
                vision_for_decoder, 1,
                _idx.unsqueeze(-1).expand(-1, -1, vision_for_decoder.size(-1)))

        # 🔬 decoder_pool_vision=K: gop 197 token thi giac xuong K token bang average pooling.
        #
        # LY DO CO CHE: cross-attention cua decoder VON DA tu danh trong so tren 197 token theo
        # tung query, nen viec TCVG danh trong so tung patch la DU THUA — T0 dat cung ket qua
        # mien phi qua decoder, va alpha=1 tro thanh nghiem toi uu hop le (T2-T0 = +0.24 n.s.).
        # Sau khi pool ve K token, thong tin theo tung patch KHONG CON, nen decoder khong the
        # chon loc nua: moi su chon loc BUOC PHAI xay ra trong TCVG.
        # Day la phep do CHAN DOAN (kien truc bi sua co y), khong phai mo hinh de xuat.
        _pv = getattr(self, 'decoder_pool_vision', 0)
        if _pv and _pv > 0 and vision_for_decoder is not None and vision_for_decoder.size(1) > int(_pv):
            vision_for_decoder = F.adaptive_avg_pool1d(
                vision_for_decoder.transpose(1, 2), int(_pv)).transpose(1, 2)

        # 🔥 Concatenate vision + text for decoder cross-attention
        # Order matters: vision first = decoder sees vision tokens first
        # 🔥 decoder_vision_only: bo text khoi encoder_hidden_states.
        # Mac dinh decoder nhin CA vision LAN text ghep lai, nen no tu thuc hien duoc
        # chon loc patch co dieu kien theo cau hoi -> TCVG du thua (do duoc: T2-T0 = +0.24).
        # Khi chi con vision, TCVG la co che chon loc DUY NHAT -> co lap duoc dong gop that.
        # 🔬 H1 text-path dropout: KHI TRAIN, ngau nhien che text tokens khoi cross-attention
        # cua decoder voi xac suat p (theo TUNG MAU). Buoc duong vision+TCVG dung vung mot minh
        # o p% so buoc -> TCVG phai mang thong tin loai vi decoder khong luon co text de di vong.
        # Inference: giu ca hai (khong che). Khac decoder_vision_only (bo VINH VIEN -> hai -0.48);
        # day la REGULARIZATION dung vao duong du thua, giu nguyen inference.
        _tpd = getattr(self, 'text_path_dropout', 0.0)
        _drop_text = None
        if self.training and _tpd > 0.0:
            _drop_text = (torch.rand(batch_size, device=attention_mask.device) < _tpd)  # [B]

        if getattr(self, 'decoder_vision_only', False):
            encoder_hidden_states = vision_for_decoder
            encoder_attention_mask = torch.ones(
                batch_size, vision_for_decoder.size(1), device=attention_mask.device)
        else:
            encoder_hidden_states = torch.cat([vision_for_decoder, text_for_concat], dim=1)
            _vmask = torch.ones(batch_size, vision_for_decoder.size(1), device=attention_mask.device)
            _tmask = attention_mask
            if _drop_text is not None:
                # mau bi che: mask text = 0 (decoder khong attend vao text tokens cua mau do)
                _tmask = _tmask * (~_drop_text).unsqueeze(1).to(_tmask.dtype)
            encoder_attention_mask = torch.cat([_vmask, _tmask], dim=1)

        # 🔬 SummaryToken: THEM token tom tat theo loai vao cuoi chuoi (patch KHONG doi).
        if self.summary_token is not None:
            _stid = question_types if question_types is not None else (
                torch.argmax(type_logits, dim=-1) if type_logits is not None else None)
            _stok = self.summary_token(vision_for_decoder, text_cls, _stid)   # [B,1,D]
            encoder_hidden_states = torch.cat([encoder_hidden_states, _stok], dim=1)
            encoder_attention_mask = torch.cat(
                [encoder_attention_mask, torch.ones(batch_size, 1, device=encoder_attention_mask.device)], dim=1)
        _slots = None   # phai khoi tao NGOAI nhanh: khong co slot_attn thi kiem tra duoi van chay
        if self.slot_attn is not None:
            _sltid = question_types if question_types is not None else (
                torch.argmax(type_logits, dim=-1) if type_logits is not None else None)
            if getattr(self, '_slots_in_gate', 0):
                # pre_gated: slot DA nam trong encoder_hidden_states (di qua gate cung patch).
                # Noi lai o day se lam decoder thay slot HAI LAN -> lech phan phoi. Bo qua.
                self._slots_in_gate = 0
            else:
                _slot_src = (getattr(self, '_slot_src_pre', None)
                             if getattr(self, 'slot_stage', 'post') == 'pre' else None)
                if _slot_src is None:
                    _slot_src = vision_for_decoder
                _slots = self.slot_attn(_slot_src, text_cls, _sltid)          # [B,K,D]
        if _slots is not None:
            encoder_hidden_states = torch.cat([encoder_hidden_states, _slots], dim=1)
            encoder_attention_mask = torch.cat(
                [encoder_attention_mask, torch.ones(batch_size, _slots.size(1), device=encoder_attention_mask.device)], dim=1)

        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=None,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask
        )
        
        # 6. Generate answer logits
        base_answer_logits = self.lm_head(decoder_outputs.last_hidden_state)

        # 🔬 KL ve BARTpho pretrained tren cac vi tri token DAP AN.
        #   Chuan hoa theo CE: lambda_eff = lambda * CE.detach()/KL.detach() -> ty trong KL trong
        #   tong loss GIU NGUYEN suot qua trinh train. Bat buoc, vi do duoc: CE roi 15.46 -> 0.088
        #   (176x) trong khi KL gan nhu dung yen (11.77 -> 8.41), nen mot lambda CO DINH se chuyen
        #   tu 0.8% len 48.9% ty trong va nuot luon nhiem vu.
        kl_pretrained_loss = None
        if self.kl_pretrained_lambda > 0 and labels is not None and self._ref_decoder is not None:
            with torch.no_grad():
                _rd = self._ref_decoder(input_ids=decoder_input_ids, attention_mask=None,
                                        encoder_hidden_states=encoder_hidden_states,
                                        encoder_attention_mask=encoder_attention_mask)
                _rl = self._ref_lm_head(_rd.last_hidden_state).float()
            _m = (labels != -100)
            if _m.any():
                _lp = F.log_softmax(base_answer_logits.float(), dim=-1)
                _lr = F.log_softmax(_rl, dim=-1)
                kl_pretrained_loss = (_lp.exp() * (_lp - _lr)).sum(-1)[_m].mean()
        
        # 🔥 Apply type-aware logits biasing (soft vocab conditioning)
        # Needs type_ids: use ground-truth (training) or type_head prediction (inference)
        if self.use_logits_bias and self.logits_bias is not None:
            if question_types is not None:
                answer_logits = self.logits_bias(base_answer_logits, question_types)
            elif type_logits is not None:
                # Inference: use predicted types from type_head
                predicted_types = torch.argmax(type_logits, dim=-1)
                answer_logits = self.logits_bias(base_answer_logits, predicted_types)
            else:
                # No type info available (type_head disabled) → skip bias
                answer_logits = base_answer_logits
        else:
            answer_logits = base_answer_logits
        
        # 7. 🔥 MULTI-TASK LOSS: CE + Type + Contrastive + Gate Divergence
        answer_loss = None
        total_loss = None
        vision_kd_loss = None
        text_kd_loss = None
        contrastive_loss = None
        divergence_loss = None

        if labels is not None:
            # (A) Answer generation loss (weighted CE + label smoothing)
            if answer_weights is not None:
                answer_weights = answer_weights.clamp(min=1e-6, max=100.0)

            if (self.type_label_smoothing is not None
                    and question_types is not None
                    and self.focal_gamma == 0):
                # Per-type label smoothing: different ε per question type.
                # Rationale: COUNT needs ε=0 (off-by-one errors dominate, need precision),
                # COLOR needs ε=0.05 (subtle distinctions), LOCATION/OBJECT keep ε=0.1.
                # Manual LS formula (matches PyTorch):
                #   L = (1-ε)*CE_hard + ε*(-mean(log_softmax))
                B, T = labels.size()
                V = answer_logits.size(-1)
                logits_flat = answer_logits.view(-1, V)   # [B*T, V]
                labels_flat = labels.view(-1)              # [B*T]
                valid_mask  = (labels_flat != -100)

                # Hard CE per token (no smoothing)
                ce_hard = F.cross_entropy(
                    logits_flat, labels_flat,
                    ignore_index=-100,
                    weight=answer_weights if answer_weights is not None else None,
                    reduction='none',
                )  # [B*T]

                # Uniform cross-entropy: -mean(log_softmax) per token
                log_probs  = F.log_softmax(logits_flat, dim=-1)  # [B*T, V]
                ce_uniform = -log_probs.mean(dim=-1)              # [B*T]
                ce_uniform[~valid_mask] = 0.0

                # Build per-sample epsilon, expand to per-token
                eps_sample = torch.tensor(
                    [self.type_label_smoothing.get(int(t), self.label_smoothing)
                     for t in question_types],
                    device=labels.device, dtype=answer_logits.dtype,
                )  # [B]
                eps_tok = eps_sample.unsqueeze(1).expand(B, T).reshape(-1)  # [B*T]

                loss_per_tok = (1.0 - eps_tok) * ce_hard + eps_tok * ce_uniform
                n_valid  = valid_mask.float().sum().clamp(min=1)
                answer_loss = loss_per_tok[valid_mask].sum() / n_valid

            elif self.focal_gamma > 0:
                # Focal loss: FL(p_t) = (1-p_t)^γ * CE(p_t)
                # Down-weights easy examples (high p_t), relatively up-weights hard ones.
                # p_t computed from raw logits (before label smoothing) for accurate weighting.
                logits_flat = answer_logits.view(-1, answer_logits.size(-1))
                labels_flat = labels.view(-1)
                valid_mask = (labels_flat != -100)

                # CE per token with label smoothing (for gradient quality)
                ce_per_tok = F.cross_entropy(
                    logits_flat, labels_flat,
                    ignore_index=-100,
                    weight=answer_weights if answer_weights is not None else None,
                    label_smoothing=self.label_smoothing,
                    reduction='none',
                )

                # p_t: model probability for the true class (no label smoothing)
                with torch.no_grad():
                    labels_safe = labels_flat.clone()
                    labels_safe[~valid_mask] = 0  # avoid gather on -100
                    log_pt = F.log_softmax(logits_flat, dim=-1).gather(
                        1, labels_safe.unsqueeze(1)
                    ).squeeze(1)
                    pt = log_pt.exp().clamp(min=1e-6, max=1.0 - 1e-6)
                    focal_weight = (1.0 - pt) ** self.focal_gamma
                    focal_weight[~valid_mask] = 0.0  # zero out ignored positions

                focal_loss = focal_weight * ce_per_tok
                n_valid = valid_mask.float().sum().clamp(min=1)
                answer_loss = focal_loss.sum() / n_valid

            elif sample_weights is not None:
                # Sample-level type-conditional weighting
                loss_per_tok = F.cross_entropy(
                    answer_logits.view(-1, answer_logits.size(-1)),
                    labels.view(-1),
                    ignore_index=-100,
                    weight=answer_weights if answer_weights is not None else None,
                    label_smoothing=self.label_smoothing,
                    reduction='none',
                )
                bs, seq = labels.size()
                loss_per_tok = loss_per_tok.view(bs, seq)
                valid = (labels != -100).float()
                seq_len = valid.sum(dim=1).clamp(min=1)
                per_sample_loss = (loss_per_tok * valid).sum(dim=1) / seq_len
                sw = sample_weights.to(per_sample_loss.dtype)
                answer_loss = (per_sample_loss * sw).sum() / sw.sum()
            elif token_weights is not None:
                # 🔬 GGE (Han et al., ICCV 2021): trong so tung vi tri nhan, chuan hoa bang TONG
                # TRONG SO. Tinh chat bat buoc: token_weights = 1 o moi vi tri hop le thi bieu thuc
                # nay TRUNG KHOP TUYET DOI voi nhanh CE thuong ben duoi (cung la trung binh theo
                # token). Nho vay doi chung cua GGE la CHINH baseline, khong phai mot nhanh loss
                # khac -> khong lan confound do dai chuoi vao phep so.
                # Dat o mure TOKEN chu khong phai sample_weights vi nhanh sample_weights chuan hoa
                # theo so MAU, doi nhanh se doi luon cach chuan hoa.
                _lpt = F.cross_entropy(
                    answer_logits.view(-1, answer_logits.size(-1)),
                    labels.view(-1),
                    ignore_index=-100,
                    weight=answer_weights if answer_weights is not None else None,
                    label_smoothing=self.label_smoothing,
                    reduction='none',
                ).view(labels.size())
                _valid = (labels != -100).to(_lpt.dtype)
                _w = token_weights.to(_lpt.dtype) * _valid
                # BAY: voi weight=answer_weights, F.cross_entropy(reduction='mean') chuan hoa bang
                # TONG TRONG SO LOP cua cac nhan, KHONG phai so token. Neu chia cho so token thi
                # token_weights=1 se KHONG trung baseline khi co --answer_weights (recipe co dung).
                if answer_weights is not None:
                    _cw = answer_weights.to(_lpt.dtype)[labels.clamp(min=0)] * _valid
                else:
                    _cw = _valid
                answer_loss = (_lpt * _w).sum() / (_cw * token_weights.to(_lpt.dtype)).sum().clamp(min=1e-6)
            else:
                answer_loss = F.cross_entropy(
                    answer_logits.view(-1, answer_logits.size(-1)),
                    labels.view(-1),
                    ignore_index=-100,
                    weight=answer_weights if answer_weights is not None else None,
                    label_smoothing=self.label_smoothing
                )

            if torch.isnan(answer_loss) or torch.isinf(answer_loss):
                answer_loss = F.cross_entropy(
                    answer_logits.view(-1, answer_logits.size(-1)),
                    labels.view(-1),
                    ignore_index=-100,
                    label_smoothing=self.label_smoothing
                )

            # (B) 🔥 Cross-Modal Contrastive Alignment Loss
            # Aligns fused vision ↔ Vietnamese text CLS in a shared 128D space.
            # Applied only during training (labels is not None) and only when
            # batch_size > 1 (need at least one negative per anchor).
            if self.use_contrastive and self.vision_contrastive_head is not None and batch_size > 1 and not text_only:
                contrastive_loss = self.compute_contrastive_loss(
                    fused_vision=vision_prefgate,  # Flamingo-fused, pre-gating (clean signal)
                    text_cls=text_cls,             # LoRA-adapted BOS/CLS
                    labels=labels                  # for false-negative masking
                )

            # (B2) 🔬 Gate-Answer Contrastive (GAC): supervise CHINH gate qua gated-vision <-> ANSWER
            _gac_l = getattr(self, 'gate_answer_contrastive_lambda', 0.0)
            _gac_gated = getattr(self, '_gac_gated', None)
            _gac_alpha = getattr(self, '_gac_alpha', None)
            if (_gac_l and _gac_l > 0 and self.gac_vision_head is not None
                    and batch_size > 1 and not text_only
                    and self.use_vision_gate and _gac_gated is not None and labels is not None):
                _gac_loss = self.compute_answer_gate_contrastive(_gac_gated, labels, alpha=_gac_alpha)
                total_loss_gac = _gac_l * _gac_loss   # them vao total_loss ben duoi (sau khi total_loss khoi tao)
            else:
                total_loss_gac = None

            # (B3) 🔬 Gate diversity regularizer v2 (2026-08-09, SUA LOI THIET KE v1):
            # v1 thuong -std(alpha) -- NHUNG std bi maximize CHINH XAC boi phan bo LUONG CUC
            # (da patch alpha≈min, it patch alpha≈max) -- do la GAC_ABC's failure mode (88.3%
            # patch dong ve sat san 0.1-0.2), khong phai thu v1 dinh ngan. v1 vo tinh THUONG
            # dung cai can tranh.
            # v2: thuong ENTROPY NHI PHAN trung binh theo patch -- H(a)=-a*log(a)-(1-a)*log(1-a),
            # cuc dai tai a=0.5, cuc tieu tai a->0/1. Entropy cao <=> alpha PHAN TAN quanh giua,
            # entropy thap <=> alpha don ve cuc tri (du 1 cum hay lu?ng cuc deu bi phat). Day moi
            # dung "noi suy mem" ma paper mo ta (Sec 3.3: "soft interpolation preserves a coherent
            # representational space"), khac han v1 vo tinh khuyen khich luong cuc.
            _div_l = getattr(self, 'gate_diversity_lambda', 0.0)
            if (_div_l and _div_l > 0 and self.use_vision_gate
                    and _gac_alpha is not None and not text_only):
                _a = _gac_alpha if _gac_alpha.dim() == 2 else _gac_alpha.squeeze(-1)
                _eps = 1e-6
                _ac = _a.clamp(_eps, 1 - _eps)
                _entropy = -(_ac * torch.log(_ac) + (1 - _ac) * torch.log(1 - _ac))  # [B,P], max=log(2)
                total_loss_diversity = -_div_l * _entropy.mean()
            else:
                total_loss_diversity = None

            # (C) Inter-type gate divergence loss
            # Forces VisionGating to produce different alpha patterns per question type.
            # No-op when use_vision_gate=False, question_types absent, or <2 types in batch.
            if (self.use_gate_divergence
                    and gate_stats is not None       # vision gate was active
                    and question_types is not None): # type labels available (training with PK)
                divergence_loss = self.compute_gate_divergence_loss(
                    gate_values=gate_values,
                    type_ids=question_types,
                )

            # (D) Multi-task total loss: Answer + Type (aux) + Contrastive (aux) + Divergence (aux)
            total_loss = answer_loss

            if type_loss is not None:
                total_loss = total_loss + self.type_loss_weight * type_loss

            if vq_loss is not None:
                total_loss = total_loss + self.codebook_lambda * vq_loss

            if contrastive_loss is not None:
                total_loss = total_loss + self.contrastive_lambda * contrastive_loss

            if total_loss_gac is not None:
                total_loss = total_loss + total_loss_gac
                self._last_gac = float(total_loss_gac.detach())

            if total_loss_diversity is not None:
                total_loss = total_loss + total_loss_diversity
                self._last_gate_diversity = float(total_loss_diversity.detach())

            if divergence_loss is not None:
                total_loss = total_loss + self.gate_divergence_lambda * divergence_loss

            # 🔬 QGND — NEO CAU NOI THI GIAC -> NGON NGU BANG TU VUNG CAU HOI.
            #
            # CHAN DOAN: cau noi (vision_proj + fusion) CHI duoc giam sat boi 314 chuoi dap an,
            # nen anh cua phep anh xa bi nhot trong bao cua chung. Do duoc: bo 5 dap an khoi train
            # -> mo hinh KHONG BAO GIO noi ra chung (0/3001) du trie CHO PHEP, va thay bang hang xom
            # ngu nghia (hươu cao cổ -> ngựa vằn 43/46). Tu tieng Viet chua tung huan luyen xep hang
            # 311.6/347 trong khi dap an DA hoc nhung HIEM xep 182.8 va ngau nhien la 174.
            # Cu the hon nua: "hươu cao cổ" xuat hien 164 LAN trong CAU HOI train ma mo hinh van
            # khong noi ra duoc lan nao. Tu vung cau hoi co 2336 tu, tu vung dap an chi 377.
            #
            # CACH LAM: ep dac trung thi giac (TRUOC hop nhat) du doan cac TU NOI DUNG cua cau hoi,
            # cham diem bang CHINH ma tran embedding cua BARTpho. Hai he qua:
            #   - KHONG THEM MOT THAM SO NAO (dung lai lm_head.weight, von buoc chung voi embedding)
            #   - dac trung thi giac buoc phai roi DUNG VAO CHO CUA TU trong khong gian tu vung,
            #     thay vi roi vao cho nao tien cho decoder sinh 1 trong 314 chuoi
            #
            # detach() tren embedding la BAT BUOC: neu khong, mat mat co the duoc thoa man bang cach
            # DI CHUYEN TU thay vi di chuyen THI GIAC — dung huong nguoc lai voi cai ta muon.
            # Lay TRUOC hop nhat vi sau GCA thi dac trung thi giac da chua thong tin cau hoi,
            # dau phu se doc len thay vi phai neo that.
            # 🔬 IQG — SINH LAI CAU HOI TU ANH (inverse VQA).
            #
            # VI SAO CAN, va vi sao QGND KHONG DU (da do, 6 cau hinh deu 0/199):
            #   QGND day CAU NOI bang mot muc tieu PHAN LOAI da nhan ("nhung tu nao co trong cau
            #   hoi"). No khong day DECODER "hay SINH tu nay". Decoder chi tung nhan gradient sinh
            #   cho dung 309 chuoi dap an, nen chinh sach sinh cua no chua bao gio duoc yeu cau
            #   phat ra mot tu ngoai tap do. Do la ly do neo 982 token ma van 0/199.
            # IQG bat DECODER sinh chuoi tuy y (2336 tu, co trinh tu) tu BO NHO THI GIAC — dung
            #   nang luc dang thieu. Day la nguon "diem tuong ung anh-chu" MOI duy nhat lay duoc
            #   ma khong can du lieu ngoai.
            #
            # BO NHO PHAI LA THI GIAC TRUOC HOP NHAT. Neu dung fused_vision thi GCA da tron cau hoi
            # vao do roi -> decoder chi viec CHEP LAI, mat mat ve 0 ma khong hoc duoc gi.
            # Day dung la cai bay ma QGND cung phai ne, va la ly do ca hai cung lay vision_features.
            _iq = getattr(self, 'iqg_lambda', 0.0)
            if (_iq > 0 and input_ids is not None and vision_features is not None
                    and labels is not None):
                _qlbl = input_ids.clone()
                _pad = self.config.pad_token_id
                _qlbl[_qlbl == _pad] = -100
                if attention_mask is not None:
                    _qlbl[attention_mask == 0] = -100
                _qdec = shift_tokens_right(_qlbl.clone().masked_fill(_qlbl == -100, _pad),
                                           _pad, self.config.decoder_start_token_id)
                # 🔬 CHE TIEN TO (--iqg_mask): BAT BUOC, neu khong nhiem vu TU GIAI DUOC.
                # Do duoc o ban dau tien (khong che): IQG loss 4.6565 -> 0.0462. Sinh lai mot cau
                # hoi 13 token chi tu anh ma dat 0.046 la khong the — decoder dang doc TIEN TO cua
                # chinh cau hoi (teacher forcing) va lam mo hinh ngon ngu, khong hoc noi thi giac.
                # Doi chung hau kiem xac nhan: bo nho thi giac ZERO chi lam loss tang 0.035 nats.
                # Che ngau nhien token dau vao -> nhung vi tri bi che khong con tien to de dua vao,
                # vision buoc phai ganh.
                _mk = getattr(self, 'iqg_mask', 0.0)
                if _mk > 0:
                    _m = (torch.rand_like(_qdec, dtype=torch.float) < _mk) & (_qdec != _pad)
                    _m[:, 0] = False                      # giu token khoi dau
                    _qdec = _qdec.masked_fill(_m, _pad)
                _mem = vision_features
                _mmask = torch.ones(_mem.size(0), _mem.size(1),
                                    device=_mem.device, dtype=torch.long)
                _do = self.decoder(input_ids=_qdec, attention_mask=None,
                                   encoder_hidden_states=_mem, encoder_attention_mask=_mmask)
                _qlg = self.lm_head(_do.last_hidden_state)
                iqg_loss = F.cross_entropy(_qlg.reshape(-1, _qlg.size(-1)),
                                           _qlbl.reshape(-1), ignore_index=-100)
                # giu lai gap cu: no chi duoc tinh moi k buoc, tao dict moi se xoa mat no
                _prev_gap = getattr(self, 'last_iqg', {}).get('gap_no_vision')
                self.last_iqg = {'loss': float(iqg_loss.detach()),
                                 'tok_per_sample': float((_qlbl != -100).float().sum(1).mean())}
                if _prev_gap is not None:
                    self.last_iqg['gap_no_vision'] = _prev_gap
                # 🔬 CHOT CHONG RONG, CHAY NGAY TRONG HUAN LUYEN (bai hoc tu ban dau tien):
                # tinh lai cung mot loss voi bo nho thi giac ZERO. Neu chenh ~ 0 thi muc tieu nay
                # dang duoc giai MA KHONG CAN ANH -> vo nghia, va phai thay ngay chu khong doi
                # den luc eval xong moi phat hien.
                if getattr(self, 'iqg_check_every', 0) > 0:
                    self._iqg_step = getattr(self, '_iqg_step', 0) + 1
                    if self._iqg_step % self.iqg_check_every == 0:
                        with torch.no_grad():
                            _d0 = self.decoder(input_ids=_qdec, attention_mask=None,
                                               encoder_hidden_states=torch.zeros_like(_mem),
                                               encoder_attention_mask=_mmask)
                            _l0 = F.cross_entropy(self.lm_head(_d0.last_hidden_state)
                                                  .reshape(-1, _qlg.size(-1)),
                                                  _qlbl.reshape(-1), ignore_index=-100)
                        self.last_iqg['gap_no_vision'] = float(_l0 - iqg_loss.detach())
                total_loss = total_loss + _iq * iqg_loss

            _ql = getattr(self, 'qgnd_lambda', 0.0)
            _qids = getattr(self, 'qgnd_ids', None)
            if (_ql > 0 and _qids is not None and _qids.numel() > 0
                    and input_ids is not None and vision_features is not None):
                _E = F.normalize(self.lm_head.weight[_qids].detach().float(), dim=-1)   # [K, D]
                _v = F.normalize(vision_features.float().mean(dim=1), dim=-1)           # [B, D]
                _lg = (_v @ _E.t()) / float(getattr(self, 'qgnd_temp', 0.07))           # [B, K]
                _tg = (input_ids.unsqueeze(-1) == _qids.view(1, 1, -1)).any(dim=1).float()
                _npos = _tg.sum(dim=1).mean().clamp(min=1.0)
                _pw = torch.full((_qids.numel(),), float(_qids.numel()) / float(_npos),
                                 device=_lg.device, dtype=_lg.dtype)
                qgnd_loss = F.binary_cross_entropy_with_logits(_lg, _tg, pos_weight=_pw)
                self.last_qgnd = {'loss': float(qgnd_loss.detach()),
                                  'pos_per_sample': float(_tg.sum(dim=1).mean())}
                total_loss = total_loss + _ql * qgnd_loss

            # 🔬 (A) alpha-identity regularizer: phat lambda*(1-alpha)^2 -> gate CHI gate khi viec
            # do GIAM answer loss DU de vuot phat. Loai nao gating khong giup loss (COUNT: hai ±1
            # vo hinh voi CE) -> phat keo alpha->1 -> HET HAI. Loai nao gating giup (COLOR) -> giu alpha<1.
            _arl = getattr(self, 'alpha_reg_lambda', 0.0)
            if _arl > 0 and self.use_vision_gate and self.vision_gating is not None:
                _a = getattr(self.vision_gating, 'last_alpha', None)
                if _a is not None:
                    total_loss = total_loss + _arl * ((1.0 - _a) ** 2).mean()

            # 🔬 LAYERSCALE PER-TYPE: keo beta_type ve 0 (identity) -> loai khong loi tu tat gate
            _lsl = getattr(self, 'gate_layerscale_l2', 0.0)
            if (_lsl > 0 and self.use_vision_gate and self.vision_gating is not None
                    and getattr(self.vision_gating, 'gate_layerscale_pertype', False)):
                total_loss = total_loss + _lsl * (self.vision_gating.gate_ls ** 2).mean()

            # 🔬 DO-NO-HARM gate loss: phat relu(loss_gate_on - loss_gate_off) per-sample.
            # Reference = decoder chay lai voi vision KHONG gate (fused_vision, pre-gate).
            # Sample gate lam TE (on>off = break) -> phat -> gate hoc rut ve identity o do (giam break).
            # Sample gate GIUP (on<off = fix) -> relu=0, khong phat -> giu fix. Toi uu truc tiep flip-asymmetry.
            _hl = getattr(self, 'gate_harm_lambda', 0.0)
            if (_hl > 0 and self.training and self.use_vision_gate and self.vision_gating is not None
                    and labels is not None and self.summary_token is None and self.slot_attn is None
                    and not self.use_logits_bias):
                if getattr(self, 'decoder_vision_only', False):
                    _ehs_off = fused_vision
                    _emask_off = torch.ones(batch_size, fused_vision.size(1), device=attention_mask.device)
                else:
                    _ehs_off = torch.cat([fused_vision, text_for_concat], dim=1)
                    _emask_off = torch.cat([torch.ones(batch_size, fused_vision.size(1), device=attention_mask.device),
                                            attention_mask], dim=1)
                with torch.no_grad():
                    _dec_off = self.decoder(input_ids=decoder_input_ids, attention_mask=None,
                                            encoder_hidden_states=_ehs_off, encoder_attention_mask=_emask_off)
                    _logits_off = self.lm_head(_dec_off.last_hidden_state)
                _B, _T = labels.size(); _V = base_answer_logits.size(-1)
                _ce_on_t = F.cross_entropy(base_answer_logits.reshape(-1, _V), labels.reshape(-1),
                                           ignore_index=-100, reduction='none').view(_B, _T)     # grad qua gate
                _ce_off_t = F.cross_entropy(_logits_off.reshape(-1, _V), labels.reshape(-1),
                                            ignore_index=-100, reduction='none').view(_B, _T).detach()
                if getattr(self, 'gate_harm_protect', False):
                    # EM-aligned: CHI bao ve token ma gate-OFF da argmax DUNG -> chong right->wrong truc tiep
                    _off_correct = ((_logits_off.argmax(-1) == labels) & (labels != -100)).float()
                    _harm = (torch.relu(_ce_on_t - _ce_off_t) * _off_correct).sum() / _off_correct.sum().clamp(min=1)
                else:
                    _valid = (labels != -100).float()
                    _harm = (torch.relu(_ce_on_t - _ce_off_t) * _valid).sum() / _valid.sum().clamp(min=1)
                total_loss = total_loss + _hl * _harm
                self._last_harm = float(_harm.detach())

        return DeterministicVQAOutput(
            kl_pretrained_loss=kl_pretrained_loss,
            answer_logits=answer_logits,
            answer_loss=answer_loss,
            type_loss=type_loss,
            total_loss=total_loss,
            answer_cls_logits=answer_cls_logits,
            type_logits=type_logits,
            gate_stats=gate_stats,
            vision_kd_loss=vision_kd_loss,    # always None (KD removed)
            text_kd_loss=text_kd_loss,         # always None (KD removed)
            contrastive_loss=contrastive_loss,
            divergence_loss=divergence_loss    # inter-type gate divergence
        )
    
    def _decode_seq(self, token_ids) -> str:
        """Decode token ids, explicitly stripping decoder_start/BOS tokens that
        BartphoTokenizer may not include in all_special_ids (causing 'ôr' artifacts)."""
        ids = [t for t in token_ids.tolist()
               if t != self.config.decoder_start_token_id
               and t != self.tokenizer.bos_token_id]
        return self.tokenizer.decode(ids, skip_special_tokens=True).strip()

    @torch.inference_mode()  # Faster than @torch.no_grad()!
    def _sample_decoder_once(
        self,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        max_length: int,
        temperature: float,
        predicted_types,
        batch_size: int,
        device,
    ):
        """Single temperature-sampled decoder pass. Used by majority-vote generation."""
        generated_ids = torch.full(
            (batch_size, 1), self.config.decoder_start_token_id,
            dtype=torch.long, device=device,
        )
        past_key_values = None
        done = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(max_length):
            cur_input = generated_ids if past_key_values is None else generated_ids[:, -1:]
            dec_out = self.decoder(
                input_ids=cur_input,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = dec_out.past_key_values

            base_logits = self.lm_head(dec_out.last_hidden_state[:, -1:, :])
            logits = (self.logits_bias(base_logits, predicted_types)
                      if self.use_logits_bias and self.logits_bias is not None
                         and predicted_types is not None
                      else base_logits)

            probs = torch.softmax(logits[:, 0, :] / temperature, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1)  # [B, 1]
            next_tokens = torch.where(
                done.unsqueeze(-1),
                torch.full_like(next_tokens, self.config.eos_token_id),
                next_tokens,
            )
            generated_ids = torch.cat([generated_ids, next_tokens], dim=1)
            done = done | (next_tokens.squeeze(-1) == self.config.eos_token_id)
            if done.all():
                break

        return [self._decode_seq(generated_ids[i]) for i in range(batch_size)]


    def _enrich_l6(self, vision_features):
        # PROBE: v_dec = v_L12 + enrich([v_L12 ; proj(L6)]). Uncond, no gate. zero-init -> T0 tai init.
        if not getattr(self, 'vision_l6_enrich', False):
            return vision_features
        af = getattr(self, '_gate_alpha_feats', None)
        if af is None or af.size(1) != vision_features.size(1):
            return vision_features
        return vision_features + self.l6_enrich(torch.cat([vision_features, af], dim=-1))

    def _box_tokens_from_region_map(self, region_map, dtype):
        """region_map [B,196] chi so ca the tung patch -> ([B,K,D] token box, [B,K] mask).

        Tinh tren luoi 14x14. Ca the id 0 la nen, bo qua. K = box_max_inst.
        Bat bien voi thu tu id: attention bat bien hoan vi tren tap key/value, va dac trung
        cua moi token chi phu thuoc vao TAP patch cua ca the do, khong phu thuoc gia tri id.
        """
        B, P = region_map.shape
        g = int(round(P ** 0.5))
        K = int(getattr(self, 'box_max_inst', 32))
        dev = region_map.device
        yy = (torch.arange(P, device=dev) // g).to(dtype) / max(g - 1, 1)
        xx = (torch.arange(P, device=dev) % g).to(dtype) / max(g - 1, 1)
        feats = torch.zeros(B, K, 5, device=dev, dtype=dtype)
        mask = torch.zeros(B, K, device=dev, dtype=dtype)
        for b in range(B):
            ids = torch.unique(region_map[b])
            ids = ids[ids > 0][:K]
            for j, i in enumerate(ids):
                m = (region_map[b] == i)
                n = m.sum().to(dtype)
                sy, sx = yy[m], xx[m]
                feats[b, j] = torch.stack([sy.mean(), sx.mean(),
                                           sy.max() - sy.min(), sx.max() - sx.min(),
                                           n / float(P)])
                mask[b, j] = 1.0
        return self.box_token_proj(feats), mask


    def _apply_psa(self, v):
        """🔬 Cho cac patch attend LAN NHAU. Dat GIUA Flamingo va TCVG.

        Ly do: fusion_type='text2vision' co query=vision, key/value=text — patch attend sang TEXT.
        Decoder thi attend tu text sang vision. Nen sau SigLIP dong bang, patch KHONG BAO GIO
        attend lan nhau trong phan hoc duoc. Model khong co cho nao hoc "hai patch cung mot vat",
        ma DEM can dung dieu do (COUNT: du dia oracle +23.64, la loai te nhat 66.22).
        out_proj zero-init -> tai init tra ve v NGUYEN VEN (non-harm tuyet doi).
        """
        if not getattr(self, 'patch_self_attn', False) or self.psa is None:
            return v
        h = self.psa_ln(v)
        a, _ = self.psa(h, h, h, need_weights=False)
        return v + self.psa_out(a)

    def _apply_type_moe(self, v, type_ids, type_probs=None):
        # type-routed experts: v_out = v + expert_{type}(v), identity o init.
        # HARD: route theo argmax type_ids (mis-route = ap nham nguyen FFN -> sap YesNo tren ViVQA-X).
        # SOFT: v_out = v + sum_t p_t * expert_t(v), p_t = softmax(type_logits) -> mis-predict thanh
        #       pha tron nhe thay vi tham hoa, khop train/test, robust voi type-misprediction.
        if not getattr(self, 'type_moe', False):
            return v
        if getattr(self, 'type_moe_soft', False) and type_probs is not None:
            delta = torch.zeros_like(v)
            for t in range(4):
                p = type_probs[:, t].view(-1, *([1] * (v.dim() - 1)))  # [B,1,1] broadcast
                delta = delta + p * self.type_experts[t](v)
            return v + delta
        if type_ids is None:
            return v
        delta = torch.zeros_like(v)
        for t in range(4):
            m = (type_ids == t)
            if m.any():
                delta[m] = self.type_experts[t](v[m])
        return v + delta

    def generate(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_length: int = 5,
        num_beams: int = 1,
        temperature: float = 1.0,
        do_sample: bool = False,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.0,
        return_type_preds: bool = False,
        num_samples: int = 1,
        vote_temp: float = 0.8,
        prefix_trie: dict = None,
        gold_types: torch.Tensor = None,
        legacy_beam: bool = False,
        region_map: torch.Tensor = None,  # 🔬 [B, num_patches] chi so COCO-region (tcvg_spatial_blend), None = hanh vi cu
        peek_embedding: torch.Tensor = None,  # 🔬 [B, hidden] tin hieu "sap can gi" (tcvg_dynamic_peek); None + flag bat -> tu sinh draft pass 1
    ):
        """
        Generate answers với greedy (num_beams=1) hoặc beam search (num_beams>1).

        num_beams=1 → greedy argmax (nhanh, dùng khi train sampling)
        num_beams>1 → beam search thực sự qua HuggingFace decoder.generate()
                      (chậm hơn ~num_beams lần, nhưng tốt hơn EM ~1-2%)
        """
        batch_size = pixel_values.size(0)
        device = pixel_values.device

        # 🔬 dynamic_peek, pass 1/2: chua co peek_embedding -> sinh draft NHANH (greedy, peek=0
        # tuc hanh vi tinh cu) de lay tin hieu "sap can gi", roi moi chay THAT (pass 2) voi peek
        # do. peek=0 (khong phai None) tranh de quy vo han (dieu kien duoi chi kich hoat khi
        # peek_embedding is None).
        if getattr(self, 'tcvg_dynamic_peek', False) and peek_embedding is None:
            with torch.no_grad():
                _draft_texts = self.generate(
                    pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask,
                    max_length=max_length, num_beams=1, region_map=region_map,
                    peek_embedding=torch.zeros(batch_size, self.decoder.config.d_model, device=device),
                )
                _draft_enc = self.tokenizer(_draft_texts, truncation=True, padding=True,
                                             max_length=max_length, return_tensors='pt').to(device)
                _draft_emb = self.decoder.embed_tokens(_draft_enc['input_ids'])
                _draft_mask = _draft_enc['attention_mask'].unsqueeze(-1).float()
                peek_embedding = (_draft_emb * _draft_mask).sum(dim=1) / _draft_mask.sum(dim=1).clamp(min=1.0)

        # ── 1. Vision encoding ────────────────────────────────────────────
        _bbl = getattr(self, 'vision_backbone_layer', -1)
        _need_hs = (getattr(self, 'gate_vision_layer', -1) >= 0) or (_bbl >= 0)
        vision_outputs = self.vision_encoder(pixel_values=pixel_values, output_hidden_states=_need_hs)
        self._gate_alpha_feats = (self.gate_layer_proj(vision_outputs.hidden_states[self.gate_vision_layer][:, -self.num_patches:, :]) if getattr(self, 'gate_vision_layer', -1) >= 0 else None)
        patch_tokens = vision_outputs.hidden_states[_bbl] if _bbl >= 0 else vision_outputs.last_hidden_state  # [B, seq_len, hidden]

        # Remove CLS token if present (same as forward())
        if patch_tokens.size(1) > self.num_patches:
            patch_tokens = patch_tokens[:, 1:, :]
            assert patch_tokens.size(1) == self.num_patches, \
                f"Shape mismatch after CLS removal in generate(): {patch_tokens.size(1)} != {self.num_patches}"

        # 🔥 Apply type-conditioned adapter if present (consistent with forward())
        if self.vision_adapter is not None:
            # During inference type_ids unknown → use predicted types below
            # Pass None here, adapter uses uniform transform (no type conditioning)
            patch_tokens = self.vision_adapter(patch_tokens, type_ids=None)

        # Add position embeddings
        patch_tokens = patch_tokens + self.vision_pos_embed.expand(batch_size, -1, -1)
        vision_features = self.vision_proj(patch_tokens)
        vision_features = self._enrich_l6(vision_features)  # 🔬 PROBE

        # Optionally prepend SigLIP pooler_output as a global vision token
        if self.use_siglip_pooler and hasattr(vision_outputs, 'pooler_output') and vision_outputs.pooler_output is not None:
            global_feat = self.siglip_global_proj(vision_outputs.pooler_output)  # [B, bart_hidden]
            vision_features = torch.cat([global_feat.unsqueeze(1), vision_features], dim=1)  # [B, 197, bart_hidden]
            if self._gate_alpha_feats is not None:
                self._gate_alpha_feats = torch.cat([global_feat.unsqueeze(1), self._gate_alpha_feats], dim=1)

        # text_only_mode: zero out vision (used for shortcut analysis)
        if getattr(self, 'text_only_mode', False):
            vision_features = torch.zeros(
                batch_size, self.num_patches, vision_features.size(-1),
                device=vision_features.device, dtype=vision_features.dtype
            )

        # ── 2. Text encoding ──────────────────────────────────────────────
        text_encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        text_features = text_encoder_outputs.last_hidden_state

        # Predict type for type-conditioned generation
        if self.use_attn_pool_cls:
            text_cls = self.attn_pool(text_features, attention_mask)
        elif self.use_mean_pool_cls:
            _mask = attention_mask.float().unsqueeze(-1)
            text_cls = (text_features * _mask).sum(1) / _mask.sum(1).clamp(min=1)
        else:
            text_cls = text_features[:, 0, :]
        predicted_types = None
        # 🔬 codebook khong giam sat: cum tu t_cls thay cho type head
        cb_emb_g = cb_idx_g = None
        if self.type_codebook is not None:
            cb_emb_g, cb_idx_g, _, _ppl_g = self.type_codebook(text_cls)
            self.last_codebook_idx = cb_idx_g.detach()
            self.last_codebook_ppl = float(_ppl_g)
            predicted_types = cb_idx_g
        self._type_vec = None
        type_logits = None
        if self.use_type_task and self.type_head is not None:
            if getattr(self, 'type_branch', None) is not None:
                self._type_vec = self.type_branch(text_cls)   # [B, D] feed gate
                type_logits = self.type_head(self._type_vec)
            else:
                type_logits = self.type_head(text_cls)
            if self.type_codebook is None:
                predicted_types = torch.argmax(type_logits, dim=-1)  # [B]

        # 🔥 Oracle diagnostic: nếu truyền gold_types thì TCVG/adapter dùng nhãn chuẩn
        # thay vì dự đoán của type head. predicted_types vẫn giữ nguyên để báo cáo
        # type accuracy. Chỉ dùng để đo oracle gap, KHÔNG phải chế độ triển khai.
        gate_types = predicted_types if self.type_codebook is not None else (
            gold_types if gold_types is not None else predicted_types)
        # 🔬 gate_type_blind: doi xung voi nhanh train — gate KHONG duoc biet loai, ke ca khi
        # co gold_types truyen vao. Giu train/test cung mot che do.
        if getattr(self, 'gate_type_blind', False):
            gate_types = None

        # ── 3. Fusion ─────────────────────────────────────────────────────
        vision_features_pre_flamingo = vision_features  # for delta gate
        fused_vision = vision_features
        fused_text = text_features
        # 🔥 tcvg_fusion_gate (generate): alpha dieu khien cuong do hop nhat GCA theo tung patch
        _fuse_alpha_g = None
        if getattr(self, 'tcvg_fusion_gate', False) and self.use_vision_gate:
            if getattr(self, 'tcvg_fg_2pass', False):
                _v1g, _t1g = vision_features, text_features
                for _flg in self.flamingo_fusion:
                    _v1g, _t1g = _flg(_v1g, _t1g, attention_mask)
                _fuse_alpha_g = self.vision_gating.compute_alpha(
                    _v1g, _t1g, type_ids=gate_types, text_attention_mask=attention_mask)
            else:
                _fuse_alpha_g = self.vision_gating.compute_alpha(
                    vision_features, text_features, type_ids=gate_types,
                    text_attention_mask=attention_mask)
        _gca_attn_weights_g = None  # 🔬 HYP #1
        if getattr(self, 'concat_fusion', False):
            fused_vision, fused_text = self._concat_fuse(fused_vision, fused_text, attention_mask)
        else:
            _btg = _btmg = None
            if getattr(self, 'gca_box_tokens', False) and region_map is not None:
                _btg, _btmg = self._box_tokens_from_region_map(region_map, fused_vision.dtype)
            for _li, fusion_layer in enumerate(self.flamingo_fusion):
                fused_vision, fused_text = fusion_layer(
                    fused_vision, fused_text, attention_mask, residual_scale=_fuse_alpha_g,
                    extra_kv=_btg, extra_kv_mask=_btmg)
                if (self.vision_gating_mid is not None
                        and _li == 0 and len(self.flamingo_fusion) > 1):
                    fused_vision, _ = self.vision_gating_mid(
                        fused_vision, fused_text,
                        type_ids=gate_types,
                        text_attention_mask=attention_mask)
            if getattr(self, 'tcvg_alpha_from_gca', False) and self.fusion_type == 'text2vision':
                _gca_attn_weights_g = getattr(self.flamingo_fusion[-1], 'last_attn_weights', None)
        text_features = fused_text

        # Type-specific adapter: BOS/CLS position only, post-Flamingo (see forward() comment)
        if self.type_text_adapter is not None and gate_types is not None:
            _cls = self.type_text_adapter(text_features[:, 0, :], gate_types)
            text_features = torch.cat([_cls.unsqueeze(1), text_features[:, 1:, :]], dim=1)

        # ── 4. Vision gating (AFTER Flamingo) ────────────────────────────
        # 🔥 Che do tcvg_fusion_gate: alpha DA duoc ap vao residual cua GCA o tren, nen KHONG
        # tron hau ky nua. Neu tron them thi gate bi ap HAI LAN luc suy luan trong khi luc train
        # chi ap MOT LAN -> lech phan phoi hoan toan, do duoc val EM = 0.0.
        if getattr(self, 'tcvg_fusion_gate', False):
            gated_vision = fused_vision
        elif self.use_vision_gate:
            _gate_type_ovr_g = self._type_vec if getattr(self, '_type_vec', None) is not None else cb_emb_g
            # 🔬 kien truc hai tang -- PHAI khop y het forward() (bai hoc tcvg_topk_random: sua
            # forward ma quen generate -> moi eval EM chay nhanh khac han luc train)
            # 🔬 patch_self_attn — PHAI khop y het forward() (bai hoc tcvg_topk_random: sua forward
            #    ma quen generate -> EM luc eval khac han luc train)
            fused_vision = self._apply_psa(fused_vision)
            _gate_in_g = fused_vision
            if (self.slot_attn is not None and getattr(self, 'slot_stage', 'post') == 'pre_gated'):
                _slots_pg_g = self.slot_attn(fused_vision, text_features[:, 0, :], gate_types)
                self._slots_in_gate = _slots_pg_g.size(1)
                _gate_in_g = torch.cat([fused_vision, _slots_pg_g], dim=1)
            gated_vision, _ = self.vision_gating(
                _gate_in_g, text_features,
                type_ids=gate_types,
                type_emb_override=_gate_type_ovr_g,   # 🔬 type_vec (detach branch) hoac codebook
                alpha_feats=getattr(self,'_gate_alpha_feats',None),
                text_attention_mask=attention_mask,
                detach_for_gate=self.gate_detach_input,
                vision_orig=(vision_features_pre_flamingo
                             if (self.use_delta_gate or getattr(self, 'gate_blend_vorig', False)
                                 or getattr(self, 'gate_gca_residual', False)) else None),
                region_map=region_map,
                peek_embedding=peek_embedding,
                gca_attn_weights=_gca_attn_weights_g,
            )
        else:
            gated_vision = fused_vision
        _moe_probs_g = F.softmax(type_logits, dim=-1) if type_logits is not None else None
        gated_vision = self._apply_type_moe(gated_vision, gate_types, type_probs=_moe_probs_g)

        # ── 5. Encoder hidden states ──────────────────────────────────────
        # 🔬 BUG (phat hien 2026-08-08): nhanh nay KHONG BAO GIO check tcvg_topk_random —
        # moi eval EM (eval.py goi model.generate(), khong phai forward()) voi --probe_topk_random
        # tu truoc den gio VAN CHON top-alpha that, giong het nhanh "top" -> moi ket qua
        # "topk-rand == topk-top" (gap=+0.00) tren TOAN BO investigation la SO SANH VOI CHINH
        # NO, khong phai voi random that. forward() (dung cho training) co xu ly dung (dong
        # 2635), generate() (dung cho eval EM that) thi thieu. Da bo sung cho khop forward().
        _topk = getattr(self, 'tcvg_topk', 0)
        if _topk and _topk > 0 and self.use_vision_gate and getattr(self.vision_gating, 'last_alpha', None) is not None:
            _k = min(int(_topk), gated_vision.size(1))
            _a = self.vision_gating.last_alpha
            if _a.dim() == 3:
                _a = _a.squeeze(-1)
            _npad = gated_vision.size(1) - _a.size(1)
            _bottom = getattr(self, 'tcvg_topk_bottom', False)
            _fill = -1e9 if _bottom else 1e9  # token toan cuc luon duoc giu, bat ke tieu chi chon
            if _npad > 0:
                _a = torch.cat([_a.new_full((_a.size(0), _npad), _fill), _a], dim=1)
            if getattr(self, 'tcvg_topk_random', False):
                _rnd = torch.rand_like(_a)
                if _npad > 0:
                    _rnd[:, :_npad] = 1e9
                _idx = _rnd.topk(_k, dim=1).indices.sort(dim=1).values
            elif _bottom:
                # 🔬 doi chung 2: giu k patch alpha THAP NHAT (nguoc top) — xem alpha thap co
                # phai la tin hieu (thay vi top cao) khong, sau khi phat hien top < random.
                _idx = _a.topk(_k, dim=1, largest=False).indices.sort(dim=1).values
            else:
                _idx = _a.topk(_k, dim=1).indices.sort(dim=1).values
            gated_vision = torch.gather(
                gated_vision, 1, _idx.unsqueeze(-1).expand(-1, -1, gated_vision.size(-1)))

        # 🔬 decoder_pool_vision — phai lam GIONG forward(), xem giai thich o forward()
        _pv = getattr(self, 'decoder_pool_vision', 0)
        if _pv and _pv > 0 and gated_vision.size(1) > int(_pv):
            gated_vision = F.adaptive_avg_pool1d(
                gated_vision.transpose(1, 2), int(_pv)).transpose(1, 2)

        if getattr(self, 'decoder_vision_only', False):
            encoder_hidden_states = gated_vision
            encoder_attention_mask = torch.ones(batch_size, gated_vision.size(1), device=device)
        else:
            encoder_hidden_states = torch.cat([gated_vision, text_features], dim=1)
            encoder_attention_mask = torch.cat([
                torch.ones(batch_size, gated_vision.size(1), device=device),
                attention_mask
            ], dim=1)
        if self.summary_token is not None:
            _stok = self.summary_token(gated_vision, text_cls, gate_types)   # [B,1,D]
            encoder_hidden_states = torch.cat([encoder_hidden_states, _stok], dim=1)
            encoder_attention_mask = torch.cat(
                [encoder_attention_mask, torch.ones(batch_size, 1, device=device)], dim=1)
        if self.slot_attn is not None:
            # 🔬 slot_stage=pre: PHAI khop y het forward(), neu khong train mot kieu eval kieu khac
            # (lop bug tcvg_topk_random o ngay duoi da tung xay ra dung vi le do).
            if getattr(self, '_slots_in_gate', 0):
                self._slots_in_gate = 0        # pre_gated: slot da o trong gated_vision, khong noi lai
            else:
                _slot_src_g = fused_vision if getattr(self, 'slot_stage', 'post') == 'pre' else gated_vision
                _slots = self.slot_attn(_slot_src_g, text_cls, gate_types)    # [B,K,D]
                encoder_hidden_states = torch.cat([encoder_hidden_states, _slots], dim=1)
                encoder_attention_mask = torch.cat(
                    [encoder_attention_mask, torch.ones(batch_size, _slots.size(1), device=device)], dim=1)

        # ── 6. Decoding ───────────────────────────────────────────────────
        if num_samples > 1:
            # ── Majority voting: sample N times, pick most frequent answer ──
            from collections import Counter
            buckets = [[] for _ in range(batch_size)]
            for _ in range(num_samples):
                decoded = self._sample_decoder_once(
                    encoder_hidden_states, encoder_attention_mask,
                    max_length, vote_temp, predicted_types, batch_size, device,
                )
                for i, d in enumerate(decoded):
                    buckets[i].append(d)
            final = [Counter(b).most_common(1)[0][0] for b in buckets]
            if return_type_preds:
                type_ids = predicted_types.cpu().tolist() if predicted_types is not None else [0] * batch_size
                return final, type_ids
            return final

        if num_beams > 1:
            # ── Beam search với KV-cache (incremental decoding) ───────────
            expanded_hidden = encoder_hidden_states.unsqueeze(1) \
                .expand(-1, num_beams, -1, -1) \
                .reshape(batch_size * num_beams,
                         encoder_hidden_states.size(1),
                         encoder_hidden_states.size(2))
            expanded_mask = encoder_attention_mask.unsqueeze(1) \
                .expand(-1, num_beams, -1) \
                .reshape(batch_size * num_beams, encoder_attention_mask.size(1))
            expanded_types = predicted_types.repeat_interleave(num_beams) if predicted_types is not None else None  # [B*beams]

            bos_ids = torch.full(
                (batch_size, 1),
                self.config.decoder_start_token_id,
                dtype=torch.long, device=device
            )
            # [B, beams, 1]
            beam_seqs   = bos_ids.unsqueeze(1).expand(-1, num_beams, -1).clone()
            beam_scores = torch.zeros(batch_size, num_beams, device=device)
            beam_scores[:, 1:] = -1e9   # chỉ beam 0 active lúc đầu
            done        = torch.zeros(batch_size, dtype=torch.bool, device=device)
            # 🔥 Cờ finished theo TỪNG beam. Thiếu cái này thì beam đã phát EOS vẫn bị
            # cộng thêm log P(EOS) âm ở mọi bước còn lại → câu ngắn bị phạt, beam search
            # lệch sang câu dài. done ở trên chỉ là cờ cấp sample (all beams cùng EOS).
            beam_finished = torch.zeros(batch_size, num_beams, dtype=torch.bool, device=device)
            past_key_values = None      # KV-cache

            for step in range(max_length):
                # Với KV-cache: chỉ feed token mới nhất sau bước đầu
                if step == 0 or past_key_values is None:
                    cur_input = beam_seqs.reshape(batch_size * num_beams, -1)
                else:
                    cur_input = beam_seqs[:, :, -1:].reshape(batch_size * num_beams, 1)

                dec_out = self.decoder(
                    input_ids=cur_input,
                    encoder_hidden_states=expanded_hidden,
                    encoder_attention_mask=expanded_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                past_key_values = dec_out.past_key_values  # cập nhật cache

                # Logits của token cuối
                base_logits = self.lm_head(dec_out.last_hidden_state[:, -1:, :])  # [B*beams, 1, V]
                if self.use_logits_bias and self.logits_bias is not None and expanded_types is not None:
                    logits = self.logits_bias(base_logits, expanded_types)         # [B*beams, 1, V]
                else:
                    logits = base_logits

                # Repetition penalty: divide positive logits, multiply negative logits
                # Special tokens (BOS/EOS/PAD/decoder_start) are excluded because
                # decoder_start_token_id == eos_token_id in BART/BARTpho — penalizing
                # EOS from step 0 would prevent the model from ever stopping.
                if repetition_penalty != 1.0:
                    logits_2d = logits[:, 0, :].clone()  # [B*beams, V]
                    flat_seqs = beam_seqs.reshape(batch_size * num_beams, -1)  # [B*beams, L]
                    # Build mask: True = special token (skip penalty)
                    sp_mask = torch.zeros_like(flat_seqs, dtype=torch.bool)
                    for _attr in ('bos_token_id', 'eos_token_id', 'pad_token_id', 'decoder_start_token_id'):
                        _sid = getattr(self.config, _attr, None)
                        if _sid is not None:
                            sp_mask = sp_mask | (flat_seqs == _sid)
                    prev_scores = torch.gather(logits_2d, 1, flat_seqs)
                    penalized = torch.where(prev_scores < 0, prev_scores * repetition_penalty, prev_scores / repetition_penalty)
                    penalized = torch.where(sp_mask, prev_scores, penalized)  # restore special tokens
                    logits_2d.scatter_(1, flat_seqs, penalized)
                    logits = logits_2d.unsqueeze(1)

                log_probs   = torch.log_softmax(logits[:, 0, :], dim=-1)           # [B*beams, V]

                # 🔥 Prefix trie constrained decoding: mask invalid next tokens
                if prefix_trie is not None:
                    vocab_size_tmp = log_probs.size(-1)
                    eos_id = self.config.eos_token_id
                    for i in range(batch_size * num_beams):
                        b, k = i // num_beams, i % num_beams
                        # Tokens generated so far (skip initial decoder_start token)
                        seq = beam_seqs[b, k, 1:].tolist()
                        # Traverse trie
                        node = prefix_trie
                        valid = True
                        past_eos = False
                        for t in seq:
                            if t == eos_id:
                                past_eos = True
                                node = {}
                                break
                            if t in node:
                                node = node[t]
                            else:
                                valid = False
                                break
                        mask = torch.full((vocab_size_tmp,), -1e9, device=device, dtype=log_probs.dtype)
                        if valid and node:
                            # In-trie: allow only next trie tokens
                            for t in node.keys():
                                if t < vocab_size_tmp:
                                    mask[t] = 0.0
                        else:
                            # Off-trie (valid=False) OR already past EOS (node={}):
                            # Force EOS to prevent garbage tokens being decoded
                            if eos_id < vocab_size_tmp:
                                mask[eos_id] = 0.0
                        log_probs[i] = log_probs[i] + mask

                log_probs   = log_probs.reshape(batch_size, num_beams, -1)         # [B, beams, V]

                vocab_size  = log_probs.size(-1)
                # Các sample đã done: giữ nguyên beam_scores (không cộng thêm),
                # force chọn EOS để sequence không thay đổi nữa
                eos_mask = done.unsqueeze(-1).unsqueeze(-1)  # [B, 1, 1]
                eos_log_probs = torch.full_like(log_probs, -1e9)
                eos_log_probs[:, :, self.config.eos_token_id] = 0.0
                log_probs = torch.where(eos_mask.expand_as(log_probs), eos_log_probs, log_probs)

                # 🔥 Beam đã kết thúc: đóng băng điểm (cộng 0, không phạt) và ép ở lại EOS.
                # Cùng cách xử lý như eos_log_probs ở trên nhưng theo từng beam.
                # legacy_beam=True: tai hien dung hanh vi CO BUG truoc 2026-07-26
                # (khong dong bang beam da xong) — chi dung de do lai anh huong cua bug.
                if beam_finished.any() and not legacy_beam:
                    log_probs = torch.where(
                        beam_finished.unsqueeze(-1).expand_as(log_probs),
                        eos_log_probs, log_probs
                    )

                scores      = beam_scores.unsqueeze(-1) + log_probs                # [B, beams, V]
                scores_flat = scores.reshape(batch_size, -1)                       # [B, beams*V]

                topk_scores, topk_idx = scores_flat.topk(num_beams, dim=-1)        # [B, beams]
                beam_idx  = topk_idx // vocab_size
                token_idx = topk_idx  % vocab_size

                # Rebuild sequences — reorder cache theo beam mới
                new_seqs = []
                for b in range(batch_size):
                    new_seqs.append(torch.stack([
                        torch.cat([beam_seqs[b, beam_idx[b, k]], token_idx[b, k:k+1]])
                        for k in range(num_beams)
                    ]))
                beam_seqs   = torch.stack(new_seqs)   # [B, beams, L+1]
                beam_scores = topk_scores              # [B, beams]
                # Theo beam mới: kế thừa cờ finished của beam cha, rồi đánh dấu beam vừa phát EOS
                beam_finished = (torch.gather(beam_finished, 1, beam_idx)
                                 | (token_idx == self.config.eos_token_id))

                # Reorder KV-cache theo beam mới
                if past_key_values is not None:
                    flat_beam_idx = (
                        torch.arange(batch_size, device=device).unsqueeze(1) * num_beams
                        + beam_idx
                    ).reshape(-1)   # [B*beams]
                    if hasattr(past_key_values, 'reorder_cache'):
                        # New transformers Cache API (DynamicCache, etc.) — reorder in-place
                        past_key_values.reorder_cache(flat_beam_idx)
                    else:
                        # Legacy tuple format
                        past_key_values = tuple(
                            tuple(t.index_select(0, flat_beam_idx) if t is not None else None for t in layer)
                            for layer in past_key_values
                        )

                if legacy_beam:
                    done = done | (token_idx == self.config.eos_token_id).all(dim=-1)
                else:
                    done = done | beam_finished.all(dim=-1)
                if done.all():
                    break

            best_seqs = beam_seqs[:, 0, :]  # beam có score cao nhất
            decoded = [self._decode_seq(best_seqs[i]) for i in range(batch_size)]
            if return_type_preds:
                type_ids = predicted_types.cpu().tolist() if predicted_types is not None else [0] * batch_size
                return decoded, type_ids
            return decoded

        else:
            # ── Greedy decoding với KV-cache (num_beams=1, nhanh nhất) ────
            generated_ids   = torch.full(
                (batch_size, 1),
                self.config.decoder_start_token_id,
                dtype=torch.long, device=device
            )
            past_key_values = None
            done            = torch.zeros(batch_size, dtype=torch.bool, device=device)

            for _ in range(max_length):
                # Bước đầu: feed BOS; các bước sau: chỉ feed token mới nhất
                cur_input = generated_ids if past_key_values is None \
                            else generated_ids[:, -1:]

                decoder_outputs = self.decoder(
                    input_ids=cur_input,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                past_key_values = decoder_outputs.past_key_values

                base_logits = self.lm_head(decoder_outputs.last_hidden_state[:, -1:, :])
                if self.use_logits_bias and self.logits_bias is not None and predicted_types is not None:
                    logits = self.logits_bias(base_logits, predicted_types)
                else:
                    logits = base_logits

                # Repetition penalty (same special-token exclusion as beam search)
                if repetition_penalty != 1.0:
                    logits_2d = logits[:, 0, :].clone()  # [B, V]
                    sp_mask = torch.zeros_like(generated_ids, dtype=torch.bool)
                    for _attr in ('bos_token_id', 'eos_token_id', 'pad_token_id', 'decoder_start_token_id'):
                        _sid = getattr(self.config, _attr, None)
                        if _sid is not None:
                            sp_mask = sp_mask | (generated_ids == _sid)
                    prev_scores = torch.gather(logits_2d, 1, generated_ids)
                    penalized = torch.where(prev_scores < 0, prev_scores * repetition_penalty, prev_scores / repetition_penalty)
                    penalized = torch.where(sp_mask, prev_scores, penalized)
                    logits_2d.scatter_(1, generated_ids, penalized)
                    logits = logits_2d.unsqueeze(1)

                next_tokens = torch.argmax(logits[:, 0, :], dim=-1, keepdim=True)  # [B, 1]

                # Sample đã EOS: tiếp tục emit EOS để decode bỏ qua khi skip_special_tokens
                next_tokens = torch.where(
                    done.unsqueeze(-1),
                    torch.full_like(next_tokens, self.config.eos_token_id),
                    next_tokens
                )
                generated_ids = torch.cat([generated_ids, next_tokens], dim=1)

                done = done | (next_tokens.squeeze(-1) == self.config.eos_token_id)
                if done.all():
                    break

            decoded = [self._decode_seq(generated_ids[i]) for i in range(batch_size)]
            if return_type_preds:
                type_ids = predicted_types.cpu().tolist() if predicted_types is not None else [0] * batch_size
                return decoded, type_ids
            return decoded

    @torch.inference_mode()
    def generate_sample(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_length: int = 10,
        temperature: float = 1.0,
    ):
        """
        Autoregressive sampling decode for SCST. Returns List[str].
        Same encoding pipeline as generate(); only the token selection differs
        (multinomial instead of argmax).
        """
        batch_size = pixel_values.size(0)
        device = pixel_values.device

        # ── Encode (identical to generate()) ─────────────────────────────────
        _need_hs = getattr(self, 'gate_vision_layer', -1) >= 0
        vision_outputs = self.vision_encoder(pixel_values=pixel_values, output_hidden_states=_need_hs)
        self._gate_alpha_feats = (self.gate_layer_proj(vision_outputs.hidden_states[self.gate_vision_layer][:, -self.num_patches:, :]) if _need_hs else None)
        patch_tokens = vision_outputs.last_hidden_state
        if patch_tokens.size(1) > self.num_patches:
            patch_tokens = patch_tokens[:, 1:, :]
        if self.vision_adapter is not None:
            patch_tokens = self.vision_adapter(patch_tokens, type_ids=None)
        patch_tokens = patch_tokens + self.vision_pos_embed.expand(batch_size, -1, -1)
        vision_features = self.vision_proj(patch_tokens)
        vision_features = self._enrich_l6(vision_features)  # 🔬 PROBE

        text_features = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        if self.use_attn_pool_cls:
            text_cls = self.attn_pool(text_features, attention_mask)
        elif self.use_mean_pool_cls:
            _mask = attention_mask.float().unsqueeze(-1)
            text_cls = (text_features * _mask).sum(1) / _mask.sum(1).clamp(min=1)
        else:
            text_cls = text_features[:, 0, :]
        predicted_types = None
        _tv_s = None
        if self.use_type_task and self.type_head is not None:
            if getattr(self, 'type_branch', None) is not None:
                _tv_s = self.type_branch(text_cls)
                predicted_types = torch.argmax(self.type_head(_tv_s), dim=-1)
            else:
                predicted_types = torch.argmax(self.type_head(text_cls), dim=-1)

        vision_features_pre_flamingo = vision_features  # for delta gate
        fused_vision, fused_text = vision_features, text_features
        if getattr(self, 'concat_fusion', False):
            fused_vision, fused_text = self._concat_fuse(fused_vision, fused_text, attention_mask)
        else:
            for layer in self.flamingo_fusion:
                fused_vision, fused_text = layer(fused_vision, fused_text, attention_mask)

        # Gate AFTER Flamingo (same as forward())
        gated_vision, _ = (
            self.vision_gating(fused_vision, fused_text, type_ids=predicted_types,
                               type_emb_override=_tv_s, alpha_feats=getattr(self,'_gate_alpha_feats',None),
                               text_attention_mask=attention_mask,
                               detach_for_gate=self.gate_detach_input,
                               vision_orig=vision_features_pre_flamingo if self.use_delta_gate else None)
            if self.use_vision_gate else (fused_vision, None)
        )

        encoder_hidden_states = torch.cat([gated_vision, fused_text], dim=1)
        encoder_attention_mask = torch.cat([
            torch.ones(batch_size, gated_vision.size(1), device=device),
            attention_mask
        ], dim=1)

        # ── Sampling decode ───────────────────────────────────────────────────
        generated_ids = torch.full(
            (batch_size, 1), self.config.decoder_start_token_id,
            dtype=torch.long, device=device
        )
        past_key_values = None
        done = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(max_length):
            cur_input = generated_ids if past_key_values is None else generated_ids[:, -1:]
            dec_out = self.decoder(
                input_ids=cur_input,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = dec_out.past_key_values
            base_logits = self.lm_head(dec_out.last_hidden_state[:, -1:, :])
            logits = self.logits_bias(base_logits, predicted_types) \
                if (self.use_logits_bias and self.logits_bias is not None and predicted_types is not None) \
                else base_logits
            probs = torch.softmax(logits[:, 0, :] / max(temperature, 1e-6), dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1)  # [B, 1]
            next_tokens = torch.where(
                done.unsqueeze(-1),
                torch.full_like(next_tokens, self.config.eos_token_id),
                next_tokens
            )
            generated_ids = torch.cat([generated_ids, next_tokens], dim=1)
            done = done | (next_tokens.squeeze(-1) == self.config.eos_token_id)
            if done.all():
                break

        return [self._decode_seq(generated_ids[i]) for i in range(batch_size)]

    def compute_seq_logprob(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        target_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Teacher-forced per-sample sum of log probs for target_ids.
        Called during SCST to compute the policy log prob of sampled sequences.
        Requires gradients — do NOT wrap in torch.no_grad().

        target_ids : [B, seq_len] with -100 at padding positions
        Returns    : [B] scalar log prob per sample
        """
        outputs = self.forward(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=target_ids,
        )
        # answer_logits: [B, seq_len, vocab_size]
        log_probs = F.log_softmax(outputs.answer_logits, dim=-1)
        tgt = target_ids.clone()
        mask = (tgt != -100)
        tgt[~mask] = 0  # avoid out-of-bounds index; masked out below
        gathered = log_probs.gather(2, tgt.unsqueeze(-1)).squeeze(-1)  # [B, seq_len]
        return (gathered * mask.float()).sum(-1)  # [B]


if __name__ == '__main__':
    print("="*80)
    print("DETERMINISTIC VQA MODEL (NO LATENT REASONING)")
    print("="*80)
    print("\nKey features:")
    print("  ✅ No VAE/KL regularization")
    print("  ✅ Direct cross-attention fusion")
    print("  ✅ Optimized for low-resource VQA")
    print("  ✅ Stable training, no KL tuning needed")
    print("="*80)
    
    model = DeterministicVQA(
        num_fusion_layers=4,  # Match default in __init__
        gradient_checkpointing=False
    )
    
    print(f"\nTotal params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
    print("Model ready for training! 🎉")
