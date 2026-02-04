"""
TEACHER REPRESENTATIONS EXTRACTION FOR OFFLINE DISTILLATION
============================================================

Extract high-capacity teacher REPRESENTATIONS (not logits!) for knowledge distillation.

Teachers (NO student checkpoint needed!):
  1. Vision: SigLIP-SO400M/14 (~400-430M params, 384px, multilingual)
  2. Text: PhoBERT-large (307M params, Vietnamese-optimized)

What is saved (REPRESENTATIONS ONLY):
  - vision_patch_emb_train.npy: [10200, 729, hidden] - Spatial features
  - vision_cls_emb_train.npy: [10200, hidden] - Global scene
  - text_token_emb_train.npy: [10200, seq, 1024] - Question embeddings
  - text_cls_emb_train.npy: [10200, 1024] - Question representation
  
Storage: ~2.0GB for full dataset (train + val)
Runtime: ~1.5 hours on GPU for 12K samples

Usage (NO checkpoint required!):
    python extract_teacher_logits.py \
        --csv_path train.csv \
        --image_folder vivqa/images \
        --output_dir /kaggle/working/teacher_cache \
        --batch_size 32
"""

import os
import json
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image

from transformers import (
    AutoModel,
    AutoImageProcessor,
    AutoTokenizer,
    AutoModelForMaskedLM
)

# Reuse dataset from existing code
import sys
sys.path.append(os.path.dirname(__file__))
from dataset import VQAGenDataset  # CORRECTED: VQAGenDataset not ViVQADataset


class TeacherVisionEncoder:
    """
    SigLIP-SO400M/14 Vision Teacher (~400-430M params)
    
    CRITICAL CORRECTION:
    - SO400M = 400M training image-text PAIRS (not params!)
    - Model params: ~400-430M (NOT 878M - that was config misread)
    
    Why this is the RIGHT teacher for Vietnamese VQA:
    ✅ Multilingual contrastive alignment (includes Vietnamese contexts)
    ✅ 4-5× student capacity (400M vs 90M SigLIP-base)
    ✅ 729 patches (patch14@384px) vs 196 patches (patch16@224px) = richer spatial info
    ✅ Contrastive signal is "smooth" → excellent for KD (vs classification teachers)
    
    What to distill (THESIS-CRITICAL):
    1. CLS embedding: Global scene understanding
    2. Patch embeddings: Spatial visual features (downsample 729→196 to match student)
    3. Image-text similarity: Cross-modal alignment score
    4. (Optional) Intermediate layer features: Multi-scale representations
    
    ❌ NOT JUST LOGITS! Representation learning is key for VQA.
    """
    def __init__(self, model_name='google/siglip-so400m-patch14-384', device='cuda'):
        print(f"[Vision Teacher] Loading {model_name}...")
        print(f"  📊 Model: ~400-430M params (SO400M = 400M training pairs)")
        self.device = device
        
        # Load vision model directly (not full SigLIP model)
        from transformers import SiglipVisionModel
        self.vision_encoder = SiglipVisionModel.from_pretrained(
            model_name,
            torch_dtype=torch.float16  # 🔥 Load in FP16 to save VRAM!
        ).to(device)
        self.vision_encoder.eval()
        
        # Processor for 384px images
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        
        # Get hidden dimension
        self.hidden_dim = self.vision_encoder.config.hidden_size
        
        # 🔥 Print memory usage
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(device) / 1e9
            reserved = torch.cuda.memory_reserved(device) / 1e9
            print(f"  ✓ Vision teacher loaded: {self.hidden_dim}D features, 729 patches (patch14@384px)")
            print(f"  ✓ GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
        else:
            print(f"  ✓ Vision teacher loaded: {self.hidden_dim}D features, 729 patches (patch14@384px)")
        print(f"  ✓ Advantage: 4-5× student capacity + multilingual alignment")
    
    @torch.no_grad()
    def extract_features(self, images):
        """
        Extract vision features from teacher
        
        Args:
            images: List of PIL Images or [B, 3, H, W] tensor
        
        Returns:
            patch_embeddings: [B, 729, hidden_dim] - Spatial visual features
            cls_embedding: [B, hidden_dim] - Global scene representation
            intermediate_features: [B, num_layers, 729, hidden_dim] - Multi-scale features
            
        NOTE: Removed attention_weights - use CLS embedding instead for global context
        """
        # Preprocess images to 384x384
        if isinstance(images, list):
            pixel_values = self.processor(images, return_tensors='pt')['pixel_values']
        else:
            pixel_values = images
        
        pixel_values = pixel_values.to(self.device)
        
        # Forward through vision encoder with intermediate outputs
        outputs = self.vision_encoder(
            pixel_values=pixel_values,
            output_hidden_states=True  # Get intermediate layers
        )
        
        # Extract features
        last_hidden = outputs.last_hidden_state  # [B, 730, hidden_dim] (729 patches + 1 CLS)
        
        cls_embedding = last_hidden[:, 0, :]  # [B, hidden_dim] - Global scene understanding
        patch_embeddings = last_hidden[:, 1:, :]  # [B, 729, hidden_dim] - Spatial features
        
        # Extract intermediate layer features (multi-scale representations)
        # hidden_states: tuple of (num_layers + 1) tensors
        # We take layers [6, 12, 18, 24] for multi-scale (assuming 24 layers)
        num_layers = len(outputs.hidden_states) - 1  # Exclude embedding layer
        sample_layers = [num_layers // 4, num_layers // 2, 3 * num_layers // 4, num_layers - 1]
        
        intermediate_features = []
        for layer_idx in sample_layers:
            layer_hidden = outputs.hidden_states[layer_idx + 1]  # +1 to skip embedding layer
            layer_patches = layer_hidden[:, 1:, :]  # Remove CLS, keep patches
            intermediate_features.append(layer_patches)
        
        # Stack: [B, 4, 729, hidden_dim]
        intermediate_features = torch.stack(intermediate_features, dim=1)
        
        return patch_embeddings, cls_embedding, intermediate_features


class TeacherTextEncoder:
    """
    PhoBERT-large Text Teacher (307M params)
    
    CRITICAL ROLE CLARIFICATION:
    PhoBERT is a REPRESENTATION teacher, NOT a GENERATIVE teacher!
    
    ✅ What PhoBERT SHOULD teach:
    1. Question embeddings (contextual Vietnamese understanding)
    2. Token-level attention patterns (which words matter)
    3. Question-Answer semantic similarity (cross-modal alignment)
    
    ❌ What PhoBERT CANNOT teach:
    - Answer generation (it's MLM, not seq2seq!)
    - Answer logits distribution (no generative head!)
    - Long-form rationales (not designed for generation!)
    
    Why PhoBERT-large is the RIGHT representation teacher:
    ✅ Vietnamese native (tokenization + morphology correct)
    ✅ Trained on 20GB Vietnamese corpus (news + wiki)
    ✅ 2× student capacity (307M vs ~135M BARTpho encoder)
    ✅ Lightweight (≈2GB VRAM FP16)
    ✅ Contextual embeddings are high-quality
    
    Advantages over student BARTpho-base encoder:
    - Deeper understanding of Vietnamese syntax
    - Better handling of rare words / compound words
    - Stronger semantic representations
    """
    def __init__(self, model_name='vinai/phobert-large', device='cuda'):
        print(f"[Text Teacher] Loading {model_name}...")
        print(f"  📊 Role: REPRESENTATION teacher (NOT generative!)")
        self.device = device
        
        # Load PhoBERT-large in FP16 to save VRAM
        self.model = AutoModelForMaskedLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16  # 🔥 Load in FP16!
        ).to(device)
        self.model.eval()
        
        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Hidden dimension
        self.hidden_dim = self.model.config.hidden_size
        
        # 🔥 Print memory usage
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(device) / 1e9
            reserved = torch.cuda.memory_reserved(device) / 1e9
            print(f"  ✓ Text teacher loaded: {self.hidden_dim}D features")
            print(f"  ✓ GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
        else:
            print(f"  ✓ Text teacher loaded: {self.hidden_dim}D features")
        print(f"  ✓ Will teach: question embeddings + attention patterns + Q-A similarity")
    
    @torch.no_grad()
    def extract_question_embeddings(self, questions):
        """
        Extract question representations from PhoBERT teacher
        
        Args:
            questions: List of Vietnamese question strings
        
        Returns:
            token_embeddings: [B, seq_len, hidden_dim] - Contextual token embeddings
            cls_embedding: [B, hidden_dim] - Question-level representation
            attention_weights: [B, num_heads, seq_len, seq_len] - Token attention patterns
        """
        # Tokenize
        encodings = self.tokenizer(
            questions,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )
        
        input_ids = encodings['input_ids'].to(self.device)
        attention_mask = encodings['attention_mask'].to(self.device)
        
        # Forward through PhoBERT with attention outputs
        outputs = self.model.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            output_attentions=True
        )
        
        # Extract features
        token_embeddings = outputs.last_hidden_state  # [B, seq_len, hidden_dim]
        cls_embedding = token_embeddings[:, 0, :]  # [B, hidden_dim] - CLS token
        
        # Extract attention from last layer (teacher's learned attention pattern)
        # Shape: [B, num_heads, seq_len, seq_len]
        attention_weights = outputs.attentions[-1]
        
        return token_embeddings, cls_embedding, attention_weights
    
    @torch.no_grad()
    def compute_qa_similarity(self, questions, answers):
        """
        Compute semantic similarity between questions and answers
        
        This teaches the student which question-answer pairs are semantically aligned.
        
        Args:
            questions: List of question strings
            answers: List of answer strings
        
        Returns:
            qa_similarity: [B] - Cosine similarity scores between Q and A embeddings
        """
        # Encode questions
        q_encodings = self.tokenizer(
            questions,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )
        q_outputs = self.model.roberta(
            input_ids=q_encodings['input_ids'].to(self.device),
            attention_mask=q_encodings['attention_mask'].to(self.device)
        )
        q_cls = q_outputs.last_hidden_state[:, 0, :]  # [B, hidden_dim]
        
        # Encode answers
        a_encodings = self.tokenizer(
            answers,
            padding=True,
            truncation=True,
            max_length=64,
            return_tensors='pt'
        )
        a_outputs = self.model.roberta(
            input_ids=a_encodings['input_ids'].to(self.device),
            attention_mask=a_encodings['attention_mask'].to(self.device)
        )
        a_cls = a_outputs.last_hidden_state[:, 0, :]  # [B, hidden_dim]
        
        # Compute cosine similarity
        qa_similarity = F.cosine_similarity(q_cls, a_cls, dim=-1)  # [B]
        
        return qa_similarity


def extract_teachers_for_dataset(
    csv_path,
    image_folder,
    output_dir,
    student_checkpoint_path,
    batch_size=8,  # Reduced from 16 to avoid OOM (large teacher models!)
    max_samples=None,
    device='cuda'
):
    """
    Extract teacher features for entire dataset and save to .npy files
    
    CORRECTED APPROACH (VISION + TEXT REPRESENTATIONS ONLY):
    - Vision: SigLIP-SO400M (~400-430M params) for patch embeddings + CLS + intermediate features
    - Text: PhoBERT-large (307M) for question embeddings + attention patterns + Q-A similarity
    - Answer: Learn from ground truth labels (NO answer teacher needed)
    
    Args:
        csv_path: Path to train.csv or val.csv
        image_folder: Path to vivqa/images
        output_dir: Where to save .npy files
        student_checkpoint_path: [UNUSED - kept for compatibility]
        batch_size: Batch size for extraction
        max_samples: Limit samples for testing (None = full dataset)
    """
    print("="*80)
    print("TEACHER REPRESENTATIONS EXTRACTION")
    print("="*80)
    print("Vision: SigLIP-SO400M (~400-430M params)")
    print("Text: PhoBERT-large (307M params, representation teacher)")
    print("Answer: Ground truth labels ONLY (no distillation)")
    print("="*80)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load dataset
    print(f"\n[1/5] Loading dataset: {csv_path}")
    
    # CRITICAL: Use TEACHER's vision processor (SO400M), not student's!
    # Teacher expects 384x384 images (729 patches), student uses 224x224 (256 patches)
    from transformers import AutoProcessor
    vision_processor = AutoProcessor.from_pretrained('google/siglip-so400m-patch14-384')
    print(f"  Using SigLIP-SO400M processor: 384x384 images → 729 patches")
    
    dataset = VQAGenDataset(
        csv_path=csv_path,
        image_folder=image_folder,
        vision_processor=vision_processor,
        include_question_type=False
    )
    
    if max_samples:
        dataset.data = dataset.data[:max_samples]
        print(f"  ⚠️  Limited to {max_samples} samples for testing")
    
    print(f"  ✓ Loaded {len(dataset)} samples")
    
    # Custom collate to get raw text (teachers need text, not tokens!)
    def collate_with_text(batch):
        # Standard collation
        pixel_values = torch.stack([b['pixel_values'] for b in batch])
        input_ids = torch.stack([b['input_ids'] for b in batch])
        attention_mask = torch.stack([b['attention_mask'] for b in batch])
        labels = torch.stack([b['labels'] for b in batch])
        
        # Get raw text from dataset by indices
        # Need to access dataset.data directly
        indices = [i for i in range(len(batch))]  # This won't work - need actual indices!
        
        return {
            'pixel_values': pixel_values,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'batch_indices': indices  # Will get text via dataset.data.iloc[idx]
        }
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # CRITICAL: maintain index order!
        num_workers=0,  # Single process to track indices
        pin_memory=True
    )
    
    # ========================================================================
    # SEQUENTIAL EXTRACTION (to avoid OOM - 2 large models don't fit together!)
    # ========================================================================
    # Strategy: Extract vision features first, then text features
    # This keeps only 1 teacher model on GPU at a time
    
    print(f"\n[2/5] STEP 1: Extracting VISION features...")
    print(f"  (Will extract text features separately to avoid OOM)")
    print(f"  🔥 Using memory-mapped arrays to avoid RAM overflow")
    
    # Initialize vision teacher ONLY
    vision_teacher = TeacherVisionEncoder(device=device)
    
    # Get feature dimensions from first batch
    print(f"  Getting feature dimensions from first sample...")
    first_batch = next(iter(dataloader))
    first_pixel = first_batch['pixel_values'][:1].to(device)
    with torch.no_grad():
        sample_patch, sample_cls, sample_intermediate = vision_teacher.extract_features(first_pixel)
    
    patch_shape = sample_patch.shape[1:]  # (729, hidden)
    cls_shape = sample_cls.shape[1:]      # (hidden,)
    intermediate_shape = sample_intermediate.shape[1:]  # (4, 729, hidden)
    
    del first_batch, first_pixel, sample_patch, sample_cls, sample_intermediate
    torch.cuda.empty_cache()
    
    # Create memory-mapped arrays (can be larger than RAM!)
    num_samples = len(dataset)
    split_name = Path(csv_path).stem
    
    print(f"  Creating memory-mapped arrays for {num_samples} samples...")
    vision_patch_mmap = np.lib.format.open_memmap(
        f"{output_dir}/vision_patch_emb_{split_name}.npy",
        mode='w+',
        dtype=np.float32,
        shape=(num_samples, *patch_shape)
    )
    vision_cls_mmap = np.lib.format.open_memmap(
        f"{output_dir}/vision_cls_emb_{split_name}.npy",
        mode='w+',
        dtype=np.float32,
        shape=(num_samples, *cls_shape)
    )
    vision_intermediate_mmap = np.lib.format.open_memmap(
        f"{output_dir}/vision_intermediate_{split_name}.npy",
        mode='w+',
        dtype=np.float32,
        shape=(num_samples, *intermediate_shape)
    )
    
    # Extract vision features and write directly to memmap
    global_idx = 0
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Vision")):
        pixel_values = batch['pixel_values'].to(device)
        batch_size_actual = pixel_values.shape[0]
        
        # Extract vision features
        patch_emb, cls_emb, intermediate_feats = vision_teacher.extract_features(pixel_values)
        
        # Write to memmap arrays (by index, not batch!)
        start_idx = global_idx
        end_idx = global_idx + batch_size_actual
        
        vision_patch_mmap[start_idx:end_idx] = patch_emb.cpu().numpy()
        vision_cls_mmap[start_idx:end_idx] = cls_emb.cpu().numpy()
        vision_intermediate_mmap[start_idx:end_idx] = intermediate_feats.cpu().numpy()
        
        global_idx += batch_size_actual
        
        # Free GPU memory after each batch
        del pixel_values, patch_emb, cls_emb, intermediate_feats
        if batch_idx % 10 == 0:
            torch.cuda.empty_cache()
    
    # Flush memmap to disk
    vision_patch_mmap.flush()
    vision_cls_mmap.flush()
    vision_intermediate_mmap.flush()
    
    del vision_patch_mmap, vision_cls_mmap, vision_intermediate_mmap
    
    # Free vision teacher from GPU
    del vision_teacher
    torch.cuda.empty_cache()
    print(f"  ✓ Vision extraction complete. GPU memory freed.")
    
    # ========================================================================
    # STEP 2: Extract TEXT features
    # ========================================================================
    
    print(f"\n[3/5] STEP 2: Extracting TEXT features...")
    print(f"  🔥 Using memory-mapped arrays to avoid RAM overflow")
    
    # Initialize text teacher ONLY
    text_teacher = TeacherTextEncoder(device=device)
    
    # Get feature dimensions from first sample
    print(f"  Getting feature dimensions from first sample...")
    first_questions = [dataset.data.iloc[0]['question']]
    first_answers = [dataset.data.iloc[0]['answer']]
    
    with torch.no_grad():
        sample_token, sample_cls, sample_attn = text_teacher.extract_question_embeddings(first_questions)
        sample_qa_sim = text_teacher.compute_qa_similarity(first_questions, first_answers)
    
    token_shape = sample_token.shape[1:]  # (seq, 1024)
    cls_shape = sample_cls.shape[1:]      # (1024,)
    attn_shape = sample_attn.shape[1:]    # (heads, seq, seq)
    qa_sim_shape = sample_qa_sim.shape[1:] if len(sample_qa_sim.shape) > 1 else ()  # scalar or shape
    
    del sample_token, sample_cls, sample_attn, sample_qa_sim
    torch.cuda.empty_cache()
    
    # Create memory-mapped arrays for text features
    print(f"  Creating memory-mapped arrays for {num_samples} samples...")
    text_token_mmap = np.lib.format.open_memmap(
        f"{output_dir}/text_token_emb_{split_name}.npy",
        mode='w+',
        dtype=np.float32,
        shape=(num_samples, *token_shape)
    )
    text_cls_mmap = np.lib.format.open_memmap(
        f"{output_dir}/text_cls_emb_{split_name}.npy",
        mode='w+',
        dtype=np.float32,
        shape=(num_samples, *cls_shape)
    )
    text_attention_mmap = np.lib.format.open_memmap(
        f"{output_dir}/text_attention_{split_name}.npy",
        mode='w+',
        dtype=np.float32,
        shape=(num_samples, *attn_shape)
    )
    text_qa_sim_mmap = np.lib.format.open_memmap(
        f"{output_dir}/text_qa_similarity_{split_name}.npy",
        mode='w+',
        dtype=np.float32,
        shape=(num_samples,) if qa_sim_shape == () else (num_samples, *qa_sim_shape)
    )
    
    # Extract text features and write directly to memmap
    global_idx = 0
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Text")):
        # Get raw text from dataset.data (teachers need text, not tokens!)
        batch_size_actual = batch['pixel_values'].shape[0]
        batch_indices = range(global_idx, global_idx + batch_size_actual)
        questions = [dataset.data.iloc[i]['question'] for i in batch_indices]
        answers = [dataset.data.iloc[i]['answer'] for i in batch_indices]
        
        # Extract text features
        token_emb, text_cls, attention_weights = text_teacher.extract_question_embeddings(questions)
        qa_similarity = text_teacher.compute_qa_similarity(questions, answers)
        
        # Write to memmap arrays
        start_idx = global_idx
        end_idx = global_idx + batch_size_actual
        
        text_token_mmap[start_idx:end_idx] = token_emb.cpu().numpy()
        text_cls_mmap[start_idx:end_idx] = text_cls.cpu().numpy()
        text_attention_mmap[start_idx:end_idx] = attention_weights.cpu().numpy()
        text_qa_sim_mmap[start_idx:end_idx] = qa_similarity.cpu().numpy()
        
        global_idx += batch_size_actual
        
        # Free GPU memory after each batch
        del token_emb, text_cls, attention_weights, qa_similarity
        if batch_idx % 10 == 0:
            torch.cuda.empty_cache()
    
    # Flush memmap to disk
    text_token_mmap.flush()
    text_cls_mmap.flush()
    text_attention_mmap.flush()
    text_qa_sim_mmap.flush()
    
    del text_token_mmap, text_cls_mmap, text_attention_mmap, text_qa_sim_mmap
    
    # Free text teacher from GPU
    del text_teacher
    torch.cuda.empty_cache()
    print(f"  ✓ Text extraction complete. GPU memory freed.")
    
    # ========================================================================
    # STEP 3: Save metadata
    # ========================================================================
    
    print(f"\n[4/5] Creating metadata...")
    
    # Load one sample to get final shapes
    vision_patch_emb = np.load(f"{output_dir}/vision_patch_emb_{split_name}.npy", mmap_mode='r')
    vision_cls_emb = np.load(f"{output_dir}/vision_cls_emb_{split_name}.npy", mmap_mode='r')
    vision_intermediate = np.load(f"{output_dir}/vision_intermediate_{split_name}.npy", mmap_mode='r')
    text_token_emb = np.load(f"{output_dir}/text_token_emb_{split_name}.npy", mmap_mode='r')
    text_cls_emb = np.load(f"{output_dir}/text_cls_emb_{split_name}.npy", mmap_mode='r')
    text_attention = np.load(f"{output_dir}/text_attention_{split_name}.npy", mmap_mode='r')
    text_qa_similarity = np.load(f"{output_dir}/text_qa_similarity_{split_name}.npy", mmap_mode='r')
    
    # Save metadata
    metadata = {
        'num_samples': len(vision_patch_emb),
        'vision_patch_emb_shape': vision_patch_emb.shape,
        'vision_cls_emb_shape': vision_cls_emb.shape,
        'vision_intermediate_shape': vision_intermediate.shape,
        'text_token_emb_shape': text_token_emb.shape,
        'text_cls_emb_shape': text_cls_emb.shape,
        'text_attention_shape': text_attention.shape,
        'text_qa_similarity_shape': text_qa_similarity.shape,
        'vision_teacher': 'google/siglip-so400m-patch14-384 (~400-430M params)',
        'text_teacher': 'vinai/phobert-large (307M params, representation teacher)',
        'answer_supervision': 'Ground truth labels ONLY (no distillation)',
        'csv_path': csv_path
    }
    
    with open(f"{output_dir}/metadata_{split_name}.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"✅ EXTRACTION COMPLETE!")
    print(f"{'='*80}")
    print(f"Samples: {len(vision_patch_emb)}")
    print(f"\nVision outputs:")
    print(f"  - Patch embeddings: {vision_patch_emb.shape} ({vision_patch_emb.nbytes/1e9:.2f} GB)")
    print(f"  - CLS embedding: {vision_cls_emb.shape} ({vision_cls_emb.nbytes/1e6:.2f} MB)")
    print(f"  - Intermediate features: {vision_intermediate.shape} ({vision_intermediate.nbytes/1e9:.2f} GB)")
    print(f"\nText outputs:")
    print(f"  - Token embeddings: {text_token_emb.shape} ({text_token_emb.nbytes/1e9:.2f} GB)")
    print(f"  - CLS embedding: {text_cls_emb.shape} ({text_cls_emb.nbytes/1e6:.2f} MB)")
    print(f"  - Attention patterns: {text_attention.shape} ({text_attention.nbytes/1e6:.2f} MB)")
    print(f"  - Q-A similarity: {text_qa_similarity.shape} ({text_qa_similarity.nbytes/1e6:.2f} MB)")
    
    total_size = (
        vision_patch_emb.nbytes + vision_cls_emb.nbytes + vision_intermediate.nbytes +
        text_token_emb.nbytes + text_cls_emb.nbytes + text_attention.nbytes + text_qa_similarity.nbytes
    )
    print(f"\n💾 Total storage: {total_size/1e9:.2f} GB")
    print(f"📁 Saved to: {output_dir}")
    print(f"{'='*80}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Extract teacher representations for offline distillation (Vision + Text ONLY)'
    )
    
    parser.add_argument('--csv_path', type=str, required=True,
                       help='Path to train.csv or val.csv')
    parser.add_argument('--image_folder', type=str, required=True,
                       help='Path to vivqa/images folder')
    parser.add_argument('--output_dir', type=str, default='/kaggle/working/teacher_cache',
                       help='Output directory for .npy files')
    parser.add_argument('--student_checkpoint', type=str, default=None,
                       help='[UNUSED - kept for compatibility]')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for extraction (default: 8, lower if OOM on smaller GPUs)')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Limit samples for testing (None = full dataset)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda or cpu)')
    
    args = parser.parse_args()
    
    # Run extraction
    extract_teachers_for_dataset(
        csv_path=args.csv_path,
        image_folder=args.image_folder,
        output_dir=args.output_dir,
        student_checkpoint_path=args.student_checkpoint,  # Ignored
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        device=args.device
    )
    
    print("\n" + "="*80)
    print("NEXT STEPS:")
    print("="*80)
    print("1. Extract validation set:")
    print(f"   python extract_teacher_logits.py \\")
    print(f"       --csv_path OpenViVQA/dev.json \\")
    print(f"       --image_folder vivqa/images \\")
    print(f"       --output_dir {args.output_dir}")
    print()
    print("2. Implement training code with vision + text distillation:")
    print("   - Add TeacherDistillationDataset loading vision+text .npy files")
    print("   - Add distillation losses (vision KD + text KD ONLY)")
    print("   - Answer generation learns from ground truth labels!")
    print("   - Weight: 0.3*vision_kd + 0.3*text_kd + 0.4*answer_kd + 1.0*ce_loss")
    print()
    print("3. Train with distillation:")
    print("   python train_no_latent.py \\")
    print("       --use_teacher_distillation \\")
    print(f"       --teacher_cache_dir {args.output_dir} \\")
    print("       --distill_alpha 0.5 \\")
    print("       --text_lora_r 96")
    print("="*80)
