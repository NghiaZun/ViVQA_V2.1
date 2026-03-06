import torch
from torch.utils.data import Dataset, Sampler
from transformers import BartphoTokenizer
from PIL import Image
import pandas as pd
import os
import re
import random
from collections import defaultdict
from typing import Iterator, List, Optional


def detect_question_type(question: str) -> int:
    """
    Detect Vietnamese VQA question type from question text.

    Types:
        0 = OBJECT   (Đây là gì? Cái gì? Ai? Con gì?)
        1 = COUNT    (Có bao nhiêu? Mấy cái?)
        2 = COLOR    (Màu gì? Màu sắc?)
        3 = LOCATION (Ở đâu? Phía nào? Đặt/để/nằm ở chỗ nào?)

    Design notes (v2 — fixed OBJECT→LOCATION confusion):
    -------------------------------------------------------
    Old rule triggered LOCATION on ANY sentence containing
    trên/trong/dưới/bên/giữa/ngoài — these are common Vietnamese
    prepositions that appear in OBJECT questions too, e.g.:
        "cái gì đang đỗ TRÊN ổ đĩa"  → was wrongly LOCATION
        "những gì đang đi TRÊN đường ray" → was wrongly LOCATION

    Fix: LOCATION requires an EXPLICIT location-question marker:
      • "ở đâu" / "ở nào" / "từ đâu"  — direct where-question
      • "đặt ở" / "để ở" / "nằm ở" / "đứng ở" / "ngồi ở" — verb+ở
      • "phía nào" / "phía trước/sau"  — directional question
      • "vị trí"                        — position question
      • bare "đâu" / "nơi nào" / "chỗ nào" — interrogative place

    Bare prepositions (trên/trong/dưới/bên/giữa/ngoài/trái/phải)
    are NOT counted unless attached to ở/đặt/để/nằm patterns above.
    """
    q_lower = question.lower().strip()

    # ── 1. COUNT: "bao nhiêu", "mấy", "số lượng" ────────────────────────────
    # Check first — "bao nhiêu màu" should be COUNT not COLOR
    if re.search(r'(bao nhiêu|mấy\b|số lượng)', q_lower):
        return 1

    # ── 2. LOCATION: explicit where-question markers only ────────────────────
    # Must come BEFORE color check: "kéo màu xanh để ở đâu" → LOCATION not COLOR
    #
    # Strong markers: interrogative words that directly ask "where"
    _loc_strong = r'(ở\s*(đâu|nào|chỗ\s*nào)|từ\s*đâu|đâu\b|nơi\s*nào|chỗ\s*nào|vị\s*trí)'
    # Verb+ở patterns: "đặt ở", "để ở", "nằm ở", "đứng ở", "treo ở", "gắn ở"
    _loc_verb   = r'(đặt|để|nằm|đứng|ngồi|treo|gắn|đỗ)\s*ở'
    # Directional question: "phía nào", "hướng nào"
    _loc_dir    = r'(phía\s*(nào|trước|sau|đông|tây|nam|bắc)|hướng\s*nào)'

    if re.search(f'({_loc_strong}|{_loc_verb}|{_loc_dir})', q_lower):
        return 3

    # ── 3. COLOR: question is specifically ASKING about color ────────────────
    # Require interrogative color form, NOT bare "màu" as noun/adjective modifier
    # e.g. "màu gì", "màu sắc", "màu nào", "có màu gì" → COLOR
    # but "cái gì màu nâu" / "túi màu đỏ ... gì" → OBJECT (màu is just a descriptor)
    #
    # Rule: COLOR only when "màu" is followed by an interrogative (gì/sắc/nào/gì không)
    # OR the question starts with "màu" as the topic
    if re.search(r'màu\s*(gì|sắc|nào|như\s*thế\s*nào)', q_lower):
        return 2
    # Also catch "có màu gì", "là màu gì" patterns
    if re.search(r'(có|là|được)\s*màu\s*(gì|nào|sắc)', q_lower):
        return 2

    # ── 4. Default: OBJECT ───────────────────────────────────────────────────
    return 0


class VQAGenDataset(Dataset):
    def __init__(self, csv_path, image_folder,
                 vision_processor,
                 tokenizer_name='vinai/bartpho-syllable',
                 max_q_len=32, max_a_len=10,
                 include_question_type=False,  # 🔥 Enable question type
                 auto_detect_type=False,  # 🔥 NEW: Auto-detect from question text
                 use_distillation=False,  # 🔥🔥🔥 Enable teacher inputs
                 teacher_vision_processor=None):  # 🔥🔥🔥 Teacher's processor (384px)

        self.data = pd.read_csv(csv_path)
        self.image_folder = image_folder
        self.vision_processor = vision_processor
        # Dùng BARTpho tokenizer cho cả question và answer
        self.tokenizer = BartphoTokenizer.from_pretrained(tokenizer_name)
        self.max_q_len = max_q_len
        self.max_a_len = max_a_len
        self.include_question_type = include_question_type
        self.auto_detect_type = auto_detect_type  # 🔥 NEW
        self.use_distillation = use_distillation  # 🔥🔥🔥
        self.teacher_vision_processor = teacher_vision_processor  # 🔥🔥🔥

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        question, answer, img_id = row['question'], row['answer'], str(row['img_id'])

        # Load image
        img_path = os.path.join(self.image_folder, f"{img_id}.jpg")
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"[Warning] Failed to load image: {img_path} - {e}")
            image = Image.new('RGB', (224, 224), color='white')

        vision_inputs = self.vision_processor(images=image, return_tensors='pt')
        pixel_values = vision_inputs['pixel_values'].squeeze(0)  # (3, H, W)

        # Tokenize question (BARTpho)
        q_enc = self.tokenizer(question,
                              truncation=True,
                              padding='max_length',
                              max_length=self.max_q_len,
                              return_tensors='pt')

        input_ids = q_enc['input_ids'].squeeze(0)
        attention_mask = q_enc['attention_mask'].squeeze(0)

        # Tokenize answer (BARTpho)
        a_enc = self.tokenizer(answer,
                              truncation=True,
                              padding='max_length',
                              max_length=self.max_a_len,
                              return_tensors='pt')

        labels = a_enc['input_ids'].squeeze(0)
        labels[labels == self.tokenizer.pad_token_id] = -100  # important for loss masking

        # 🔥 Return dict format with optional question_type from CSV
        result = {
            'pixel_values': pixel_values,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }
        
        # 🔥🔥🔥 Add teacher inputs for online distillation
        if self.use_distillation and self.teacher_vision_processor is not None:
            # Process same image at 384px for vision teacher
            teacher_vision_inputs = self.teacher_vision_processor(images=image, return_tensors='pt')
            result['images_384'] = teacher_vision_inputs['pixel_values'].squeeze(0)  # [3, 384, 384]
            
            # Raw question string for text teacher
            result['raw_question'] = question
        
        # 🔥 Get question type
        if self.include_question_type:
            if self.auto_detect_type:
                # Auto-detect from question text (fallback if CSV doesn't have type column)
                question_type = detect_question_type(question)
            elif 'type' in row:
                question_type = int(row['type'])
            elif 'question_type' in row:
                question_type = int(row['question_type'])
            else:
                # Fallback: auto-detect if no CSV column
                question_type = detect_question_type(question)
            
            result['question_type'] = question_type
        
        return result


# ============================================================================
# PK SAMPLER
# ============================================================================

class PKSampler(Sampler):
    """
    PK Sampling cho VQA dataset với 4 question types.

    Cách hoạt động:
        Mỗi batch = P types × K samples/type  →  batch_size = P × K

        Ví dụ P=4, K=8 → batch=32:
            TYPE_0 (OBJECT):   idx1 idx2 ... idx8
            TYPE_1 (COUNT):    idx1 idx2 ... idx8
            TYPE_2 (COLOR):    idx1 idx2 ... idx8
            TYPE_3 (LOCATION): idx1 idx2 ... idx8

    Lợi ích:
        - Mỗi batch luôn có đủ diversity về question type
        - use_type_loss nhận gradient từ đủ 4 class mỗi step
        - Tránh batch toàn COUNT hoặc toàn COLOR

    Args:
        dataset:      VQAGenDataset (hoặc Subset của nó)
        p:            Số lượng types per batch. Default 4 (dùng hết 4 types)
        k:            Số samples per type per batch. Default 8
        shuffle:      Shuffle trong từng type pool mỗi epoch. Default True
        drop_last:    Bỏ batch cuối nếu không đủ P × K. Default True

    Cách xác định type:
        1. Nếu dataset.data có cột 'type' → dùng trực tiếp (nhanh, chính xác)
        2. Nếu không → auto-detect từ question text qua detect_question_type()
    """

    TYPE_NAMES = {0: 'OBJECT', 1: 'COUNT', 2: 'COLOR', 3: 'LOCATION'}

    def __init__(
        self,
        dataset,
        p: int = 4,
        k: int = 8,
        shuffle: bool = True,
        drop_last: bool = True
    ):
        super().__init__(dataset)
        self.p = p
        self.k = k
        self.shuffle = shuffle
        self.drop_last = drop_last

        # Resolve inner dataset (handles torch.utils.data.Subset)
        from torch.utils.data import Subset
        if isinstance(dataset, Subset):
            inner = dataset.dataset
            indices = list(dataset.indices)
        else:
            inner = dataset
            indices = list(range(len(dataset)))

        # ── Build type → [idx] mapping ───────────────────────────────────
        self.type_to_indices: dict[int, List[int]] = defaultdict(list)

        has_type_col = (
            hasattr(inner, 'data')
            and isinstance(inner.data, pd.DataFrame)
            and ('type' in inner.data.columns or 'question_type' in inner.data.columns)
        )
        type_col = 'type' if (has_type_col and 'type' in inner.data.columns) else 'question_type'

        for pos, real_idx in enumerate(indices):
            if has_type_col:
                t = int(inner.data.iloc[real_idx][type_col])
            else:
                # Fallback: auto-detect from question text
                row = inner.data.iloc[real_idx]
                t = detect_question_type(str(row['question']))
            self.type_to_indices[t].append(pos)  # pos = index in this dataset view

        # Print distribution summary
        print("[PKSampler] Type distribution:")
        for t in sorted(self.type_to_indices):
            n = len(self.type_to_indices[t])
            print(f"  Type {t} ({self.TYPE_NAMES.get(t, '?')}): {n:,} samples")

        # Warn if a type has fewer samples than K
        for t, idxs in self.type_to_indices.items():
            if len(idxs) < k:
                print(f"  ⚠️  Type {t} ({self.TYPE_NAMES.get(t, '?')}) has only {len(idxs)} samples "
                      f"< K={k}. Will sample with replacement.")

        # Determine which P types to use (sorted by type id, take first p)
        available_types = sorted(self.type_to_indices.keys())
        if len(available_types) < p:
            print(f"  ⚠️  Only {len(available_types)} types available, reducing P={p} → {len(available_types)}")
            self.p = len(available_types)
        self.active_types: List[int] = available_types[:self.p]

        # Compute total batches
        min_type_size = min(len(self.type_to_indices[t]) for t in self.active_types)
        self._num_batches = min_type_size // k
        if not drop_last and min_type_size % k != 0:
            self._num_batches += 1

        print(f"[PKSampler] Config: P={self.p}, K={k}, batch_size={self.p * k}")
        print(f"[PKSampler] Batches per epoch: {self._num_batches} "
              f"({'drop_last' if drop_last else 'keep_last'})")

    # ── Sampler interface ─────────────────────────────────────────────────

    def __len__(self) -> int:
        """Total number of samples yielded per epoch."""
        return self._num_batches * self.p * self.k

    def __iter__(self) -> Iterator[int]:
        """Yield indices forming P×K structured batches."""
        # Shuffle each type pool
        pools: dict[int, List[int]] = {}
        for t in self.active_types:
            idxs = list(self.type_to_indices[t])
            if self.shuffle:
                random.shuffle(idxs)
            # If fewer than needed, sample with replacement to fill
            needed = self._num_batches * self.k
            if len(idxs) < needed:
                extra = random.choices(idxs, k=needed - len(idxs))
                idxs = idxs + extra
            pools[t] = idxs

        # Yield interleaved: for each batch position, yield K from each type
        for batch_i in range(self._num_batches):
            start = batch_i * self.k
            end = start + self.k
            for t in self.active_types:
                yield from pools[t][start:end]