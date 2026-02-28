import torch
from torch.utils.data import Dataset
from transformers import BartphoTokenizer
from PIL import Image
import pandas as pd
import os
import re


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