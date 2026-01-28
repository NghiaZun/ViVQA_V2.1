# 🔥 TYPE-CONDITIONED MULTI-TASK VQA ARCHITECTURE

## ✅ ĐÃ IMPLEMENT (Theo tinh thần Viblo article)

Kiến trúc **SOFT MULTI-TASK + TYPE-CONDITIONED GENERATION**

---

## 🧠 TRIẾT LÝ THIẾT KẾ

### ❌ KHÔNG làm (Hard Pipeline)
```
Question → [IF type==COLOR] → Color vocab only
         → [IF type==COUNT] → Number vocab only
         → Hard decision, brittle
```

### ✅ LÀM (Soft Multi-task)
```
Question → Type head (auxiliary signal)
         ↓
    Type embedding
         ↓
(Type + Question) → Gate vision patches (soft attention)
         ↓
Type-conditioned features → Decoder
         ↓
Type bias logits (soft reweighting, not masking!)
```

---

## 🔥 KIẾN TRÚC CHI TIẾT

### 1️⃣ Type Prediction Head (Auxiliary Task)

**Mục đích:** Bắt question encoder học khái niệm "type" (nhưng không dùng để quyết định cứng)

```python
class TypePredictionHead(nn.Module):
    """
    Types:
        0 = OBJECT (Đây là gì?)
        1 = COUNT (Có bao nhiêu?)
        2 = COLOR (Màu gì?)
        3 = LOCATION (Ở đâu?)
    """
    def forward(self, text_cls):
        # text_cls: [B, D] - CLS token from question
        return self.classifier(text_cls)  # [B, 4]
```

**Loss:**
```python
type_loss = CrossEntropy(type_logits, ground_truth_types)
# λ = 0.2 → auxiliary signal, không dominates
total_loss = answer_loss + 0.2 * type_loss
```

---

### 2️⃣ Type-Conditioned Vision Gating

**Key Idea:** Question types cần vision features khác nhau

| Type | Vision Focus |
|------|-------------|
| COLOR | Color-rich patches |
| COUNT | Object distribution (global) |
| LOCATION | Spatial arrangement |
| OBJECT | Salient regions |

**Implementation:**
```python
class VisionGating(nn.Module):
    def __init__(self, hidden_dim, num_types=4):
        self.type_embedding = nn.Embedding(num_types, hidden_dim)
        self.query_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.gate_net = ...  # Attention mechanism
    
    def forward(self, vision, text, type_ids):
        # 1. Embed type
        type_emb = self.type_embedding(type_ids)  # [B, D]
        
        # 2. Combine (question_cls + type)
        query = concat([text_cls, type_emb])  # [B, 2D]
        query = self.query_proj(query)  # [B, D]
        
        # 3. Attention: query @ vision → α (importance)
        alpha = gate_net(concat([vision, query]))  # [B, P]
        
        # 4. Gated vision
        gated = alpha * vision + (1-alpha) * text_context
        return gated
```

**Hiệu ứng:**
- COUNT question → α cao ở patches chứa objects
- COLOR question → α cao ở patches có màu nổi
- LOCATION question → α cao ở patches không gian

---

### 3️⃣ Type-Aware Logits Biasing

**Mục đích:** Hướng decoder về answer space hợp lý (SOFT, không chặn!)

```python
class TypeAwareLogitsBias(nn.Module):
    def __init__(self, vocab_size, num_types=4):
        # Learnable bias per type: [num_types, vocab_size]
        self.type_biases = nn.Parameter(
            torch.randn(num_types, vocab_size) * 0.1
        )
    
    def forward(self, base_logits, type_ids):
        bias = self.type_biases[type_ids]  # [B, vocab_size]
        return base_logits + bias  # SOFT reweighting!
```

**Ví dụ:**
- Type=COLOR → `type_biases[2]` boost logits của `["đỏ", "xanh", "vàng", ...]`
- Type=COUNT → `type_biases[1]` boost logits của `["một", "hai", "1", "2", ...]`
- Type=LOCATION → `type_biases[3]` boost logits của `["trên", "dưới", "trái", "phải", ...]`

**⚠️ QUAN TRỌNG:**
- Tokens ngoài preferred vocab **VẪN CÓ xác suất**
- Chỉ thấp hơn, không bị mask hoàn toàn
- Model vẫn tự do sinh, không bị đóng khung

---

## 🔄 LUỒNG FORWARD PASS

```python
def forward(pixel_values, input_ids, labels, question_types):
    # 1. Encode question
    text_features = text_encoder(input_ids)
    text_cls = text_features[:, 0, :]
    
    # 2. 🔥 Type prediction (auxiliary)
    type_logits = type_head(text_cls)
    type_loss = CE(type_logits, question_types)
    
    # 3. Encode vision
    vision_features = vision_encoder(pixel_values)
    
    # 4. Fusion (Flamingo cross-attention)
    fused_vision = flamingo_fusion(vision_features, text_features)
    
    # 5. 🔥 Type-conditioned vision gating
    gated_vision = vision_gating(
        fused_vision, 
        text_features,
        type_ids=question_types  # Use GT during training
    )
    
    # 6. Decoder
    decoder_out = decoder(
        encoder_hidden_states=concat([gated_vision, text_features])
    )
    
    # 7. Generate logits
    base_logits = lm_head(decoder_out)
    
    # 8. 🔥 Type-aware biasing
    answer_logits = logits_bias(base_logits, question_types)
    
    # 9. 🔥 Multi-task loss
    answer_loss = CE(answer_logits, labels)
    total_loss = answer_loss + 0.2 * type_loss
    
    return answer_logits, total_loss, type_loss
```

---

## 🎯 TẠI SAO KIẾN TRÚC NÀY ĐÚNG?

### ✅ Giữ được generative nature
- Model vẫn sinh tự do, không bị rule-based answer
- Type chỉ là **soft signal**, không phải hard decision

### ✅ Vision thực sự quyết định nội dung
- Gating theo type → các type nhìn patches khác nhau
- COUNT nhìn distribution, COLOR nhìn màu, LOCATION nhìn không gian

### ✅ Question chỉ quyết định "cách nhìn"
- Question text → type → cách select vision
- Vision content → answer details

### ✅ Không hack, không cheat
- Không if-else trong code
- Không hard mask vocab
- Mọi thứ learnable, differentiable

### ✅ Đúng tinh thần multi-task learning hiện đại
- Auxiliary type loss giúp question encoder học pattern
- Type embedding giúp vision gating adapt
- Logits bias giúp decoder focus answer space
- Nhưng tất cả đều SOFT, model vẫn có quyền tự do!

---

## 📊 TRAINING

### Dataset
```python
# Auto-detect type from Vietnamese patterns
def detect_question_type(question: str) -> int:
    if re.search(r'màu\s*(gì|sắc)', question):
        return 2  # COLOR
    if re.search(r'(bao nhiêu|mấy)', question):
        return 1  # COUNT
    if re.search(r'(ở\s*đâu|trên|dưới)', question):
        return 3  # LOCATION
    return 0  # OBJECT (default)

dataset = VQAGenDataset(
    csv_path='train.csv',
    include_question_type=True,
    auto_detect_type=True  # Auto-detect from question
)
```

### Command
```bash
python train_no_latent.py \
    --train_csv train.csv \
    --image_dir images/ \
    --use_vision_gate \       # Enable type-conditioned gating
    --use_type_loss \         # Enable multi-task type loss
    --epochs 50 \
    --batch_size 32 \
    --lr 2e-4
```

### Expected Metrics
```
Epoch 1: loss=2.45 ans=2.30 type=0.15 α_mean=0.65
Epoch 10: loss=1.20 ans=1.10 type=0.10 α_mean=0.72
Epoch 50: loss=0.45 ans=0.42 type=0.03 α_mean=0.80
```

**Type loss giảm nhanh:** Question encoder học type pattern
**α_mean tăng:** Vision gating học select patches quan trọng
**Answer loss giảm ổn định:** Multi-task learning work!

---

## 🔬 SO SÁNH VỚI CÁCH CŨ

### ❌ Cách cũ (Hard Pipeline)
```python
if type == COLOR:
    vocab_mask = color_vocab_only
elif type == COUNT:
    vocab_mask = number_vocab_only
answer = generate(logits * vocab_mask)
```

**Vấn đề:**
- ❌ Type prediction sai → toàn bộ answer sai
- ❌ Không học được từ gradient (discrete decision)
- ❌ Rigid, không generalize

### ✅ Cách mới (Soft Multi-task)
```python
type_emb = embed(predicted_type)  # Differentiable
vision = gate(vision, type_emb)    # Soft attention
logits = decoder(vision) + type_bias  # Soft reweighting
```

**Ưu điểm:**
- ✅ Type prediction sai → vẫn có gradient flow
- ✅ Soft signal → model tự học balance
- ✅ Flexible, generalize tốt hơn

---

## 🚀 NEXT STEPS

1. **Train với type-conditioned architecture:**
   ```bash
   python train_no_latent.py --use_vision_gate --use_type_loss
   ```

2. **Monitor metrics:**
   - Type accuracy (auxiliary task)
   - α_mean, α_std (vision gating behavior)
   - Answer EM/F1 (main task)

3. **Analyze learned biases:**
   ```python
   # Check what tokens are boosted per type
   type_biases = model.logits_bias.type_biases
   top_tokens_per_type = torch.topk(type_biases, k=20, dim=-1)
   ```

4. **Visualize attention maps:**
   ```python
   # See which patches are attended for each type
   gate_values = model.vision_gating(...)
   # gate_values: [B, 256] - importance per patch
   ```

---

## ✨ TÓM TẮT

**Kiến trúc này implement đúng tinh thần Viblo article:**

1. ✅ **Multi-task soft:** Type là auxiliary signal, không phải hard decision
2. ✅ **Type-conditioned generation:** Type → gate vision → bias logits (all soft!)
3. ✅ **Không hack:** Mọi thứ learnable, differentiable, no if-else
4. ✅ **Vision-driven:** Vision content quyết định answer, question quyết định "cách nhìn"

**Đúng câu bạn hỏi:**
> "question → type, mỗi type thì gated vision vào question phải không?"

👉 **ĐÚNG!** Question → learn type → (type + question) → gate vision → answer

Không phải: question → type → rule-based answer ❌
