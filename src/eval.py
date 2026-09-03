"""
MINIMAL EVAL FOR SIGLIP - KAGGLE COMPATIBLE
"""
import os
import unicodedata
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
from collections import Counter, defaultdict

from dataset import VQAGenDataset
from model import DeterministicVQA
from dataset import detect_question_type as _detect_type_int


# Map integer type → string label for display
_TYPE_NAMES = {0: 'OBJECT', 1: 'COUNT', 2: 'COLOR', 3: 'LOCATION'}


_SYNONYM_MAP = {
    # --- existing ---
    'ngựa rằn'         : 'ngựa vằn',
    'tủ đá'            : 'tủ lạnh',
    'máy tính'         : 'laptop',
    'máy vi tính'      : 'laptop',
    'vali'             : 'hành lý',
    # --- label noise / dataset inconsistency ---
    'hươu cao cổ khắc' : 'hươu cao cổ',   # 9x "khắc" thừa trong label
    'màu nâua'         : 'màu nâu',        # 2x typo
    'màu nâu gấu'      : 'màu nâu',        # 2x label noise
    # --- Vietnamese synonyms ---
    'nhà vệ sinh'      : 'phòng tắm',      # 9x toilet=bathroom
    'đĩa'              : 'đĩa ăn',         # 3x plate=dish
    'tủ đông'          : 'tủ lạnh',        # 3x freezer=fridge
    'đường phố'        : 'đường',          # 2x street=road
    'đường bộ'         : 'đường',          # 2x road=road
    # --- v3 additions (confirmed from run25 error analysis) ---
    'nón'              : 'mũ',             # 3x headwear synonyms
    'bữa ăn tối'       : 'bữa ăn',        # 2x dinner≈meal in VQA context
    'cửa tiệm'         : 'cửa hàng',       # same meaning: store/shop
    'bữa trưa'         : 'bữa ăn',        # 1x lunch≈meal
}


def _normalize_vn(text: str, use_synonyms: bool = False) -> str:
    """NFC normalization cho tiếng Việt — tránh false negative do byte khác nhau."""
    t = unicodedata.normalize('NFC', text).strip().lower()
    if use_synonyms:
        t = _SYNONYM_MAP.get(t, t)
    return t


def _decode_gt(tokenizer, label_token_ids: list) -> str:
    """Decode ground-truth labels, filtering BOS token manually.
    BARTpho's BOS is not in all_special_ids so skip_special_tokens won't remove it."""
    ids = [t for t in label_token_ids if t != tokenizer.bos_token_id]
    return tokenizer.decode(ids, skip_special_tokens=True).strip()


def compute_exact_match(prediction: str, ground_truth: str, use_synonyms: bool = False) -> float:
    return 1.0 if _normalize_vn(prediction, use_synonyms) == _normalize_vn(ground_truth, use_synonyms) else 0.0


def compute_prf(prediction: str, ground_truth: str, use_synonyms: bool = False,
                token_mode: str = 'set') -> tuple:
    """Precision / Recall / F1 muc TUNG CAU (macro-average o ngoai).

    token_mode:
      'set'      — trung token theo TAP HOP: |P n G| / |P|, |P n G| / |G|. Cong thuc trong paper.
      'multiset' — kieu SQuAD (Counter, dem ca token lap). Hanh vi CU cua file nay.
    DO 2026-08-15: hai che do cho ket qua Y HET tren ViVQA (0/3001 dong khac nhau), vi khong
    dap an nao co token lap (0/328). Nen doi mac dinh sang 'set' KHONG lam doi bat ky so nao
    da bao cao. Van giu tuy chon de kiem lai duoc.

    Bien: ca hai rong -> (1,1,1); mot ben rong -> (0,0,0); khong giao -> (0,0,0).
    """
    pred_tokens = _normalize_vn(prediction, use_synonyms).split()
    gt_tokens   = _normalize_vn(ground_truth, use_synonyms).split()

    if len(pred_tokens) == 0 and len(gt_tokens) == 0:
        return 1.0, 1.0, 1.0
    if len(pred_tokens) == 0 or len(gt_tokens) == 0:
        return 0.0, 0.0, 0.0

    if token_mode == 'multiset':
        common = Counter(pred_tokens) & Counter(gt_tokens)
        num_same = sum(common.values())
        denom_p, denom_g = len(pred_tokens), len(gt_tokens)
    else:
        pred_set, gt_set = set(pred_tokens), set(gt_tokens)
        num_same = len(pred_set & gt_set)
        denom_p, denom_g = len(pred_set), len(gt_set)

    if num_same == 0:
        return 0.0, 0.0, 0.0
    precision = num_same / denom_p
    recall = num_same / denom_g
    return precision, recall, 2 * precision * recall / (precision + recall)


def compute_f1_score(prediction: str, ground_truth: str, use_synonyms: bool = False,
                     token_mode: str = 'set') -> float:
    """F1 tung cau. Wrapper mong quanh compute_prf — giu nguyen chu ky cu cho 40+ script."""
    return compute_prf(prediction, ground_truth, use_synonyms, token_mode)[2]


def detect_question_type(question_text: str) -> str:
    """Wrapper around dataset.detect_question_type — same logic as training."""
    return _TYPE_NAMES[_detect_type_int(question_text)]


def build_valid_answers_set(train_csv):
    """Build set of NFC-normalized lowercase valid answers (for snap post-processing)."""
    import pandas as pd
    df = pd.read_csv(train_csv)
    return {unicodedata.normalize('NFC', str(a).strip().lower()) for a in df['answer'].unique()}


def snap_to_valid_answer(pred, valid_answers_set):
    """Fix garbled constrained-decoding outputs: if pred not in trie vocab,
    snap to the longest valid answer that is a character-level prefix of pred.

    Root cause: BARTpho SentencePiece sometimes concatenates syllables without
    spaces at answer boundaries, producing 'cái ghếa', 'diềuván lướt sóng', etc.
    These are NOT in the training trie, but their valid prefix IS (e.g. 'cái ghế').
    """
    pred_n = unicodedata.normalize('NFC', pred.strip().lower())
    if pred_n in valid_answers_set:
        return pred.strip()
    candidates = [a for a in valid_answers_set if pred_n.startswith(a) and len(a) > 0]
    if candidates:
        return max(candidates, key=len)
    return pred.strip()


def build_answer_trie(train_csv, tokenizer):
    """Build prefix trie from all unique answers in training CSV.

    IMPORTANT: Dataset encodes answers WITH special tokens (add_special_tokens=True),
    so labels = [BOS=0, tokens..., EOS=2, PAD=-100...].
    After shift_tokens_right, model learns: [BOS] → BOS → actual_tokens → EOS.
    The trie must match this protocol: include BOS=0 as mandatory first token.
    """
    import pandas as pd
    df = pd.read_csv(train_csv)
    pad_id = tokenizer.pad_token_id
    trie = {}
    for answer in df['answer'].unique():
        # encode WITH special tokens to match training label format:
        # [BOS=0, answer_tokens..., EOS=2]
        tokens = tokenizer.encode(str(answer))
        tokens = [t for t in tokens if t != pad_id]  # strip padding just in case
        node = trie
        for t in tokens:
            if t not in node:
                node[t] = {}
            node = node[t]
    print(f"[Trie] Built from {len(df['answer'].unique())} unique answers ({len(df)} total rows)")
    return trie


def _fit_oracle_alpha(model, pixel_values, input_ids, attention_mask, labels,
                      question_types, region_map, mode, steps, lr):
    """TRAN TREN cua ho gate: tim alpha per-sample cuc dai log-likelihood dap an GOLD.

    Day KHONG phai mot phuong phap -- no dung nhan that luc suy luan, nen la gian lan co chu y.
    Muc dich duy nhat: tra loi cau hoi "cong thuc blend alpha*v + (1-alpha)*t co CHUA loi giai
    khong", tach bach hai kha nang da lan lon suot ca investigation:
      tran THAP  -> ho gate khong chua loi giai; moi cach tinh alpha deu vo vong (dong ho gate).
      tran CAO   -> loi giai CO trong ho; cai thieu la TIN HIEU HOC, khong phai kien truc
                    (mo lai huong giam sat alpha: distill tu alpha_oracle, hoac nhan vung COCO).

    Khoi tao theta = logit(alpha cua chinh model) nen buoc 0 = T2 chinh xac; oracle chi di len
    tu do. Chi theta co gradient (moi tham so model da requires_grad_(False) o evaluate()).

    Returns: alpha [B, P] hoac [B, 1] (mode='scalar'), da detach, trong [0, 1].
    """
    was_override = getattr(model.vision_gating, 'alpha_override', None)
    model.vision_gating.alpha_override = None

    # Buoc 0: lay alpha that cua model lam diem xuat phat
    with torch.no_grad():
        model(pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask,
              labels=labels, question_types=question_types, region_map=region_map)
        a0 = model.vision_gating.last_alpha.detach().float()   # [B, P]
    if a0.dim() == 3:
        a0 = a0.squeeze(-1)
    if mode.startswith('scalar'):
        a0 = a0.mean(dim=1, keepdim=True)                      # [B, 1] mot bac tu do/mau
    theta = torch.logit(a0.clamp(1e-4, 1 - 1e-4)).requires_grad_(True)
    opt = torch.optim.Adam([theta], lr=lr)

    with torch.enable_grad():
        for _ in range(steps):
            opt.zero_grad(set_to_none=True)
            model.vision_gating.alpha_override = torch.sigmoid(theta)
            out = model(pixel_values=pixel_values, input_ids=input_ids,
                        attention_mask=attention_mask, labels=labels,
                        question_types=question_types, region_map=region_map)
            # CE THUAN tren token gold -- KHONG dung answer_loss cua model (co cdw_ce +
            # answer_weights + label_smoothing) de tran do duoc la likelihood that, khong
            # phai mot muc tieu da bi cac trong so phu lam lech.
            lg = out.answer_logits
            loss = F.cross_entropy(lg.reshape(-1, lg.size(-1)), labels.reshape(-1),
                                   ignore_index=-100)
            loss.backward()
            opt.step()

    model.vision_gating.alpha_override = was_override
    return torch.sigmoid(theta).detach()


def evaluate(model, dataloader, device, tokenizer, num_beams=3, repetition_penalty=1.0,
             max_length=20, use_synonyms=False, num_samples=1, vote_temp=0.8,
             prefix_trie=None, valid_answers_set=None, type_mode='predicted', legacy_beam=False, zero_vision=False,
             alpha_file=None, dump_first_logits=None, dump_seq_logprob=None, dump_model_alpha=None,
             token_mode='set',
             diag_harm=False, region_map_lookup=None,
             oracle_alpha=None, oracle_steps=40, oracle_lr=0.5, dump_oracle_alpha=None):
    model.eval()
    _oa_dump = [] if dump_oracle_alpha else None
    _wrong_targets = []
    # pool dap an SAI: lay tu tap dap an hop le, dung cho mode 'perpatchwrong'
    _WRONG_POOL = sorted(valid_answers_set) if valid_answers_set else ['màu đỏ', 'hai', 'phòng bếp']
    _dfl_logit, _dfl_row = [], []
    _dsl = []
    _ma = []
    _oa_ids = []

    if oracle_alpha:
        if getattr(model, 'vision_gating', None) is None:
            raise SystemExit('[oracle_alpha] checkpoint khong co vision_gating -- can mot model T2.')
        for _p in model.parameters():
            _p.requires_grad_(False)     # chi theta duoc hoc; model dung yen hoan toan
        print(f'[oracle_alpha] mode={oracle_alpha} steps={oracle_steps} lr={oracle_lr} '
              f'-- alpha duoc toi uu tren NHAN GOLD (tran tren, khong phai phuong phap)')

    _INT_TO_TYPE = {0: 'OBJECT', 1: 'COUNT', 2: 'COLOR', 3: 'LOCATION'}
    all_pred_types = []   # type do type head du doan (de tinh confusion matrix)

    all_predictions = []
    all_ground_truths = []
    all_questions = []
    all_question_types = []
    all_alpha = []          # 🔬 per-sample mean gate alpha (de phan tich flip fix/break)
    all_ceon = []; all_ceoff = []   # 🔬 diag_harm: per-sample CE gate on/off (gold labels)
    all_hon = []; all_hoff = []      # 🔬 per-sample decoder ENTROPY (uncertainty, deployable)
    _acap = {}
    def _ahook(_m, _i, o):
        if isinstance(o, tuple) and len(o) > 1 and torch.is_tensor(o[1]):
            _acap['a'] = o[1].detach().float().cpu()
    _ahandle = model.vision_gating.register_forward_hook(_ahook) if getattr(model, 'vision_gating', None) is not None else None

    exact_matches = []
    f1_scores = []

    # Per-type tracking
    type_exact_matches = defaultdict(list)
    type_f1_scores = defaultdict(list)
    precision_scores = []
    recall_scores = []
    type_precision_scores = defaultdict(list)
    type_recall_scores = defaultdict(list)

    # Type prediction accuracy tracking (only when model has type_head)
    type_pred_correct = []
    type_pred_per_type = defaultdict(list)  # ground_type → [correct/incorrect]

    # 🔬 Voi codebook khong giam sat, generate() tra ve CHI SO CUM qua cung duong nay
    # (predicted_types := cb_idx). Nen phai bat return_type_preds ke ca khi khong co type_head.
    has_type_head = (
        (getattr(model, 'use_type_task', False) and model.type_head is not None)
        or getattr(model, 'type_codebook', None) is not None
    )
    _is_codebook = getattr(model, 'type_codebook', None) is not None

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Evaluating")

        for batch in pbar:
            pixel_values = batch['pixel_values'].to(device)
            if zero_vision:
                pixel_values = torch.zeros_like(pixel_values)   # do tran dong gop cua thi giac
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # 🔬 region_map (COCO instance region, tcvg_spatial_blend): giong het logic o
            # train.py -- PHAI dung nhau giua train/eval, neu khong se train mot kieu, eval
            # kieu khac (lop bug da gap voi tcvg_topk_random truoc day).
            region_map = None
            if region_map_lookup is not None and 'img_id' in batch:
                _ids = batch['img_id'].tolist() if torch.is_tensor(batch['img_id']) else list(batch['img_id'])
                _rows = [region_map_lookup.get(int(_iid)) for _iid in _ids]
                _np_default = next((r.shape[0] for r in _rows if r is not None), None)
                if _np_default is not None:
                    _mat = np.stack([r if r is not None else np.zeros(_np_default, dtype=np.int16)
                                      for r in _rows], axis=0)
                    region_map = torch.from_numpy(_mat).long().to(device)

            # Generate (encoder runs once inside here)
            # Oracle mode: TCVG dung gold type tu CSV thay vi type head
            _gold = None
            if type_mode == 'gold':
                _csv_types = batch.get('question_type')
                if _csv_types is not None:
                    _gold = _csv_types.to(device).long()
            elif type_mode in ('shuffled', 'wrong', 'const'):
                # 🔬 DO TAC DONG cua tin hieu loai: pha nhan loai dua vao TCVG roi DEM XEM
                # bao nhieu du doan thay doi. Day la thuoc do TAC DONG, tach hoan toan khoi
                # accuracy — neu gan 0% thi dieu-kien-hoa-theo-loai la VO HIEU o mo hinh da train,
                # bat ke EM cao hay thap.
                #   shuffled: hoan vi ngau nhien nhan loai trong batch (giu phan phoi)
                #   wrong   : ep sang mot loai KHAC (type+1 mod 4) — sai co he thong
                #   const   : ep tat ca ve loai 0
                _csv_types = batch.get('question_type')
                if _csv_types is not None:
                    _t = _csv_types.to(device).long()
                    if type_mode == 'shuffled':
                        _gold = _t[torch.randperm(_t.size(0), device=device)]
                    elif type_mode == 'wrong':
                        _gold = (_t + 1) % 4
                    else:
                        _gold = torch.zeros_like(_t)

            # 🔬 ORACLE ALPHA: toi uu alpha tren nhan gold TRUOC khi decode. Decode van chay
            # y het moi eval khac (cung beam/trie/synonyms) nen so EM so sanh truc tiep duoc.
            if oracle_alpha:
                # 🔬 mode 'perpatchwrong': DOI CHUNG DO DIEU KHIEN DUOC.
                # perpatch oracle co 197 bac tu do MOI MAU. Voi 197 bac tu do co the fit gan nhu
                # bat cu gi, nen +11.27 diem (83.61 vs 72.34) CO THE chi do "gate dieu khien duoc",
                # khong phai "bai toan con du dia". Doi chung shuffle KHONG bat duoc dieu nay vi
                # overfit thuan cung cho alpha rieng theo mau.
                # Phep thu: khop alpha ve mot dap an SAI ngau nhien. Neu model cung noi ra dap an
                # sai do voi ti le cao thi alpha chi la mot cai num dieu khien, va +11.27 vo nghia.
                _fit_labels = labels
                if oracle_alpha == 'perpatchwrong':
                    _wr = []
                    for _b in range(labels.size(0)):
                        _cand = _WRONG_POOL[torch.randint(len(_WRONG_POOL), (1,)).item()]
                        _wr.append(_cand)
                    _wt = tokenizer(_wr, return_tensors='pt', padding='max_length',
                                    truncation=True, max_length=labels.size(1))
                    _fit_labels = _wt.input_ids.to(labels.device)
                    _fit_labels[_fit_labels == tokenizer.pad_token_id] = -100
                    _wrong_targets.extend(_wr)
                _oa = _fit_oracle_alpha(
                    model, pixel_values, input_ids, attention_mask, _fit_labels,
                    question_types=(batch.get('question_type').to(device).long()
                                    if batch.get('question_type') is not None else None),
                    region_map=region_map,
                    mode=('perpatch' if oracle_alpha=='perpatchwrong' else oracle_alpha),
                    steps=oracle_steps, lr=oracle_lr)
                if _oa_dump is not None:
                    # luu TRUOC khi roll (nhan distill phai la alpha CUA CHINH mau do)
                    _oa_dump.append(_oa.detach().float().cpu().numpy().astype('float16'))
                    _bid = batch.get('img_id')
                    _oa_ids.extend(_bid.tolist() if torch.is_tensor(_bid)
                                   else (list(_bid) if _bid is not None else []))
                if oracle_alpha in ('shuffle', 'scalarshuffle'):
                    # doi chung NULL: alpha toi uu cua mau i ap cho mau i+1 (cac mau trong batch
                    # khong lien quan nhau). Neu EM van cao -> muc tang KHONG den tu viec chon
                    # dung patch cho DUNG mau, ma chi la dich chuyen bien do chung.
                    _oa = torch.roll(_oa, 1, dims=0)
                model.vision_gating.alpha_override = _oa

            _acap.clear()
            # 🔬 --dump_first_logits: lay phan bo token DAP AN DAU TIEN bang mot luot forward
            #   teacher-forced. Voi COUNT thi 9/11 tu so la 1 token, nen day CHINH LA phan bo tren
            #   cau tra loi -> dung de hieu chinh lech-mot ma khong dinh gi den beam search.
            if dump_first_logits is not None and 'row_idx' in batch:
                with torch.no_grad():
                    _fo = model(pixel_values=pixel_values, input_ids=input_ids,
                                attention_mask=attention_mask, labels=labels)
                    _dfl_logit.append(_fo.answer_logits[:, 0, :].float().cpu())
                    _dfl_row.append(batch['row_idx'].cpu())

            # 🔬 --dump_model_alpha: luu alpha PER-PATCH ma model tu tinh (khong ep gi ca).
            #   Can cho phep do top-k: giu k patch alpha cao nhat, ep phan con lai xuong nen.
            if dump_model_alpha is not None:
                _ma_pending = True

            # 🔬 --alpha_from_file: ep alpha theo nhan ngoai (vi du mat na box), tra theo row_idx.
            #   NaN trong file = giu alpha cua model o o do. Do EM TRUC TIEP thay vi do tuong quan
            #   voi oracle alpha -- vi oracle chi la MOT diem trong tap nghiem, khong phai diem
            #   duy nhat hay diem hoc duoc.
            if alpha_file is not None and 'row_idx' in batch:
                _af = alpha_file[batch['row_idx'].to(alpha_file.device)]
                model.vision_gating.alpha_override = _af.to(device)
            gen_out = model.generate(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                num_beams=num_beams,
                repetition_penalty=repetition_penalty,
                return_type_preds=has_type_head,
                region_map=region_map,
                num_samples=num_samples,
                vote_temp=vote_temp,
                prefix_trie=prefix_trie,
                gold_types=_gold,
                legacy_beam=legacy_beam,
            )
            if oracle_alpha or alpha_file is not None:
                model.vision_gating.alpha_override = None   # batch sau tu tinh lai tu dau

            if dump_model_alpha is not None and getattr(model.vision_gating,'last_alpha',None) is not None:
                _ma.append(model.vision_gating.last_alpha.detach().float().cpu())

            if has_type_head:
                predictions, pred_type_ids = gen_out
            else:
                predictions = gen_out
                pred_type_ids = None

            # Ground-truth type: from CSV column (same source as training)
            batch_gt_types = []
            csv_type_ids = batch.get('question_type')
            for i, inp in enumerate(input_ids):
                question_text = tokenizer.decode(inp, skip_special_tokens=True)
                all_questions.append(question_text)
                if csv_type_ids is not None:
                    q_type = _INT_TO_TYPE[int(csv_type_ids[i])]
                else:
                    q_type = detect_question_type(question_text)
                all_question_types.append(q_type)
                batch_gt_types.append(q_type)

            # Type prediction accuracy: use type preds already computed in generate()
            if has_type_head and pred_type_ids is not None:
                if _is_codebook:
                    # Cum tu phat hien: chi so la TUY Y, khong anh xa sang ten loai duoc va
                    # khong tinh "accuracy" duoc. Giu nguyen chi so, danh gia bang purity/NMI.
                    all_pred_types.extend([f"cum{int(t)}" for t in pred_type_ids])
                else:
                    pred_type_names = [_INT_TO_TYPE[t] for t in pred_type_ids]
                    all_pred_types.extend(pred_type_names)
                    for pred_t, gt_t in zip(pred_type_names, batch_gt_types):
                        correct = int(pred_t == gt_t)
                        type_pred_correct.append(correct)
                        type_pred_per_type[gt_t].append(correct)

            # Decode ground truths
            for label in labels:
                label_tokens = label[label != -100].cpu().tolist()
                gt_text = _decode_gt(tokenizer, label_tokens)
                all_ground_truths.append(gt_text)

            # 🔬 --dump_seq_logprob: do TIN CAY cua model tren chinh dap an no vua sinh ra.
            #   Dung de model TU CHON muc alpha luc suy luan (khong dung dap an gold):
            #   chay nhieu muc alpha -> lay dap an co log-prob cao nhat.
            if dump_seq_logprob is not None:
                with torch.no_grad():
                    _enc = model.tokenizer(list(predictions), truncation=True, padding='max_length',
                                           max_length=max_length, return_tensors='pt')
                    _ids = _enc['input_ids'].to(device)
                    _ids[_ids == model.tokenizer.pad_token_id] = -100
                    _lp = model.compute_seq_logprob(pixel_values, input_ids, attention_mask, _ids)
                    _ntok = (_ids != -100).sum(1).clamp(min=1)
                    _dsl.append((_lp / _ntok).float().cpu())   # log-prob TRUNG BINH moi token

            # Snap garbled predictions back to nearest valid trie answer
            if valid_answers_set is not None:
                predictions = [snap_to_valid_answer(p, valid_answers_set) for p in predictions]

            all_predictions.extend(predictions)

            # 🔬 capture per-sample mean alpha (aligned voi predictions order)
            if _ahandle is not None:
                _a = _acap.get('a')
                if _a is not None:
                    if _a.dim() == 3:
                        _a = _a.squeeze(-1)
                    for i in range(len(predictions)):
                        all_alpha.append(float(_a[i].mean()) if i < _a.size(0) else float('nan'))
                else:
                    all_alpha.extend([float('nan')] * len(predictions))

            # 🔬 diag_harm cho model KHONG gate (T0): chi tinh H 1-pass de so RC-curve
            if diag_harm and getattr(model, 'vision_gating', None) is None:
                _qt0 = batch.get('question_type'); _qt0 = _qt0.to(device).long() if _qt0 is not None else None
                _o0 = model(pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask, labels=labels, question_types=_qt0)
                _p0 = torch.softmax(_o0.answer_logits, -1)
                _H0 = -(_p0 * torch.log(_p0 + 1e-9)).sum(-1)
                _m0 = (labels != -100).float()
                _h0 = ((_H0 * _m0).sum(1) / _m0.sum(1).clamp(min=1)).detach().float().cpu().numpy()
                _ce0f = torch.nn.functional.cross_entropy(_o0.answer_logits.reshape(-1, _o0.answer_logits.size(-1)),
                        labels.reshape(-1), ignore_index=-100, reduction='none').view(labels.size()).detach()
                _ce0 = ((_ce0f * _m0).sum(1) / _m0.sum(1).clamp(min=1)).float().cpu().numpy()
                all_hon.extend(_h0.tolist()); all_hoff.extend(_h0.tolist())
                all_ceon.extend(_ce0.tolist()); all_ceoff.extend(_ce0.tolist())
                all_alpha.extend([float('nan')]*len(predictions))
            # 🔬 diag_harm: per-sample CE voi gate ON va OFF (gold labels) de test tach fix/break
            if diag_harm and getattr(model, 'vision_gating', None) is not None:
                _qt = batch.get('question_type')
                _qt = _qt.to(device).long() if _qt is not None else None
                def _persample_ce(_out):
                    _lg = _out.answer_logits; _B, _T, _V = _lg.shape
                    _ce = torch.nn.functional.cross_entropy(
                        _lg.reshape(-1, _V), labels.reshape(-1), ignore_index=-100, reduction='none').view(_B, _T)
                    _m = (labels != -100).float()
                    return ((_ce * _m).sum(1) / _m.sum(1).clamp(min=1)).detach().float().cpu().numpy()
                def _persample_ent(_out):
                    _lg = _out.answer_logits; _p = torch.softmax(_lg, -1)
                    _H = -(_p * torch.log(_p + 1e-9)).sum(-1)              # [B,T] entropy per token
                    _m = (labels != -100).float()
                    return ((_H * _m).sum(1) / _m.sum(1).clamp(min=1)).detach().float().cpu().numpy()
                _sv = model.use_vision_gate
                model.use_vision_gate = True
                _o_on = model(pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask, labels=labels, question_types=_qt)
                _on = _persample_ce(_o_on); _hon = _persample_ent(_o_on)
                model.use_vision_gate = False
                _o_off = model(pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask, labels=labels, question_types=_qt)
                _off = _persample_ce(_o_off); _hoff = _persample_ent(_o_off)
                model.use_vision_gate = _sv
                all_ceon.extend(_on.tolist()); all_ceoff.extend(_off.tolist())
                all_hon.extend(_hon.tolist()); all_hoff.extend(_hoff.tolist())

            # Metrics (overall and per-type)
            batch_start_idx = len(all_ground_truths) - len(predictions)
            for i, (pred, gt) in enumerate(zip(predictions, all_ground_truths[-len(predictions):])):
                em = compute_exact_match(pred, gt, use_synonyms)
                prec, rec, f1 = compute_prf(pred, gt, use_synonyms, token_mode)
                q_type = all_question_types[batch_start_idx + i]

                exact_matches.append(em)
                precision_scores.append(prec)
                recall_scores.append(rec)
                f1_scores.append(f1)

                type_exact_matches[q_type].append(em)
                type_precision_scores[q_type].append(prec)
                type_recall_scores[q_type].append(rec)
                type_f1_scores[q_type].append(f1)

            current_em = sum(exact_matches) / len(exact_matches) * 100
            current_f1 = sum(f1_scores) / len(f1_scores) * 100

            pbar.set_postfix({
                'EM': f"{current_em:.1f}%",
                'F1': f"{current_f1:.1f}%"
            })

    if dump_model_alpha is not None and _ma:
        _A = torch.cat(_ma).numpy()
        np.save(dump_model_alpha, _A.astype('float16'))
        print(f'[dump_model_alpha] {_A.shape} -> {dump_model_alpha}')

    if dump_seq_logprob is not None and _dsl:
        np.save(dump_seq_logprob, torch.cat(_dsl).numpy())
        print(f'[dump_seq_logprob] {torch.cat(_dsl).shape} -> {dump_seq_logprob}')

    if dump_first_logits is not None and _dfl_logit:
        _L = torch.cat(_dfl_logit).numpy(); _R = torch.cat(_dfl_row).numpy()
        np.savez_compressed(dump_first_logits, logits=_L.astype('float16'), row_idx=_R)
        print(f'[dump_first_logits] {_L.shape} -> {dump_first_logits}')

    if _oa_dump:
        # Thu tu hang = thu tu CSV (DataLoader shuffle=False) -> train.py tra cuu bang chi so hang.
        # Luu kem img_id de doi chieu: neu lech thu tu thi nhan distill se lech mau, sai am tham.
        _arr = np.concatenate(_oa_dump, axis=0)
        np.savez_compressed(dump_oracle_alpha, alpha=_arr,
                            img_id=np.array(_oa_ids) if _oa_ids else np.array([]))
        print(f'[dump_oracle_alpha] {_arr.shape} -> {dump_oracle_alpha}')

    exact_match_acc = sum(exact_matches) / len(exact_matches) * 100
    f1_score_avg = sum(f1_scores) / len(f1_scores) * 100
    precision_avg = sum(precision_scores) / len(precision_scores) * 100
    recall_avg = sum(recall_scores) / len(recall_scores) * 100

    # Per-type metrics
    per_type_results = {}
    for q_type in sorted(type_exact_matches.keys()):
        type_em = sum(type_exact_matches[q_type]) / len(type_exact_matches[q_type]) * 100 if type_exact_matches[q_type] else 0
        type_f1 = sum(type_f1_scores[q_type]) / len(type_f1_scores[q_type]) * 100 if type_f1_scores[q_type] else 0
        _n = len(type_exact_matches[q_type])
        per_type_results[q_type] = {
            'exact_match': type_em,
            'f1_score': type_f1,
            'precision': sum(type_precision_scores[q_type]) / _n * 100 if _n else 0,
            'recall': sum(type_recall_scores[q_type]) / _n * 100 if _n else 0,
            'count': _n
        }

    # Type prediction accuracy (if model has type_head)
    type_pred_accuracy = None
    if type_pred_correct:
        overall_acc = sum(type_pred_correct) / len(type_pred_correct) * 100
        per_type_acc = {
            t: sum(v) / len(v) * 100
            for t, v in type_pred_per_type.items() if v
        }
        type_pred_accuracy = {'overall': overall_acc, 'per_type': per_type_acc}

    if _ahandle is not None:
        _ahandle.remove()

    return {
        'exact_match': exact_match_acc,
        'f1_score': f1_score_avg,
        'precision': precision_avg,
        'recall': recall_avg,
        'per_type': per_type_results,
        'type_pred_accuracy': type_pred_accuracy,
        'alpha_mean': all_alpha,
        'ce_on': all_ceon,
        'ce_off': all_ceoff,
        'h_on': all_hon,
        'h_off': all_hoff,
        'predictions': all_predictions,
        'ground_truths': all_ground_truths,
        'questions': all_questions,
        'question_types': all_question_types,
        'pred_question_types': all_pred_types,
        'wrong_targets': _wrong_targets,   # 🔬 chi co o mode perpatchwrong
    }



def run_measurement_mode(args, model, dataloader, device):
    """Muc 6/7/8 cua experiment.md — do tren model DA dung san boi main()."""
    import json, time
    import numpy as np
    import torch

    os.makedirs(args.measure_out, exist_ok=True)
    tag = args.measure_tag or 'model'

    # ---------------- Muc 7: dem tham so theo thanh phan ----------------
    if args.param_count:
        groups = [
            ('SigLIP (vision encoder)', 'vision_encoder'),
            ('BARTpho encoder', 'encoder'),
            ('GCA (flamingo_fusion)', 'flamingo_fusion'),
            ('TCVG (vision_gating)', 'vision_gating'),
            ('Type head', 'type_head'),
            ('Decoder', 'decoder'),
            ('LM head', 'lm_head'),
        ]
        seen, rows = set(), []
        named = list(model.named_parameters())
        for name, prefix in groups:
            sel = [(n, p) for n, p in named if n.startswith(prefix) and n not in seen]
            seen.update(n for n, _ in sel)
            tot = sum(p.numel() for _, p in sel)
            tr = sum(p.numel() for _, p in sel if p.requires_grad)
            rows.append({'component': name, 'total': tot, 'trainable': tr,
                         'frozen': tot - tr,
                         'trainable_pct': round(100.0 * tr / max(tot, 1), 2)})
        rest = [(n, p) for n, p in named if n not in seen]
        tot = sum(p.numel() for _, p in rest)
        tr = sum(p.numel() for _, p in rest if p.requires_grad)
        rows.append({'component': 'Khac', 'total': tot, 'trainable': tr,
                     'frozen': tot - tr, 'trainable_pct': round(100.0 * tr / max(tot, 1), 2)})
        gt = sum(p.numel() for _, p in named)
        gtr = sum(p.numel() for _, p in named if p.requires_grad)
        rows.append({'component': 'TONG', 'total': gt, 'trainable': gtr, 'frozen': gt - gtr,
                     'trainable_pct': round(100.0 * gtr / max(gt, 1), 2)})
        print(f"\n{'component':28s} {'total':>13s} {'trainable':>13s} {'frozen':>13s} {'tr%':>7s}")
        for r in rows:
            print(f"{r['component']:28s} {r['total']:13,d} {r['trainable']:13,d} "
                  f"{r['frozen']:13,d} {r['trainable_pct']:7.2f}")
        with open(f'{args.measure_out}/parameter_count_{tag}.json', 'w') as f:
            json.dump(rows, f, indent=1)

    # ---------------- Muc 8: latency / throughput / VRAM ----------------
    if args.measure_latency:
        torch.cuda.reset_peak_memory_stats(device)
        lat, n_warm, n_meas = [], 50, args.measure_n
        done = 0
        with torch.no_grad():
            for batch in dataloader:
                pv = batch['pixel_values'][:1].to(device)
                ii = batch['input_ids'][:1].to(device)
                am = batch['attention_mask'][:1].to(device)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                model.generate(pixel_values=pv, input_ids=ii, attention_mask=am,
                               max_length=args.max_length, num_beams=args.num_beams,
                               repetition_penalty=args.repetition_penalty)
                torch.cuda.synchronize()
                dt = (time.perf_counter() - t0) * 1000.0
                done += 1
                if done > n_warm:
                    lat.append(dt)
                if done >= n_warm + n_meas:
                    break
        a = np.array(lat)
        out = {'tag': tag, 'n_measured': len(a), 'num_beams': args.num_beams,
               'mean_ms': float(a.mean()), 'median_ms': float(np.median(a)),
               'p95_ms': float(np.percentile(a, 95)), 'p99_ms': float(np.percentile(a, 99)),
               'samples_per_sec': float(1000.0 / a.mean()),
               'peak_alloc_MB': torch.cuda.max_memory_allocated(device) / 1e6,
               'peak_reserved_MB': torch.cuda.max_memory_reserved(device) / 1e6}
        print('\nLATENCY ' + json.dumps(out, indent=1))
        with open(f'{args.measure_out}/latency_{tag}.json', 'w') as f:
            json.dump(out, f, indent=1)

    # ---------------- Muc 6: phan phoi gate alpha ----------------
    if args.dump_gate_stats:
        if not getattr(model, 'use_vision_gate', False) or model.vision_gating is None:
            print('Model khong co TCVG — bo qua gate stats.')
            return
        import pandas as pd
        _INT = {0: 'OBJECT', 1: 'COUNT', 2: 'COLOR', 3: 'LOCATION'}
        cap = {}

        def hook(_m, _i, o):
            if isinstance(o, tuple) and len(o) > 1 and torch.is_tensor(o[1]):
                cap['alpha'] = o[1].detach().float().cpu()

        h = model.vision_gating.register_forward_hook(hook)
        recs = []
        with torch.no_grad():
            for batch in tqdm(dataloader, desc='gate stats'):
                pv = batch['pixel_values'].to(device)
                ii = batch['input_ids'].to(device)
                am = batch['attention_mask'].to(device)
                cap.clear()
                model.generate(pixel_values=pv, input_ids=ii, attention_mask=am,
                               max_length=args.max_length, num_beams=1)
                if 'alpha' not in cap:
                    continue
                al = cap['alpha']
                if al.dim() == 3:
                    al = al.squeeze(-1)
                qt = batch.get('question_type')
                for i in range(al.size(0)):
                    v = al[i].numpy().ravel()
                    recs.append({'question_type': _INT[int(qt[i])] if qt is not None else 'NA',
                                 'mean': v.mean(), 'std': v.std(), 'median': np.median(v),
                                 'min': v.min(), 'max': v.max(),
                                 'p10': np.percentile(v, 10), 'p90': np.percentile(v, 90),
                                 'frac_ge_099': float((v >= 0.99).mean()),
                                 'frac_le_001': float((v <= 0.01).mean())})
        h.remove()
        d = pd.DataFrame(recs)
        d.to_csv(f'{args.measure_out}/gate_per_sample_{tag}.csv', index=False)
        agg = d.groupby('question_type').agg(
            count=('mean', 'size'), mean=('mean', 'mean'), std=('mean', 'std'),
            median=('median', 'mean'), p10=('p10', 'mean'), p90=('p90', 'mean'),
            within_sample_std=('std', 'mean'),
            frac_ge_099=('frac_ge_099', 'mean'), frac_le_001=('frac_le_001', 'mean'))
        print('\nGATE ALPHA theo loai cau hoi:')
        print(agg.to_string(float_format=lambda x: f'{x:8.4f}'))
        agg.to_csv(f'{args.measure_out}/gate_stats_{tag}.csv')


def _n_answer_classes_from_ckpt(checkpoint, saved_args):
    """So lop cua answer_head, doc tu hinh dang tensor trong checkpoint.

    Truoc day o day gan cung 328 (so lop cua train_split.csv) trong khi cac lan train dung
    train_split_original.csv (331 lop) -> load_state_dict crash. Hinh dang tensor luon dung
    voi checkpoint dang load, khong phu thuoc CSV nao duoc truyen luc eval.
    """
    sd = checkpoint.get('model_state_dict', checkpoint)
    n = 0
    for k, v in sd.items():   # lop tuyen tinh CUOI cung cua answer_head la lop xuat
        if k.startswith('answer_head.') and k.endswith('.weight') and v.dim() == 2:
            n = int(v.shape[0])
    return n or saved_args.get('_n_ans_cls', 0) or 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--csv_path', type=str, required=True)
    parser.add_argument('--probe_topk', type=int, default=0, help='PROBE: giu top-k patch theo alpha (0=off)')
    parser.add_argument('--probe_topk_random', action='store_true', help='PROBE: giu k patch NGAU NHIEN (doi chung)')
    parser.add_argument('--probe_topk_bottom', action='store_true', help='PROBE: giu k patch alpha THAP NHAT (doi chung 2, nguoc top)')
    parser.add_argument('--tcvg_spatial_blend_region_map', type=str, default=None,
                        help='Duong dan file .pkl (tu build_patch_region_map.py) -- PHAI truyen '
                             'GIONG HET luc train neu model duoc train voi flag nay, neu khong '
                             'se eval sai (blend_target khac luc train).')
    parser.add_argument('--gate_beta_override', type=str, default=None,
                        help='ORACLE: ep beta_type gate (4 so [OBJ,COUNT,COLOR,LOC], vd "1,1,0.5,1") '
                             'de quet cuong do gate per-type tren checkpoint da train, khong train lai. '
                             'beta=1 = T2 goc, beta=0 = tat gate (T0).')
    parser.add_argument('--image_folder', type=str, required=True)
    parser.add_argument('--vision_model', type=str, default='google/siglip-base-patch16-224')
    parser.add_argument('--fusion_type', type=str, default=None,
                       choices=['text2vision', 'vision2text', 'bidirectional'],
                       help='Fusion type (default: auto-detect from checkpoint args)')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_beams', type=int, default=3,
                        help='Beam search width (default: 3, beam=5 is slower and not better)')
    parser.add_argument('--max_length', type=int, default=20,
                        help='Max generated tokens (default: 20, matches older eval)')
    parser.add_argument('--text_only', action='store_true',
                       help='Zero hoa VISION_FEATURES (tensor sau encoder), giong het che do\n                            text_only luc train (model.py:3038). KHAC --zero_vision von zero\n                            PIXEL_VALUES -> SigLIP van chay tren anh den va cho dac trung khac 0.\n                            Dung co nay de eval model train bang --text_only_warmup_epochs.')
    parser.add_argument('--zero_vision', action='store_true',
                        help='Zero toan bo pixel_values: do dong gop tuyet doi cua thi giac')
    parser.add_argument('--force_gate_open', action='store_true',
                        help='Ep alpha=1 moi patch luc suy luan: tach anh huong TRUC TIEP cua TCVG '
                             'khoi anh huong GIAN TIEP qua bieu dien hoc duoc luc train')
    parser.add_argument('--codebook_report', action='store_true',
                        help='In bang doi chieu cum tu phat hien (codebook) vs loai that, kem '
                             'purity va NMI. Cum KHONG phai nhan loai nen khong dung accuracy.')
    parser.add_argument('--codebook_size_report', type=int, default=4,
                        help='So prototype ky vong, chi de in mau bao cao')
    parser.add_argument('--dump_model_alpha', type=str, default=None,
                        help='Luu alpha PER-PATCH ma model tu tinh (.npy [N,P]). Dung cho phep do top-k.')
    parser.add_argument('--dump_seq_logprob', type=str, default=None,
                        help='Luu log-prob TRUNG BINH moi token cua dap an model vua sinh (.npy). '
                             'Dung de model TU CHON muc alpha: chay nhieu muc, lay dap an tu tin nhat.')
    parser.add_argument('--dump_first_logits', type=str, default=None,
                        help='Luu logits cua token dap an DAU TIEN (mot luot forward teacher-forced) '
                             'ra .npz. Dung de hieu chinh COUNT lech-mot: 9/11 tu so la 1 token nen '
                             'day chinh la phan bo tren cau tra loi, khong dinh gi den beam search.')
    parser.add_argument('--alpha_from_file', type=str, default=None,
                        help='File .npz khoa "alpha" [N_hang, 196 hoac 197]: ep alpha theo nhan ngoai, '
                             'tra theo row_idx. NaN = giu alpha cua model. Dung de DO EM khi alpha co '
                             'hinh dang theo box, thay vi do tuong quan voi oracle alpha.')
    parser.add_argument('--flatten_alpha', action='store_true',
                        help='Thay alpha bang trung binh cua no tren cac patch: PHA tinh chon loc '
                             'nhung GIU bien do. Doi chung cho --force_gate_open, tach bach xem '
                             'thiet hai den tu chon loc hay tu dich chuyen phan phoi bien do')
    # ── ORACLE ALPHA: do TRAN TREN cua ho gate (khong train lai) ──────────
    parser.add_argument('--oracle_alpha', type=str, default=None,
                        choices=['perpatch', 'scalar', 'shuffle', 'scalarshuffle', 'perpatchwrong'],
                        help='Toi uu alpha per-sample luc suy luan de cuc dai log-likelihood dap an '
                             'GOLD, roi decode binh thuong. Do TRAN TREN cua MOI thiet ke gate co the '
                             'co trong cong thuc blend hien tai. '
                             'perpatch: P bac tu do/mau (tran tren that su). '
                             'scalar: 1 bac tu do/mau (chi luong co ve text) -- doi chung tach bach '
                             'phan "chon loc theo patch" khoi phan "bien do co". '
                             'shuffle: dung alpha da toi uu cua mau KHAC trong batch -- doi chung NULL, '
                             'do xem bao nhieu phan cua muc tang la thong tin RIENG cua mau vs chi la '
                             'dich chuyen bien do chung.')
    parser.add_argument('--oracle_gate_fit', type=int, default=0,
                        help='TRAN TREN CUA HO HAM (khac han --oracle_alpha, von la tran cua ho '
                             'GIA TRI alpha). Toi uu THAM SO cua vision_gating (mot ham dung chung '
                             'moi mau, chi doc (v,q) -- KHONG doc dap an) tren chinh test set voi '
                             'nhan gold, N epoch, moi thu khac dong bang. Tra loi: "neu ton tai mot '
                             'CONG THUC cho ra alpha tot nhat thi duoc bao nhieu?" Vi no duoc phep '
                             'overfit thang vao test, ket qua la CAN TREN chat cua moi gate hoc duoc '
                             'tu (anh, cau hoi) -- ke ca gate distill hoan hao. Van la gian lan co '
                             'chu y (dung nhan test), khong bao gio bao cao nhu phuong phap.')
    parser.add_argument('--oracle_gate_lr', type=float, default=1e-3,
                        help='LR cho tham so gate trong --oracle_gate_fit')
    parser.add_argument('--oracle_gate_fit_csv', type=str, default=None,
                        help='Fit gate tren CSV NAY thay vi tren chinh test set. Dat = train split '
                             '-> con so THAT SU tra loi "mot cong thuc hoc tu du lieu cho bao nhieu" '
                             '(khong dung nhan test, khong overfit test). Bo trong -> fit thang tren '
                             'test = CAN TREN LONG (co overfit).')
    parser.add_argument('--oracle_gate_fit_images', type=str, default=None,
                        help='Thu muc anh cho --oracle_gate_fit_csv (vd archive/data/images/train)')
    parser.add_argument('--oracle_gate_val_csv', type=str, default=None,
                        help='CSV validation de CHON epoch fit theo val CE (early stopping). Bat '
                             'buoc phai co neu muon con so tra loi duoc cau hoi "co ton tai cong '
                             'thuc khong" -- khong co val thi fit chi do dac tinh overfit cua chinh '
                             'giao thuc fit (da xay ra: 4 epoch lr1e-3 -> test EM 70.51 < ca T0).')
    parser.add_argument('--oracle_gate_val_images', type=str, default=None,
                        help='Thu muc anh cho --oracle_gate_val_csv')
    parser.add_argument('--oracle_gate_wd', type=float, default=0.0,
                        help='Weight decay cho tham so gate khi fit')
    parser.add_argument('--oracle_gate_patience', type=int, default=2,
                        help='So epoch val CE khong cai thien thi dung som')
    parser.add_argument('--dump_oracle_alpha', type=str, default=None,
                        help='Luu alpha oracle ra .npz (theo thu tu hang CSV) de lam NHAN distill '
                             'cho train.py --gate_distill_path. Chay tren TRAIN split de sinh nhan.')
    parser.add_argument('--oracle_steps', type=int, default=40,
                        help='So buoc Adam toi uu alpha moi batch (mac dinh 40)')
    parser.add_argument('--oracle_lr', type=float, default=0.5,
                        help='Learning rate cho theta (logit cua alpha), mac dinh 0.5')
    parser.add_argument('--param_count', action='store_true', help='Muc 7')
    parser.add_argument('--measure_latency', action='store_true', help='Muc 8')
    parser.add_argument('--dump_gate_stats', action='store_true', help='Muc 6')
    parser.add_argument('--measure_out', type=str, default='analysis/measure')
    parser.add_argument('--measure_tag', type=str, default=None)
    parser.add_argument('--measure_n', type=int, default=500)
    parser.add_argument('--diag_harm', action='store_true',
                        help='Diagnostic: per-sample ce_on/ce_off (gate on/off, gold labels) -> phan tich fix/break tach duoc khong')
    parser.add_argument('--legacy_beam', action='store_true',
                        help='Tai hien beam search CO BUG (truoc 2026-07-26) de do lai anh huong')
    parser.add_argument('--type_mode', type=str, default='predicted',
                        choices=['predicted', 'gold', 'shuffled', 'wrong', 'const'],
                        help="TCVG lay type tu dau. 'predicted' (type head, che do trien khai); "
                             "'gold' (nhan chuan, oracle diagnostic); "
                             "'shuffled'/'wrong'/'const' = PHA tin hieu loai de DO TAC DONG — "
                             "so du doan thay doi so voi 'gold' cho biet dieu-kien-hoa-theo-loai "
                             "co anh huong gi den dau ra khong (tach khoi accuracy).")
    parser.add_argument('--repetition_penalty', type=float, default=1.0,
                        help='Repetition penalty to suppress repeated tokens (default: 1.0)')
    parser.add_argument('--num_samples', type=int, default=1,
                        help='Majority vote: sample N sequences, pick most frequent. 1=off (default)')
    parser.add_argument('--vote_temp', type=float, default=0.8,
                        help='Temperature for majority vote sampling (default: 0.8)')
    parser.add_argument('--use_synonyms', action='store_true',
                        help='Apply synonym normalization before computing EM/F1')
    parser.add_argument('--token_mode', type=str, default='set', choices=['set', 'multiset'],
                        help="Token overlap cho P/R/F1. 'set'=cong thuc paper (mac dinh), "
                             "'multiset'=kieu SQuAD. DO 2026-08-15: hai che do cho ket qua Y HET "
                             "tren ViVQA (0/3001 dong khac) vi khong dap an nao co token lap.")
    parser.add_argument('--use_constrained', action='store_true',
                        help='Constrained beam search: only generate tokens in training answer trie')
    parser.add_argument('--train_csv_for_trie', type=str, default='archive/train_split.csv',
                        help='Train CSV to build answer trie for constrained decoding')
    parser.add_argument('--output_csv', type=str, default=None)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Vision: {args.vision_model}")

    # Vision processor
    from transformers import AutoImageProcessor
    vision_processor = AutoImageProcessor.from_pretrained(args.vision_model)

    # Dataset
    print(f"\nLoading {args.csv_path}...")
    dataset = VQAGenDataset(
        csv_path=args.csv_path,
        image_folder=args.image_folder,
        vision_processor=vision_processor,
        tokenizer_name='vinai/bartpho-syllable',
        max_q_len=32,
        max_a_len=10,
        include_question_type=True,
        auto_detect_type=False,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=1,
        pin_memory=True
    )
    print(f"Loaded {len(dataset)} samples")
    
    # Load checkpoint
    print(f"\nLoading checkpoint...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    state_dict_keys = checkpoint['model_state_dict'].keys()
    
    # Detect features
    has_vision_lora = any('lora_A' in k or 'lora_B' in k for k in state_dict_keys if 'vision_encoder' in k)
    # Exclude vision_encoder keys to avoid false positive when vision LoRA key contains 'encoder.base_model.model'
    has_text_lora = any(k.startswith('encoder.base_model.model') for k in state_dict_keys)
    has_vision_gate = any('vision_gating' in k for k in state_dict_keys)
    has_delta_gate = any('vision_gating.orig_proj' in k for k in state_dict_keys)
    has_type_adapter = any('vision_adapter' in k for k in state_dict_keys)  # TypeConditionedVisionAdapter
    has_type_task = any(k.startswith('type_head') for k in state_dict_keys)   # 🔥 Type prediction head
    has_type_branch = any(k.startswith('type_branch') for k in state_dict_keys)  # 🔬 detached type branch
    has_blend_net = any('blend_net' in k for k in state_dict_keys)  # 🔬 learned blend target
    # 🔬 per-channel gate: gate_net lop cuoi xuat D (>1) thay vi 1 -> alpha [B,P,D]
    _sd = checkpoint['model_state_dict']
    _gn_last = [k for k in state_dict_keys if k.startswith('vision_gating.gate_net') and k.endswith('.weight')]
    has_gate_per_channel = bool(_gn_last) and _sd[sorted(_gn_last)[-1]].shape[0] > 1
    has_logits_bias = any(k.startswith('logits_bias') for k in state_dict_keys)  # 🔥 Type-aware logits bias
    
    # Detect fusion layers
    fusion_layer_indices = set()
    for key in state_dict_keys:
        if key.startswith('flamingo_fusion.'):
            parts = key.split('.')
            if len(parts) >= 2 and parts[1].isdigit():
                fusion_layer_indices.add(int(parts[1]))
    num_fusion_layers = max(fusion_layer_indices) + 1 if fusion_layer_indices else 4
    
    # 🔥 Detect fusion_type: CLI arg > checkpoint['args'] > fallback 'text2vision'
    saved_args = checkpoint.get('args', {})
    fusion_type = args.fusion_type or saved_args.get('fusion_type', 'text2vision')
    
    # 🔥 Detect LoRA ranks and alphas from checkpoint weights + saved_args
    text_lora_r = 16  # default
    vision_lora_r = 8  # default
    text_lora_alpha = saved_args.get('text_lora_alpha', 32)
    vision_lora_alpha = saved_args.get('vision_lora_alpha', 16)

    if has_text_lora:
        for key in state_dict_keys:
            # Must start with 'encoder.' to exclude vision_encoder.encoder.base_model.model... keys
            if key.startswith('encoder.base_model.model.layers.0.self_attn.q_proj.lora_A'):
                shape = checkpoint['model_state_dict'][key].shape
                text_lora_r = shape[0]
                break

    if has_vision_lora:
        # New format: vision_encoder.encoder.base_model.model.layers.X.self_attn.*.lora_A.*
        # Fallback to saved_args if key pattern not matched
        for key in state_dict_keys:
            if 'vision_encoder' in key and 'lora_A' in key:
                shape = checkpoint['model_state_dict'][key].shape
                vision_lora_r = shape[0]
                break
        else:
            vision_lora_r = saved_args.get('vision_lora_r', 8)
    
    print(f"\nCheckpoint features:")
    print(f"  Vision LoRA: {has_vision_lora}")
    if has_vision_lora:
        print(f"    └─ Rank: {vision_lora_r}")
    print(f"  Text LoRA: {has_text_lora}")
    if has_text_lora:
        print(f"    └─ Rank: {text_lora_r}")
    print(f"  Vision Gate: {has_vision_gate} (delta={has_delta_gate})")
    print(f"  Type Adapter: {has_type_adapter}")
    print(f"  Type Task Head: {has_type_task}")       # 🔥
    print(f"  Logits Bias: {has_logits_bias}")        # 🔥
    # Auto-detect architecture flags from saved checkpoint args
    use_siglip_pooler = saved_args.get('use_siglip_pooler', False)
    use_mean_pool_cls = saved_args.get('use_mean_pool_cls', False)
    use_attn_pool_cls = saved_args.get('use_attn_pool_cls', False)
    vision_gate_max_alpha = saved_args.get('vision_gate_max_alpha', 1.0)
    # 🔬 per-type alpha floor: buffer da luu trong state_dict -> tai lai chinh xac
    _mp_key = 'vision_gating.min_alpha_pertype'
    if _mp_key in checkpoint['model_state_dict']:
        vision_gate_min_alpha_pertype = checkpoint['model_state_dict'][_mp_key].tolist()
        print(f"  🔬 per-type alpha floor detected: {vision_gate_min_alpha_pertype}")
    else:
        vision_gate_min_alpha_pertype = None
    _xp_key = 'vision_gating.max_alpha_pertype'
    if _xp_key in checkpoint['model_state_dict']:
        vision_gate_max_alpha_pertype = checkpoint['model_state_dict'][_xp_key].tolist()
        print(f"  🔬 per-type alpha ceiling detected: {vision_gate_max_alpha_pertype}")
    else:
        vision_gate_max_alpha_pertype = None
    use_type_text_adapter = saved_args.get('use_type_text_adapter', False)
    type_text_adapter_bottleneck = saved_args.get('type_text_adapter_bottleneck', 64)

    print(f"  Fusion Layers: {num_fusion_layers}")
    print(f"  Fusion Type: {fusion_type}")
    print(f"  SigLIP pooler token: {use_siglip_pooler}")
    print(f"  Mean-pool text cls: {use_mean_pool_cls}")
    print(f"  Attn-pool text cls: {use_attn_pool_cls}")
    print(f"  Gate max alpha: {vision_gate_max_alpha}")
    print(f"  TypeSpecificTextAdapter: {use_type_text_adapter} (bottleneck={type_text_adapter_bottleneck})")
    print(f"  Text LoRA: r={text_lora_r}, alpha={text_lora_alpha}")
    print(f"  Vision LoRA: r={vision_lora_r}, alpha={vision_lora_alpha}")

    # Build model
    print(f"\nBuilding model...")
    model = DeterministicVQA(
        vision_model_name=args.vision_model,
        bartpho_model_name='vinai/bartpho-syllable',
        num_fusion_layers=num_fusion_layers,
        fusion_type=fusion_type,
        num_heads=8,
        dropout=0.1,
        gradient_checkpointing=False,
        use_vision_lora=has_vision_lora,
        vision_lora_r=vision_lora_r,
        vision_lora_alpha=vision_lora_alpha,
        vision_lora_dropout=0.1,
        use_decoder_lora=saved_args.get('use_decoder_lora', False),
        decoder_lora_r=saved_args.get('decoder_lora_r', 16),
        decoder_lora_alpha=saved_args.get('decoder_lora_alpha', 32),
        decoder_lora_dropout=saved_args.get('decoder_lora_dropout', 0.1),
        use_text_lora=has_text_lora,
        text_lora_r=text_lora_r,
        text_lora_alpha=text_lora_alpha,
        text_lora_dropout=0.1,
        use_vision_gate=has_vision_gate,
        vision_gate_init=saved_args.get('vision_gate_init', 1.5),
        vision_gate_min_alpha=saved_args.get('vision_gate_min_alpha', 0.35),
        vision_gate_min_alpha_pertype=vision_gate_min_alpha_pertype,
        vision_gate_max_alpha_pertype=vision_gate_max_alpha_pertype,
        vision_gate_max_alpha=vision_gate_max_alpha,
        use_delta_gate=has_delta_gate,
        use_type_task=has_type_task,
        type_branch_detach=has_type_branch,
        gate_blend_learned=has_blend_net,
        gate_no_type_emb=saved_args.get('gate_no_type_emb', False),
        gate_no_text_cls=saved_args.get('gate_no_text_cls', False),
        gate_blend_vorig=saved_args.get('gate_blend_vorig', False),
        patch_self_attn=saved_args.get('patch_self_attn', False),
        psa_heads=saved_args.get('psa_heads', 8),
        gate_alpha_budget=saved_args.get('gate_alpha_budget', False),
        gate_budget_init=saved_args.get('gate_budget_init', 0.72),
        gate_pertype_net=saved_args.get('gate_pertype_net', False),
        gate_type_blind=saved_args.get('gate_type_blind', False),
        type_from_gate_lambda=saved_args.get('type_from_gate_lambda', 0.0),
        gate_vision_layer=saved_args.get('gate_vision_layer', -1),
        vision_backbone_layer=saved_args.get('vision_backbone_layer', -1),
        gate_per_channel=has_gate_per_channel,
        gate_blend_l6=saved_args.get('gate_blend_l6', False),
        gate_l6_fuse=any('l6_fuse' in k for k in state_dict_keys) or saved_args.get('gate_l6_fuse', False),
        gate_l6_fuse_bottleneck=saved_args.get('gate_l6_fuse_bottleneck', 256),
        vision_l6_enrich=any('l6_enrich' in k for k in state_dict_keys) or saved_args.get('vision_l6_enrich', False),
        type_moe=any('type_experts' in k for k in state_dict_keys),
        type_moe_bottleneck=saved_args.get('type_moe_bottleneck', 256),
        type_moe_soft=saved_args.get('type_moe_soft', False),
        use_logits_bias=has_logits_bias,
        use_type_adapter=has_type_adapter,
        type_adapter_rank=64,
        type_adapter_bias=2.0,
        tcvg_norm_type_emb=saved_args.get('tcvg_norm_type_emb', False),
        tcvg_type_null=saved_args.get('tcvg_type_null', False),
        tcvg_gate_mode=saved_args.get('tcvg_gate_mode', 'blend'),
        tcvg_two_layer=saved_args.get('tcvg_two_layer', False),
        decoder_vision_only=saved_args.get('decoder_vision_only', False),
        tcvg_topk=saved_args.get('tcvg_topk', 0),
        tcvg_topk_random=saved_args.get('tcvg_topk_random', False),
        tcvg_fusion_gate=saved_args.get('tcvg_fusion_gate', False),
        tcvg_fg_2pass=saved_args.get('tcvg_fg_2pass', False),
        # 🔬 Cac co THEM SAU — phai truyen day du, neu khong model dung luc eval KHAC model
        # da train va load_state_dict(strict=False) se AM THAM vut trong so tuong ung.
        tcvg_type_bias=saved_args.get('tcvg_type_bias', False),
        tcvg_type_ctx=saved_args.get('tcvg_type_ctx', False),
        tcvg_ln_mode=saved_args.get('tcvg_ln_mode', 'post'),
        tcvg_attn_gate=saved_args.get('tcvg_attn_gate', False),
        tcvg_refine_gate=saved_args.get('tcvg_refine_gate', False),
        tcvg_proto_gate=saved_args.get('tcvg_proto_gate', False),
        tcvg_global_scalar_gate=saved_args.get('tcvg_global_scalar_gate', False),
        gca_box_tokens=saved_args.get('gca_box_tokens', False),
        box_class_n=(saved_args.get('box_class_n',81) if saved_args.get('box_class_lambda',0.0)>0 else 0),
        box_ground=(saved_args.get('box_ground_lambda',0.0)>0 or saved_args.get('box_count_lambda',0.0)>0),
        # So lop dap an phai lay tu CHINH CHECKPOINT, khong doan tu CSV: train dung
        # train_split_original.csv (331 lop) con so 328 gan cung o day la cua train_split.csv,
        # nen acls05/acls20 crash "[331,1024] vs [328,1024]". Hinh dang tensor la nguon dung.
        num_answer_classes=_n_answer_classes_from_ckpt(checkpoint, saved_args),
        gate_spatial_pertype=([float(x) for x in saved_args['gate_spatial_pertype'].split(',')] if saved_args.get('gate_spatial_pertype') else None),
        gate_box_content=saved_args.get('gate_box_content', False),
        box_max_inst=saved_args.get('box_max_inst', 32),
        box_class_vocab=saved_args.get('box_class_vocab', 0),
        tcvg_spatial_blend=saved_args.get('tcvg_spatial_blend', False),
        tcvg_dynamic_peek=saved_args.get('tcvg_dynamic_peek', False),
        tcvg_alpha_from_gca=saved_args.get('tcvg_alpha_from_gca', False),
        gate_layerscale_pertype=(saved_args.get('gate_layerscale_pertype', False)
                                 or bool(getattr(args, 'gate_beta_override', None))),
        gate_layerscale_init=saved_args.get('gate_layerscale_init', 1.0),
        gca_strength=saved_args.get('gca_strength', 1.0),
        text_path_dropout=saved_args.get('text_path_dropout', 0.0),
        concat_fusion=saved_args.get('concat_fusion', False),
        summary_token=saved_args.get('summary_token', False),
        slot_attn=saved_args.get('slot_attn', False),
        num_slots=saved_args.get('num_slots', 4),
        slot_init_std=saved_args.get('slot_init_std', 0.02),
        slot_tanh_gate=saved_args.get('slot_tanh_gate', False),
        slot_stage=saved_args.get('slot_stage', 'post'),
        slot_no_type=saved_args.get('slot_no_type', False),
        decoder_pool_vision=saved_args.get('decoder_pool_vision', 0),
        use_type_codebook=saved_args.get('use_type_codebook', False),
        codebook_size=saved_args.get('codebook_size', 4),
        codebook_beta=saved_args.get('codebook_beta', 0.25),
        codebook_lambda=saved_args.get('codebook_lambda', 0.1),
        use_siglip_pooler=use_siglip_pooler,
        use_mean_pool_cls=use_mean_pool_cls,
        use_attn_pool_cls=use_attn_pool_cls,
        use_type_text_adapter=use_type_text_adapter,
        type_text_adapter_bottleneck=type_text_adapter_bottleneck,
    ).to(device)
    
    # Load weights
    _res = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    if getattr(args, 'text_only', False):
        # zero VISION_FEATURES (tensor sau encoder), khop voi model.py:3038 luc train.
        # KHAC --zero_vision von zero PIXEL_VALUES -> SigLIP van chay tren anh den.
        model.text_only_mode = True
        print('  [text_only] zero hoa vision_features — khop che do train')
    print(f"Loaded weights from epoch {checkpoint.get('epoch', 'N/A')}")
    if getattr(args,'probe_topk',0) and args.probe_topk>0:
        model.tcvg_topk = int(args.probe_topk); model.tcvg_topk_random = bool(args.probe_topk_random)
        model.tcvg_topk_bottom = bool(args.probe_topk_bottom)
        print(f'[PROBE] tcvg_topk={model.tcvg_topk} random={model.tcvg_topk_random}')
    # 🔬 ORACLE BETA OVERRIDE (eval-time, khong train lai): ep beta_type = gia tri chi dinh de
    # quet cuong do gate per-type tren mot checkpoint da train. beta=1 -> dung T2, beta=0 -> T0.
    _bo = getattr(args, 'gate_beta_override', None)
    if _bo and getattr(model, 'vision_gating', None) is not None \
            and getattr(model.vision_gating, 'gate_layerscale_pertype', False):
        import torch as _t
        _vals = [float(x) for x in str(_bo).split(',')]
        with _t.no_grad():
            model.vision_gating.gate_ls.copy_(_t.tensor(_vals, dtype=model.vision_gating.gate_ls.dtype))
        print(f"[ORACLE] gate_ls (beta per-type [OBJ,COUNT,COLOR,LOC]) = {_vals}")
    # 🔬 strict=False AM THAM vut trong so khi model dung luc eval khac model da train.
    # Da tung xay ra that: eval.py khong truyen use_type_codebook -> type_codebook=None ->
    # 3 tensor codebook bi vut -> gate chay voi type_ids=None (khong dieu kien hoa loai gi)
    # va van in ra EM binh thuong. Bao that to de khong lap lai.
    _unexp = [k for k in getattr(_res, 'unexpected_keys', []) if 'teacher' not in k]
    _miss = [k for k in getattr(_res, 'missing_keys', []) if 'teacher' not in k]
    if _unexp or _miss:
        print("=" * 70)
        print("!! CANH BAO: kien truc luc eval KHONG khop checkpoint")
        if _unexp:
            print(f"   {len(_unexp)} trong so trong checkpoint BI VUT: {_unexp[:8]}")
        if _miss:
            print(f"   {len(_miss)} tham so dung KHOI TAO NGAU NHIEN: {_miss[:8]}")
        print("   -> ket qua eval KHONG dung. Kiem lai cac co truyen vao DeterministicVQA().")
        print("=" * 70)

    if args.force_gate_open and getattr(model, 'use_vision_gate', False) and model.vision_gating is not None:
        # alpha = min_alpha + (max_alpha - min_alpha) * sigmoid(...)
        # Dat ca hai = 1.0 -> alpha = 1 chinh xac, GIU NGUYEN phep chieu v_proj va layer_norm.
        # KHONG dung forward hook tra ve input: cach do bo qua v_proj + layer_norm va pha
        # huy model (do thu duoc 3.03% EM, la artifact chu khong phai ket qua).
        model.vision_gating.min_alpha = 1.0
        model.vision_gating.max_alpha = 1.0
        print('[force_gate_open] min_alpha = max_alpha = 1.0 -> alpha = 1 moi patch')

    if args.flatten_alpha and getattr(model, 'use_vision_gate', False) and model.vision_gating is not None:
        model.vision_gating.flatten_alpha = True
        print('[flatten_alpha] alpha := mean_patch(alpha) -> mat chon loc, giu bien do')

    # ── Cac che do do dac cho experiment.md (muc 6, 7, 8) ──────────────────
    if args.param_count or args.measure_latency or args.dump_gate_stats:
        run_measurement_mode(args, model, dataloader, device)
        return

    # Evaluate
    mode_str = (f"majority_vote n={args.num_samples} temp={args.vote_temp}"
                if args.num_samples > 1 else f"num_beams={args.num_beams}")
    print(f"\nEvaluating... ({mode_str}, max_length={args.max_length}, "
          f"synonyms={'on' if args.use_synonyms else 'off'})")
    prefix_trie = None
    valid_answers_set = None
    if args.use_constrained:
        prefix_trie = build_answer_trie(args.train_csv_for_trie, model.tokenizer)
        valid_answers_set = build_valid_answers_set(args.train_csv_for_trie)

    _alpha_file = None
    if getattr(args, 'alpha_from_file', None):
        import numpy as _np
        _z = _np.load(args.alpha_from_file)
        _alpha_file = torch.from_numpy(_z['alpha'].astype('float32'))
        _nv = int((~torch.isnan(_alpha_file[:, -1])).sum())
        print(f"[alpha_from_file] {tuple(_alpha_file.shape)} tu {args.alpha_from_file} | "
              f"{_nv}/{_alpha_file.size(0)} hang co nhan (con lai NaN = giu alpha cua model)")

    _region_map_lookup = None
    if getattr(args, 'tcvg_spatial_blend_region_map', None):
        import pickle as _pickle
        _region_map_lookup = _pickle.load(open(args.tcvg_spatial_blend_region_map, 'rb'))
        print(f"[RegionMap] Da tai {len(_region_map_lookup)} img_id co annotation COCO that.")

    # Chan dung che do that bai am tham: checkpoint duoc train voi --gate_box_content nhung eval
    # KHONG truyen --tcvg_spatial_blend_region_map -> region_map=None -> box_feat=None -> kenh box
    # bien mat khong bao loi, va con so do duoc la cua mot model khac. Day dung la kieu loi da lam
    # 5 co gate thanh no-op im lang truoc day, nen bao thanh loi cung chu khong phai canh bao.
    if (saved_args.get('gate_box_content', False) or saved_args.get('gca_box_tokens', False)) \
            and _region_map_lookup is None:
        raise SystemExit(
            "[gate_box_content] Checkpoint nay duoc train voi kenh noi dung box, nhung eval khong "
            "duoc truyen --tcvg_spatial_blend_region_map. Neu chay tiep thi box_feat=None va ket qua "
            "KHONG phai cua model da train. Truyen: "
            "--tcvg_spatial_blend_region_map patch_region_map.pkl")

    # 🔬 TRAN TREN CUA HO HAM: toi uu tham so gate (mot ham dung chung, chi doc (v,q)) tren test
    # set voi nhan gold. Khac --oracle_alpha o cho nay: oracle_alpha tim GIA TRI alpha rieng tung
    # mau (197 bac tu do/mau, dung dap an cua chinh mau do) -> tran cua HO GIA TRI. Cai nay tim
    # HAM (tham so chia se moi mau) -> tran cua HO HAM, tuc "neu co cong thuc thi duoc bao nhieu".
    # Duoc phep overfit thang vao test => can tren CHAT cua moi gate hoc tu (anh, cau hoi).
    if getattr(args, 'oracle_gate_fit', 0) > 0:
        if getattr(model, 'vision_gating', None) is None:
            raise SystemExit('[oracle_gate_fit] checkpoint khong co vision_gating.')
        for _p in model.parameters():
            _p.requires_grad_(False)
        _gp = [p for p in model.vision_gating.parameters()]
        for _p in _gp:
            _p.requires_grad_(True)
        _n = sum(p.numel() for p in _gp)
        print(f'[oracle_gate_fit] toi uu {_n:,} tham so cua vision_gating tren TEST '
              f'({args.oracle_gate_fit} epoch, lr={args.oracle_gate_lr}) -- CAN TREN, khong phai phuong phap')
        _opt = torch.optim.AdamW(_gp, lr=args.oracle_gate_lr, weight_decay=args.oracle_gate_wd)
        # Fit tren CSV rieng (train) neu duoc chi dinh -> khong dung nhan test, do duoc kha nang
        # TONG QUAT HOA cua cong thuc. Neu khong -> fit tren test = can tren long (co overfit).
        _fit_loader = dataloader
        if args.oracle_gate_fit_csv:
            _fit_ds = VQAGenDataset(
                csv_path=args.oracle_gate_fit_csv,
                image_folder=args.oracle_gate_fit_images or args.image_folder,
                vision_processor=vision_processor, tokenizer_name='vinai/bartpho-syllable',
                max_q_len=32, max_a_len=10, include_question_type=True, auto_detect_type=False)
            _fit_loader = DataLoader(_fit_ds, batch_size=args.batch_size, shuffle=True,
                                     num_workers=1, pin_memory=True)
            print(f'[oracle_gate_fit] fit tren {args.oracle_gate_fit_csv} ({len(_fit_ds)} mau) '
                  f'-> do tren {args.csv_path}: KHONG dung nhan test')
        # 🔬 VAL de dung dung luc: lan chay dau (4 epoch, lr 1e-3, khong regularization) da
        # OVERFIT ro rang (CE tap fit 0.0334 -> 0.0091, test EM 70.51 < ca T0 71.64, va LOCATION
        # tut manh nhat du la loai gate PHAI dung yen). Khong co val thi con so do luong dac tinh
        # cua GIAO THUC FIT chu khong phai cua cau hoi "co ton tai cong thuc khong".
        _val_loader = None
        if args.oracle_gate_val_csv:
            _val_ds = VQAGenDataset(
                csv_path=args.oracle_gate_val_csv,
                image_folder=args.oracle_gate_val_images or args.oracle_gate_fit_images or args.image_folder,
                vision_processor=vision_processor, tokenizer_name='vinai/bartpho-syllable',
                max_q_len=32, max_a_len=10, include_question_type=True, auto_detect_type=False)
            _val_loader = DataLoader(_val_ds, batch_size=args.batch_size, shuffle=False,
                                     num_workers=1, pin_memory=True)
            print(f'[oracle_gate_fit] val = {args.oracle_gate_val_csv} ({len(_val_ds)} mau), '
                  f'chon epoch theo val CE, patience={args.oracle_gate_patience}, '
                  f'wd={args.oracle_gate_wd}')

        def _val_ce():
            _s, _c = 0.0, 0
            with torch.no_grad():
                for _vb in _val_loader:
                    _vl = _vb['labels'].to(device)
                    _vo = model(pixel_values=_vb['pixel_values'].to(device),
                                input_ids=_vb['input_ids'].to(device),
                                attention_mask=_vb['attention_mask'].to(device),
                                labels=_vl,
                                question_types=(_vb['question_type'].to(device).long()
                                                if _vb.get('question_type') is not None else None))
                    _vg = _vo.answer_logits
                    _s += F.cross_entropy(_vg.reshape(-1, _vg.size(-1)), _vl.reshape(-1),
                                          ignore_index=-100).item()
                    _c += 1
            return _s / max(_c, 1)

        import copy as _copy
        _best_val, _best_state, _bad, _best_ep = float('inf'), None, 0, 0
        model.eval()   # giu dropout/BN o che do eval: chi hoc tham so gate, khong doi gi khac
        if _val_loader is not None:
            _best_val = _val_ce(); _best_state = _copy.deepcopy(model.vision_gating.state_dict())
            print(f'[oracle_gate_fit] epoch 0 (gate goc): val CE = {_best_val:.4f}')
        for _ep in range(args.oracle_gate_fit):
            _tot, _nb = 0.0, 0
            for _b in tqdm(_fit_loader, desc=f'[gate_fit] epoch {_ep + 1}/{args.oracle_gate_fit}'):
                _opt.zero_grad(set_to_none=True)
                _lb = _b['labels'].to(device)
                _out = model(pixel_values=_b['pixel_values'].to(device),
                             input_ids=_b['input_ids'].to(device),
                             attention_mask=_b['attention_mask'].to(device),
                             labels=_lb,
                             question_types=(_b['question_type'].to(device).long()
                                             if _b.get('question_type') is not None else None))
                _lg = _out.answer_logits
                _l = F.cross_entropy(_lg.reshape(-1, _lg.size(-1)), _lb.reshape(-1), ignore_index=-100)
                _l.backward()
                torch.nn.utils.clip_grad_norm_(_gp, 1.0)
                _opt.step()
                _tot += _l.item(); _nb += 1
            _msg = f'[oracle_gate_fit] epoch {_ep + 1}: CE tren tap FIT = {_tot / max(_nb, 1):.4f}'
            if _val_loader is not None:
                _v = _val_ce()
                _msg += f'   val CE = {_v:.4f}'
                if _v < _best_val - 1e-5:
                    _best_val, _bad, _best_ep = _v, 0, _ep + 1
                    _best_state = _copy.deepcopy(model.vision_gating.state_dict())
                    _msg += '  <- tot nhat'
                else:
                    _bad += 1
                    _msg += f'  (xau hon, {_bad}/{args.oracle_gate_patience})'
            print(_msg)
            if _val_loader is not None and _bad >= args.oracle_gate_patience:
                print(f'[oracle_gate_fit] dung som o epoch {_ep + 1}')
                break
        if _val_loader is not None and _best_state is not None:
            model.vision_gating.load_state_dict(_best_state)
            print(f'[oracle_gate_fit] khoi phuc gate cua epoch {_best_ep} (val CE {_best_val:.4f}). '
                  f'epoch 0 = gate goc, nen neu best_ep=0 thi FIT KHONG CAI THIEN DUOC GI '
                  f'-> ket qua se bang dung T2, va do CHINH LA cau tra loi.')
        for _p in _gp:
            _p.requires_grad_(False)

    results = evaluate(model, dataloader, device, model.tokenizer,
                       token_mode=args.token_mode,
                       num_beams=args.num_beams, repetition_penalty=args.repetition_penalty,
                       max_length=args.max_length, use_synonyms=args.use_synonyms,
                       num_samples=args.num_samples, vote_temp=args.vote_temp,
                       alpha_file=_alpha_file, dump_first_logits=args.dump_first_logits,
                       dump_seq_logprob=args.dump_seq_logprob, dump_model_alpha=args.dump_model_alpha,
                       prefix_trie=prefix_trie, valid_answers_set=valid_answers_set,
                       type_mode=args.type_mode, legacy_beam=args.legacy_beam,
                       zero_vision=args.zero_vision, diag_harm=args.diag_harm,
                       region_map_lookup=_region_map_lookup,
                       oracle_alpha=args.oracle_alpha, oracle_steps=args.oracle_steps,
                       oracle_lr=args.oracle_lr, dump_oracle_alpha=args.dump_oracle_alpha)
    
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    # GIU NGUYEN chuoi/format hai dong nay: 40+ run_*.sh grep "Exact Match" va evaluator.py
    # parse bang regex. Precision/Recall them o dong RIENG nen an toan.
    print(f"Exact Match: {results['exact_match']:.2f}%")
    print(f"F1 Score: {results['f1_score']:.2f}%")
    # KHONG duoc chua chuoi "Exact Match" o dong nay: 40+ script dung
    # `grep -a 'Exact Match' | tail -1` nen se bat nham dong nay thay vi dong that.
    print(f"Accuracy: {results['exact_match']:.2f}%   (bang EM — moi cau chi co MOT dap an dung)")
    print(f"Precision: {results['precision']:.2f}%")
    print(f"Recall: {results['recall']:.2f}%")
    print(f"(token_mode={getattr(args, 'token_mode', 'set')})")
    
    # Per-type breakdown
    if results.get('per_type'):
        print(f"\nPer Question Type:")
        # EM va F1 PHAI la hai cot so DAU TIEN: run_*.sh grep "^  (COLOR|COUNT|...)" roi doc
        # hai so dau. Prec/Rec dat SAU F1 nen khong pha gi.
        print(f"  {'Type':<12} {'EM':<8} {'F1':<8} {'Prec':<8} {'Rec':<8} {'Count':<8}")
        print(f"  {'-'*56}")
        for q_type in sorted(results['per_type'].keys()):
            type_data = results['per_type'][q_type]
            print(f"  {q_type:<12} {type_data['exact_match']:<8.2f} {type_data['f1_score']:<8.2f} "
                  f"{type_data['precision']:<8.2f} {type_data['recall']:<8.2f} {type_data['count']:<8}")

    # Type prediction accuracy (TCVG quality indicator)
    if results.get('type_pred_accuracy'):
        tpa = results['type_pred_accuracy']
        print(f"\nType Prediction Accuracy (TypePredictionHead → TCVG quality):")
        print(f"  Overall: {tpa['overall']:.2f}%")
        print(f"  {'Type':<12} {'Acc':<8}")
        print(f"  {'-'*20}")
        for t in sorted(tpa['per_type'].keys()):
            print(f"  {t:<12} {tpa['per_type'][t]:<8.2f}")

    # 🔬 Bao cao codebook khong giam sat: cum tu phat hien co trung voi loai that khong?
    # Cum KHONG phai nhan loai — chi so cum la tuy y. Nen do bang PURITY (gan moi cum cho
    # loai chiem da so trong no) va NMI (bat tuong quan bat ke cach danh so).
    if getattr(args, 'codebook_report', False):
        gold = results.get('question_types') or []
        clus = results.get('pred_question_types') or []
        if len(gold) == len(clus) and len(gold) > 0:
            from collections import Counter, defaultdict
            import math
            table = defaultdict(Counter)
            for c, g in zip(clus, gold):
                table[c][g] += 1
            n = len(gold)
            types = sorted(set(gold))
            print("\nCODEBOOK KHONG GIAM SAT — cum tu phat hien vs loai that")
            hdr = f"  {'cum':<8}" + "".join(f"{t:>12}" for t in types) + f"{'tong':>8}{'gan cho':>12}"
            print(hdr)
            print("  " + "-" * (len(hdr) - 2))
            correct = 0
            for c in sorted(table.keys(), key=str):
                row = table[c]
                tot = sum(row.values())
                best, bestn = row.most_common(1)[0]
                correct += bestn
                print(f"  {str(c):<8}" + "".join(f"{row.get(t,0):>12}" for t in types)
                      + f"{tot:>8}{best:>12}")
            purity = correct / n * 100
            # NMI
            hc = -sum((sum(v.values())/n) * math.log(sum(v.values())/n + 1e-12) for v in table.values())
            gc = Counter(gold)
            hg = -sum((v/n) * math.log(v/n + 1e-12) for v in gc.values())
            hcg = -sum((cnt/n) * math.log((cnt/n) / (sum(v.values())/n) + 1e-12)
                       for v in table.values() for cnt in v.values())
            mi = hg - hcg
            nmi = mi / max((hc * hg) ** 0.5, 1e-12)
            print(f"  So cum thuc su duoc dung: {len(table)} / {args.codebook_size_report}")
            print(f"  Purity: {purity:.2f}%   NMI: {nmi:.4f}")
            print("  (Purity cao = cum tu phat hien trung voi taxonomy loai -> khong can nhan)")

    print("="*80)
    
    # Save CSV
    if args.output_csv:
        try:
            import pandas as pd
            
            # Prepare data (include question_type)
            _prf_rows = [compute_prf(p, g, use_synonyms=args.use_synonyms,
                                     token_mode=args.token_mode)
                         for p, g in zip(results['predictions'], results['ground_truths'])]
            save_data = {
                'question': results['questions'],
                'prediction': results['predictions'],
                'ground_truth': results['ground_truths'],
                'question_type': results['question_types'],
                'pred_question_type': (results['pred_question_types']
                                       if len(results.get('pred_question_types', [])) == len(results['questions'])
                                       else [''] * len(results['questions'])),
                'exact_match': [compute_exact_match(p, g, use_synonyms=args.use_synonyms) for p, g in zip(results['predictions'], results['ground_truths'])],
                'precision': [r[0] for r in _prf_rows],
                'recall': [r[1] for r in _prf_rows],
                'f1_score': [r[2] for r in _prf_rows]
            }
            if len(results.get('wrong_targets', [])) == len(results['questions']):
                # 🔬 do DIEU KHIEN DUOC: alpha da duoc khop de ep model noi ra dap an SAI nay.
                # steer_hit = model that su noi ra no -> alpha la mot num dieu khien.
                _wt = results['wrong_targets']
                save_data['wrong_target'] = _wt
                save_data['steer_hit'] = [
                    1 if compute_exact_match(p, w, use_synonyms=args.use_synonyms) else 0
                    for p, w in zip(results['predictions'], _wt)]
            if len(results.get('alpha_mean', [])) == len(results['questions']):
                save_data['alpha_mean'] = results['alpha_mean']
            if len(results.get('ce_on', [])) == len(results['questions']):
                save_data['ce_on'] = results['ce_on']; save_data['ce_off'] = results['ce_off']
                save_data['h_on'] = results['h_on']; save_data['h_off'] = results['h_off']
            
            df = pd.DataFrame(save_data)
            
            # Ensure directory exists
            import os
            output_dir = os.path.dirname(args.output_csv) or '.'
            os.makedirs(output_dir, exist_ok=True)
            
            # Save CSV
            df.to_csv(args.output_csv, index=False, encoding='utf-8')
            
            # Verify file was saved
            if os.path.exists(args.output_csv):
                file_size = os.path.getsize(args.output_csv)
                print(f"\n✅ Saved to {args.output_csv}")
                print(f"   File size: {file_size:,} bytes ({len(df)} rows)")
            else:
                print(f"\n❌ ERROR: File was not created at {args.output_csv}")
                
        except Exception as e:
            print(f"\n❌ ERROR saving CSV: {e}")
            print(f"   Attempted path: {args.output_csv}")
            
            # Fallback: save to current directory
            try:
                fallback_path = 'results_fallback.csv'
                df.to_csv(fallback_path, index=False, encoding='utf-8')
                print(f"   Saved to fallback location: {fallback_path}")
            except Exception as e2:
                print(f"   Fallback also failed: {e2}")


if __name__ == '__main__':
    main()
