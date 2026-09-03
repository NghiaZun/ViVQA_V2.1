"""SCREEN TCVG-v2 — co THONG TIN THI GIAC nao phan biet duoc ung vien top-1 va top-2 khong?

Cau hoi quyet dinh ca chuong trinh TCVG-v2, dat TRUOC khi viet bat cu kien truc nao.

Moi phep do truoc day hoi: "vision co du doan duoc ORACLE co cuu duoc mau nay khong" -> khong
(moi khoang tin cay deu chua 0). Do la tinh chat cua TOAN TU, khong phai cua THE GIOI.
O day hoi cau khac han: trong hai ung vien model tu xep cao nhat, vision co biet cai nao DUNG khong?

  y = 1 neu gold == top1 (model dung), y = 0 neu gold == top2 (model sai nhung cuu duoc)

Nam nhanh LONG NHAU, CV 5-fold theo MAU, lap lai 5 seed CV, bootstrap CI tren so gia tang:
  A  decoder thuan             : diem ung vien, margin, entropy
  B  A + loai + prior ung vien : one-hot type, log tan suat train, do dai, cos(q_gate, e_cand)
                                 <- MOC THAM CHIEU = "decoder + type" cua tieu chi thanh cong so 3
  C  B + doc thi giac          : chi nhung dac trung SINH TU dam may patch sau TCVG
  D  vision thuan
  E  DOI CHUNG NULL            : C nhung khoi vision lay tu MAU KHAC
  F  DOI CHUNG TYPE XAO        : C nhung one-hot type bi hoan vi

QUY TAC DOC (ghi TRUOC khi chay, PREREG_tcvg_v2.md muc 5):
  AUROC(C)-AUROC(B): CI 95% khong chua 0 VA diem uoc luong >= +0.03 -> con thong tin thi giac
                     chua khai thac -> dung nguyen mau R1.
  CI chua 0        -> ket luan chinh xac: "GCA da chua toan bo bang chung thi giac ma decoder
                     dung duoc" -> R1 chet TRUOC khi cai dat.
  E phai ~= B va F khong duoc thang C, neu khong phep do vo hieu.
"""
import sys, argparse, unicodedata as ud
import torch, torch.nn.functional as F, numpy as np, pandas as pd
sys.path.insert(0, 'src')

norm = lambda s: ud.normalize('NFC', str(s)).strip().lower()


def build_model(ckpt, dev):
    from model import DeterministicVQA
    ck = torch.load(ckpt, map_location='cpu', weights_only=False)
    sa = ck['args']; sa = sa if isinstance(sa, dict) else vars(sa)
    sd = ck['model_state_dict']; K = list(sd.keys())
    tlr = next((sd[k].shape[0] for k in K
                if k.startswith('encoder.base_model.model.layers.0.self_attn.q_proj.lora_A')), 16)
    m = DeterministicVQA(
        vision_model_name=sa.get('vision_model'), bartpho_model_name='vinai/bartpho-syllable',
        num_fusion_layers=sa.get('num_fusion_layers', 2),
        fusion_type=sa.get('fusion_type', 'text2vision'),
        use_text_lora=True, text_lora_r=tlr, text_lora_alpha=sa.get('text_lora_alpha', 32),
        use_vision_gate=True, vision_gate_init=sa.get('vision_gate_init', 1.5),
        vision_gate_min_alpha=sa.get('vision_gate_min_alpha', 0.0),
        vision_gate_max_alpha=sa.get('vision_gate_max_alpha', 1.0),
        use_type_task=any(k.startswith('type_head.') or k.startswith('type_classifier.') for k in K),
        use_siglip_pooler=sa.get('use_siglip_pooler', True)).to(dev).eval()
    r = m.load_state_dict(sd, strict=False)
    assert not [k for k in r.missing_keys if 'teacher' not in k], r.missing_keys[:5]
    for p in m.parameters():
        p.requires_grad_(False)
    return m, sa


def stage_dump(a):
    from dataset import VQAGenDataset
    from torch.utils.data import DataLoader, Subset
    from transformers import AutoProcessor
    DEV = 'cuda'
    m, sa = build_model(a.checkpoint, DEV)
    tok = m.tokenizer

    C = {}
    m.vision_gating.layer_norm.register_forward_hook(
        lambda mo, i, o: C.__setitem__('post', o.detach()))
    m.vision_gating.gate_net.register_forward_pre_hook(
        lambda mo, i: C.__setitem__('gin', i[0].detach()))
    # 🔬 patch SigLIP THO (da chieu 1024D) TRUOC Flamingo — nguon 'raw'
    m.vision_proj.register_forward_hook(
        lambda mo, i, o: C.__setitem__('raw', o.detach()))

    tr = pd.read_csv(a.train_csv); tr['an'] = tr.answer.map(norm)
    TVOC = {int(t): sorted(set(g.an)) for t, g in tr.groupby('type')}
    FREQ = tr.an.value_counts().to_dict()
    LB, CEMB, CLEN = {}, {}, {}
    W = m.lm_head.weight.detach().float()                      # [V, D] tied voi embedding
    for t, voc in TVOC.items():
        e = tok(voc, return_tensors='pt', padding='max_length', truncation=True, max_length=10)
        x = e.input_ids.clone()
        real = (x != tok.pad_token_id) & (x != tok.bos_token_id) & (x != tok.eos_token_id)
        emb = torch.stack([W[x[i][real[i]]].mean(0) if real[i].any() else W[x[i]].mean(0)
                           for i in range(x.size(0))])
        CEMB[t] = F.normalize(emb, dim=-1).to(DEV)             # [Nc, D]
        CLEN[t] = real.sum(1).float().numpy()
        xx = x.to(DEV); xx[xx == tok.pad_token_id] = -100
        LB[t] = xx

    te = pd.read_csv(a.test_csv); gold = te.answer.map(norm)
    vp = AutoProcessor.from_pretrained(sa.get('vision_model'))
    ds = VQAGenDataset(csv_path=a.test_csv, image_folder=a.image_folder, vision_processor=vp,
                       tokenizer_name='vinai/bartpho-syllable', max_q_len=32, max_a_len=10,
                       include_question_type=True, auto_detect_type=False)

    def logp(lg, lb):
        return -F.cross_entropy(lg.reshape(-1, lg.size(-1)).float(), lb.reshape(-1),
                                ignore_index=-100, reduction='none').view(lb.shape).sum(1)

    rng = np.random.RandomState(0)
    idxs = []
    for t, g in te.groupby('type'):
        ii = list(g.index)
        if a.per_type and len(ii) > a.per_type:
            ii = list(rng.choice(ii, a.per_type, replace=False))
        idxs += ii
    idxs = sorted(int(i) for i in idxs)
    print(f'{len(idxs)} mau, tu vung theo loai: '
          f'{ {t: len(v) for t, v in TVOC.items()} }', flush=True)

    ROWS, VIS, POOL = [], [], []
    for n, j in enumerate(idxs):
        t = int(te.type.iloc[j])
        if gold.iloc[j] not in TVOC[t]:
            continue
        L = LB[t]; voc = TVOC[t]
        b = next(iter(DataLoader(Subset(ds, [j]), batch_size=1)))
        pv = b['pixel_values'].to(DEV); ii = b['input_ids'].to(DEV)
        am = b['attention_mask'].to(DEV); qt = b['question_type'].to(DEV).long()
        s = torch.empty(L.size(0), device=DEV)
        with torch.no_grad():
            for st in range(0, L.size(0), a.chunk):
                x = L[st:st + a.chunk]; k = x.size(0)
                o = m(pixel_values=pv.expand(k, -1, -1, -1), input_ids=ii.expand(k, -1),
                      attention_mask=am.expand(k, -1), labels=x, question_types=qt.expand(k))
                s[st:st + k] = logp(o.answer_logits, x)
        s = s.float()
        order = torch.argsort(s, descending=True)
        gi = voc.index(gold.iloc[j])
        rank = int((order == gi).nonzero()[0, 0]) + 1
        if rank > a.topk:
            continue                                   # ngoai che do van hanh
        # top-1 vs doi thu manh nhat: voi topk>2, doi thu la ung vien cao nhat KHONG PHAI gold
        c1 = int(order[0])
        c2 = int(order[1]) if c1 == gi else gi     # doi thu = gold neu gold chua dan dau
        if c1 != gi:
            c1, c2 = c1, gi                        # c1 = du doan hien tai, c2 = gold
        y = int(rank == 1)

        # ---- khoi VISION: dam may patch SAU TCVG (dung cai decoder cross-attend vao) ----
        _src = C['raw'] if a.vision_source == 'raw' else C['post']
        Vp = F.normalize(_src[0].float(), dim=-1)                # [P, D]
        gin = C['gin']; D = gin.size(-1) // 2
        q = F.normalize(gin[0, 0, D:].float(), dim=-1)          # truy van gate W_q[t_cls; e_type]
        vf = []
        for c in (c1, c2):
            ec = CEMB[t][c]                                     # [D], da chuan hoa
            sim = Vp @ ec                                       # [P]
            w = torch.softmax(sim / 0.07, dim=0)
            rc = F.normalize(w @ Vp, dim=-1)
            ent = float(-(w * (w + 1e-9).log()).sum())
            top5 = torch.topk(sim, min(5, sim.numel())).values
            vf.append([float(sim.max()), float(sim.mean()), float(top5.mean()), float(sim.std()),
                       float(rc @ ec), ent])
        vf = np.array(vf)                                       # [2, 6]
        # 🔬 READOUT CO HOC: luu ca VECTOR gop theo ung vien, khong chi cosine.
        #   Cosine la MOT huong co dinh; mot anh xa tuyen tinh hoc duoc tren 1024 chieu co the
        #   tim huong khac. Neu no thay tin hieu ma cosine khong thay -> ket luan
        #   "khong con thong tin thi giac" la ve CACH DOC, khong ve THONG TIN.
        _pool = []
        for c in (c1, c2):
            ec = CEMB[t][c]
            w = torch.softmax((Vp @ ec) / 0.07, dim=0)
            _pool.append((w @ Vp).float().cpu().numpy())
        POOL.append((_pool[0] - _pool[1]).astype(np.float16))   # huong phan biet top1 vs top2
        VIS.append(np.concatenate([vf[0], vf[1], vf[0] - vf[1]]))   # 18 dac trung

        sn = torch.softmax(s, dim=0)
        ROWS.append(dict(
            idx=j, type=t, y=y, nvoc=len(voc),
            s1=float(s[c1]), s2=float(s[c2]), margin=float(s[c1] - s[c2]),
            s1_ln=float(s[c1]) / max(CLEN[t][c1], 1), s2_ln=float(s[c2]) / max(CLEN[t][c2], 1),
            margin_ln=float(s[c1]) / max(CLEN[t][c1], 1) - float(s[c2]) / max(CLEN[t][c2], 1),
            ent=float(-(sn * (sn + 1e-9).log()).sum()),
            s_mean=float(s.mean()), s_std=float(s.std()), s1_z=float((s[c1] - s.mean()) / (s.std() + 1e-6)),
            len1=float(CLEN[t][c1]), len2=float(CLEN[t][c2]),
            logf1=float(np.log1p(FREQ.get(voc[c1], 0))), logf2=float(np.log1p(FREQ.get(voc[c2], 0))),
            qc1=float(q @ CEMB[t][c1]), qc2=float(q @ CEMB[t][c2]),
        ))
        if (n + 1) % 100 == 0:
            print(f'  {n + 1}/{len(idxs)}  giu {len(ROWS)}', flush=True)

    df = pd.DataFrame(ROWS)
    df['logf_d'] = df.logf1 - df.logf2
    df['qc_d'] = df.qc1 - df.qc2
    df['len_d'] = df.len1 - df.len2
    df.to_csv(a.out + '_meta.csv', index=False)
    np.save(a.out + '_vis.npy', np.stack(VIS))
    np.save(a.out + '_pool.npy', np.stack(POOL))
    print(f'\nluu {len(df)} mau  y=1 {int(df.y.sum())}  y=0 {int((1 - df.y).sum())}')


def stage_fit(a):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score
    df = pd.read_csv(a.out + '_meta.csv'); V = np.load(a.out + '_vis.npy')
    import os as _os
    P = np.load(a.out + '_pool.npy').astype(np.float32) if _os.path.exists(a.out + '_pool.npy') else None
    y = df.y.values
    DEC = ['s1', 's2', 'margin', 's1_ln', 's2_ln', 'margin_ln', 'ent', 's_mean', 's_std', 's1_z']
    TYP = ['nvoc', 'logf1', 'logf2', 'logf_d', 'qc1', 'qc2', 'qc_d', 'len1', 'len2', 'len_d']
    A = df[DEC].values
    T1H = np.eye(4)[df.type.values.astype(int)]
    B = np.hstack([A, T1H, df[TYP].values])
    rs = np.random.RandomState(0)
    perm_v = rs.permutation(len(df)); perm_t = rs.permutation(len(df))
    ARMS = {
        'A decoder': A,
        'B +type+prior (REF)': B,
        'C +vision': np.hstack([B, V]),
        'D vision only': V,
        'E null: vision xao': np.hstack([B, V[perm_v]]),
        'F null: type xao': np.hstack([A, T1H[perm_t], df[TYP].values, V]),
    }
    if P is not None:
        Pn = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-9)
        ARMS['G +readout CO HOC'] = np.hstack([B, Pn])
        ARMS['H null: readout xao'] = np.hstack([B, Pn[perm_v]])

    def cv_auc(X, seed):
        sk = StratifiedKFold(5, shuffle=True, random_state=seed)
        oof = np.zeros(len(y))
        for tr, va in sk.split(X, y):
            p = make_pipeline(StandardScaler(),
                              LogisticRegression(max_iter=4000, C=a.C))
            p.fit(X[tr], y[tr]); oof[va] = p.predict_proba(X[va])[:, 1]
        return oof

    print(f'n={len(y)}  y=1 {y.sum()}  ty le co so {y.mean():.3f}\n')
    OOF = {}
    for k, X in ARMS.items():
        o = np.mean([cv_auc(X, s) for s in range(a.cv_seeds)], axis=0)
        OOF[k] = o
        print(f'  {k:24s} AUROC {roc_auc_score(y, o):.4f}')
    ref = OOF['B +type+prior (REF)']
    print('\nso gia tang so voi B, bootstrap 95% CI (2000 lan, theo mau):')
    for k in [x for x in ARMS if x != 'B +type+prior (REF)']:
        d = []
        rs2 = np.random.RandomState(1)
        for _ in range(2000):
            b = rs2.randint(0, len(y), len(y))
            if y[b].min() == y[b].max():
                continue
            d.append(roc_auc_score(y[b], OOF[k][b]) - roc_auc_score(y[b], ref[b]))
        d = np.array(d)
        lo, hi = np.percentile(d, [2.5, 97.5])
        flag = 'VUOT 0' if lo > 0 else ('duoi 0' if hi < 0 else 'chua 0')
        print(f'  {k:24s} {d.mean():+.4f}  [{lo:+.4f}, {hi:+.4f}]  {flag}')

    print('\n--- phan tang margin thap 30% (tap cuu duoc) ---')
    thr = np.percentile(df.margin.values, 30)
    lm = df.margin.values <= thr
    for k in ('B +type+prior (REF)', 'C +vision'):
        print(f'  {k:24s} AUROC {roc_auc_score(y[lm], OOF[k][lm]):.4f}  (n={lm.sum()}, y=1 {y[lm].sum()})')

    print('\n--- theo loai cau hoi ---')
    for t in sorted(df.type.unique()):
        s = df.type.values == t
        if len(set(y[s])) < 2:
            continue
        print(f'  type {t}: n={s.sum():4d} y=1 {y[s].sum():4d}  '
              f'B {roc_auc_score(y[s], ref[s]):.4f}  C {roc_auc_score(y[s], OOF["C +vision"][s]):.4f}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--stage', choices=['dump', 'fit'], required=True)
    p.add_argument('--checkpoint'); p.add_argument('--out', required=True)
    p.add_argument('--train_csv', default='archive/train_split_original.csv')
    p.add_argument('--test_csv', default='archive/test.csv')
    p.add_argument('--image_folder', default='archive/data/images/test')
    p.add_argument('--per_type', type=int, default=0)
    p.add_argument('--topk', type=int, default=2,
        help='giu mau co gold trong top-K. Mac dinh 2 (ban da chay). topk=5 phu them 62%% '
             'so loi "nhan dang" ma ban top-2 DA LOAI TRU.')
    p.add_argument('--vision_source', choices=['post','raw'], default='post',
        help="post = dam may SAU GCA/TCVG/LN (mac dinh, da chay). "
             "raw = patch SigLIP THO sau vision_proj, TRUOC Flamingo — kiem tra xem GCA co "
             "BOI NHOE thong tin thi giac khong.")
    p.add_argument('--chunk', type=int, default=48)
    p.add_argument('--cv_seeds', type=int, default=5)
    p.add_argument('--C', type=float, default=1.0)
    a = p.parse_args()
    (stage_dump if a.stage == 'dump' else stage_fit)(a)
