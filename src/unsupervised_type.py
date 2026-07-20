"""
unsupervised_type.py — Khám phá loại câu hỏi KHÔNG giám sát (WORK IN PROGRESS).

Mục tiêu: thay thế bộ luật cứng `detect_question_type()` trong dataset.py bằng
cách *tự phát hiện* các loại câu hỏi từ dữ liệu, thông qua gom cụm (clustering)
embedding câu hỏi. Ý tưởng để ViGCT-VQA không phụ thuộc vào 4 loại thủ công
(OBJECT / COUNT / COLOR / LOCATION) mà có thể mở rộng sang K loại tuỳ dữ liệu.

Điểm tích hợp với pipeline hiện tại
-----------------------------------
`VQAGenDataset` đã hỗ trợ đọc nhãn loại từ cột `question_type` của CSV
(xem dataset.py). Vì vậy module này KHÔNG sửa gì trong train loop: nó chỉ
gom cụm rồi ghi thêm cột `question_type` vào CSV, còn train/eval dùng như cũ.

Quy trình:
    1. Nhúng câu hỏi bằng encoder BARTpho (mean-pool, không tính gradient).
    2. L2-normalize → KMeans++ (cosine) thành K cụm.
    3. (tuỳ chọn) Ánh xạ mỗi cụm sang tên loại dễ hiểu bằng majority-vote so
       với `detect_question_type` — để so sánh/diễn giải, không bắt buộc.
    4. Lưu tâm cụm (centroids) để gán NHẤT QUÁN cho val/test.

Cách dùng (CLI):
    # Fit trên train, sinh cột question_type + lưu centroids
    python unsupervised_type.py fit \
        --csv train_split.csv --n_types 4 \
        --out_csv train_split_uns.csv --centroids uns_centroids.npz --align_rules

    # Gán cho val/test bằng centroids đã học (đảm bảo cùng không gian nhãn)
    python unsupervised_type.py assign \
        --csv val_split.csv --centroids uns_centroids.npz --out_csv val_split_uns.csv

TRẠNG THÁI: đang phát triển — chưa được nối mặc định vào train.py.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import List, Optional, Sequence

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# 1. Nhúng câu hỏi bằng BARTpho encoder (mean-pool)
# ─────────────────────────────────────────────────────────────────────────────
class QuestionEmbedder:
    """Mã hoá câu hỏi tiếng Việt thành vector bằng encoder BARTpho (frozen)."""

    def __init__(self,
                 encoder_name: str = "vinai/bartpho-syllable",
                 device: Optional[str] = None,
                 max_len: int = 32,
                 batch_size: int = 64):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_len = max_len
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(encoder_name)
        # Chỉ cần phần encoder để lấy biểu diễn câu hỏi
        self.model = AutoModel.from_pretrained(encoder_name).to(self.device).eval()

    @staticmethod
    def _mean_pool(last_hidden, attention_mask):
        # last_hidden: [B, L, D], attention_mask: [B, L]
        mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)
        summed = (last_hidden * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-6)
        return summed / counts

    def encode(self, questions: Sequence[str]) -> np.ndarray:
        torch = self.torch
        vecs: List[np.ndarray] = []
        with torch.no_grad():
            for i in range(0, len(questions), self.batch_size):
                batch = [str(q) for q in questions[i:i + self.batch_size]]
                enc = self.tokenizer(
                    batch, padding=True, truncation=True,
                    max_length=self.max_len, return_tensors="pt",
                ).to(self.device)
                # Với BART, dùng riêng encoder để tránh tính cả decoder
                get_enc = getattr(self.model, "get_encoder", None)
                if callable(get_enc):
                    out = get_enc()(input_ids=enc.input_ids,
                                    attention_mask=enc.attention_mask)
                else:
                    out = self.model(**enc)
                pooled = self._mean_pool(out.last_hidden_state, enc.attention_mask)
                vecs.append(pooled.float().cpu().numpy())
        return np.concatenate(vecs, axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# 2. KMeans++ (numpy, cosine) — tự cài để không thêm dependency
# ─────────────────────────────────────────────────────────────────────────────
def _l2_normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + eps)


def _kmeanspp_init(x: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    n = x.shape[0]
    centers = [x[rng.integers(n)]]
    for _ in range(1, k):
        d2 = np.min(
            [np.sum((x - c) ** 2, axis=1) for c in centers], axis=0
        )
        probs = d2 / (d2.sum() + 1e-12)
        centers.append(x[rng.choice(n, p=probs)])
    return np.stack(centers, axis=0)


def kmeans(x: np.ndarray, k: int, n_iter: int = 100,
           seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """KMeans trên vector đã L2-normalize (khoảng cách ~ cosine).

    Trả về (labels [N], centroids [k, D])."""
    rng = np.random.default_rng(seed)
    x = _l2_normalize(x)
    centroids = _l2_normalize(_kmeanspp_init(x, k, rng))
    labels = np.zeros(x.shape[0], dtype=np.int64)
    for _ in range(n_iter):
        sims = x @ centroids.T                      # [N, k] cosine similarity
        new_labels = np.argmax(sims, axis=1)
        if np.array_equal(new_labels, labels):
            labels = new_labels
            break
        labels = new_labels
        for j in range(k):
            members = x[labels == j]
            if len(members) > 0:
                centroids[j] = members.mean(axis=0)
            else:  # cụm rỗng → khởi tạo lại từ điểm xa nhất
                far = np.argmin(np.max(x @ centroids.T, axis=1))
                centroids[j] = x[far]
        centroids = _l2_normalize(centroids)
    return labels, centroids


def silhouette_lite(x: np.ndarray, labels: np.ndarray,
                    sample: int = 2000, seed: int = 42) -> float:
    """Silhouette score gọn nhẹ trên tập con (để chọn K tự động)."""
    rng = np.random.default_rng(seed)
    x = _l2_normalize(x)
    idx = rng.choice(len(x), size=min(sample, len(x)), replace=False)
    xs, ls = x[idx], labels[idx]
    scores = []
    for i in range(len(xs)):
        same = xs[ls == ls[i]]
        a = np.mean(np.linalg.norm(same - xs[i], axis=1)) if len(same) > 1 else 0.0
        b = np.inf
        for c in np.unique(ls):
            if c == ls[i]:
                continue
            other = xs[ls == c]
            b = min(b, np.mean(np.linalg.norm(other - xs[i], axis=1)))
        denom = max(a, b) if max(a, b) > 0 else 1.0
        scores.append((b - a) / denom)
    return float(np.mean(scores))


# ─────────────────────────────────────────────────────────────────────────────
# 3. Bộ khám phá loại không giám sát
# ─────────────────────────────────────────────────────────────────────────────
class UnsupervisedTypeDiscovery:
    """Gom cụm câu hỏi → nhãn loại; drop-in thay cho detect_question_type()."""

    def __init__(self, n_types: int = 4,
                 encoder_name: str = "vinai/bartpho-syllable",
                 device: Optional[str] = None, seed: int = 42):
        self.n_types = n_types
        self.encoder_name = encoder_name
        self.device = device
        self.seed = seed
        self.centroids: Optional[np.ndarray] = None    # [k, D], đã L2-normalize
        self.cluster_names: Optional[dict] = None      # {cluster_id: "TÊN"} (tuỳ chọn)
        self._embedder: Optional[QuestionEmbedder] = None

    # -- lazy load encoder --
    @property
    def embedder(self) -> QuestionEmbedder:
        if self._embedder is None:
            self._embedder = QuestionEmbedder(self.encoder_name, self.device)
        return self._embedder

    def fit(self, questions: Sequence[str],
            auto_k: bool = False, k_range: Sequence[int] = (3, 4, 5, 6)) -> np.ndarray:
        """Học tâm cụm từ danh sách câu hỏi. Trả về nhãn cụm [N]."""
        emb = self.embedder.encode(questions)
        if auto_k:
            best_k, best_s = self.n_types, -1.0
            for k in k_range:
                lbl, _ = kmeans(emb, k, seed=self.seed)
                s = silhouette_lite(emb, lbl, seed=self.seed)
                print(f"  [auto-k] K={k}  silhouette={s:.4f}")
                if s > best_s:
                    best_k, best_s = k, s
            self.n_types = best_k
            print(f"  [auto-k] chọn K={best_k} (silhouette={best_s:.4f})")
        labels, self.centroids = kmeans(emb, self.n_types, seed=self.seed)
        return labels

    def predict(self, questions: Sequence[str]) -> np.ndarray:
        """Gán nhãn cụm cho câu hỏi mới bằng centroids đã học."""
        if self.centroids is None:
            raise RuntimeError("Chưa fit hoặc load centroids.")
        emb = _l2_normalize(self.embedder.encode(questions))
        return np.argmax(emb @ self.centroids.T, axis=1)

    def align_to_rules(self, questions: Sequence[str], labels: np.ndarray) -> dict:
        """Ánh xạ mỗi cụm → tên loại thủ công (majority-vote) để diễn giải.

        Không thay đổi nhãn số; chỉ tạo bảng tên giúp so sánh với rule-based.
        """
        from collections import Counter
        try:
            from dataset import detect_question_type
        except Exception:
            print("  ⚠️  Không import được detect_question_type — bỏ qua align.")
            return {}
        names = {0: "OBJECT", 1: "COUNT", 2: "COLOR", 3: "LOCATION"}
        mapping = {}
        for c in range(self.n_types):
            rule_types = [detect_question_type(str(q))
                          for q, l in zip(questions, labels) if l == c]
            if rule_types:
                top = Counter(rule_types).most_common(1)[0][0]
                mapping[c] = names.get(top, f"TYPE_{top}")
            else:
                mapping[c] = f"CLUSTER_{c}"
        self.cluster_names = mapping
        return mapping

    # -- persistence --
    def save(self, path: str) -> None:
        assert self.centroids is not None, "Chưa có centroids để lưu."
        np.savez(path, centroids=self.centroids,
                 n_types=self.n_types, encoder_name=self.encoder_name)
        meta = {"n_types": int(self.n_types), "encoder_name": self.encoder_name,
                "cluster_names": self.cluster_names, "seed": self.seed}
        with open(os.path.splitext(path)[0] + ".json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        print(f"  ✅ Đã lưu centroids → {path}")

    @classmethod
    def load(cls, path: str) -> "UnsupervisedTypeDiscovery":
        data = np.load(path, allow_pickle=True)
        obj = cls(n_types=int(data["n_types"]),
                  encoder_name=str(data["encoder_name"]))
        obj.centroids = data["centroids"]
        meta_path = os.path.splitext(path)[0] + ".json"
        if os.path.exists(meta_path):
            with open(meta_path, encoding="utf-8") as f:
                obj.cluster_names = json.load(f).get("cluster_names")
        return obj


# ─────────────────────────────────────────────────────────────────────────────
# 4. CLI: fit trên train / assign cho val-test
# ─────────────────────────────────────────────────────────────────────────────
def _write_csv_with_types(csv_path: str, labels: np.ndarray, out_csv: str,
                          col: str = "question_type") -> None:
    import pandas as pd
    df = pd.read_csv(csv_path)
    df[col] = labels.astype(int)
    df.to_csv(out_csv, index=False)
    print(f"  ✅ Ghi {out_csv}  (thêm cột '{col}', {len(df)} dòng)")


def main():
    ap = argparse.ArgumentParser(description="Unsupervised question-type discovery (WIP)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    pf = sub.add_parser("fit", help="Học cụm trên CSV train + ghi cột question_type")
    pf.add_argument("--csv", required=True)
    pf.add_argument("--text_col", default="question")
    pf.add_argument("--n_types", type=int, default=4)
    pf.add_argument("--encoder", default="vinai/bartpho-syllable")
    pf.add_argument("--out_csv", required=True)
    pf.add_argument("--centroids", default="uns_centroids.npz")
    pf.add_argument("--auto_k", action="store_true", help="Tự chọn K bằng silhouette")
    pf.add_argument("--align_rules", action="store_true",
                    help="Ánh xạ cụm → tên loại thủ công để diễn giải")
    pf.add_argument("--seed", type=int, default=42)

    pa = sub.add_parser("assign", help="Gán nhãn cho CSV mới bằng centroids đã học")
    pa.add_argument("--csv", required=True)
    pa.add_argument("--text_col", default="question")
    pa.add_argument("--centroids", required=True)
    pa.add_argument("--out_csv", required=True)

    args = ap.parse_args()
    import pandas as pd

    if args.cmd == "fit":
        questions = pd.read_csv(args.csv)[args.text_col].astype(str).tolist()
        disc = UnsupervisedTypeDiscovery(n_types=args.n_types,
                                         encoder_name=args.encoder, seed=args.seed)
        print(f"→ Fit trên {len(questions)} câu hỏi, K={args.n_types}")
        labels = disc.fit(questions, auto_k=args.auto_k)
        if args.align_rules:
            print("→ Ánh xạ cụm ↔ luật thủ công:")
            for c, name in disc.align_to_rules(questions, labels).items():
                n = int((labels == c).sum())
                print(f"    cụm {c}: {name:<10} ({n:,} câu)")
        disc.save(args.centroids)
        _write_csv_with_types(args.csv, labels, args.out_csv)

    elif args.cmd == "assign":
        questions = pd.read_csv(args.csv)[args.text_col].astype(str).tolist()
        disc = UnsupervisedTypeDiscovery.load(args.centroids)
        print(f"→ Gán {len(questions)} câu hỏi bằng centroids ({disc.n_types} cụm)")
        labels = disc.predict(questions)
        _write_csv_with_types(args.csv, labels, args.out_csv)


if __name__ == "__main__":
    main()
