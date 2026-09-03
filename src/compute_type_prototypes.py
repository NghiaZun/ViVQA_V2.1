"""
Tinh prototype ngu nghia cho type_embedding: mean-pool (theo attention_mask) embedding cua
BARTpho DONG BANG (chua fine-tune) tren cac cau hoi that thuoc tung type, thay the khoi tao
ngau nhien N(0,1). (Da thu BOS-only truoc: cosine sim ~0.999 giua cac type -- gan nhu vo nghia
vi BOS chua fine-tune chi mang tin hieu chung chung. Mean-pool tren token that cho tach biet
that: cosine sim 0.70-0.91, vi tu khoa dac trung type (mau, bao nhieu, o dau) nam TRONG cau,
khong o vi tri BOS.)

Muc dich: sua nguyen nhan goc cua symmetry-breaking (da chan doan nhieu lan trong investigation
2026-08-08/09) -- type_embedding random N(0,1) (norm~32) ap dao gradient budget nho cua AdamW,
khien "type nao bi gate manh" quyet dinh boi HUONG NGAU NHIEN LUC INIT (doi theo seed), khong
phai du lieu. Thay bang huong THAT (tu chinh cau hoi cua tung type) -> gate bat dau tu diem co
y nghia, gradient chi can TINH CHINH thay vi phai tu tim huong tu con so 0.

Tong quat: khong hardcode gia tri, tu tinh tu du lieu -> ap dung duoc cho dataset/taxonomy khac
(chi can co cot 'type' voi cac ma so nguyen).

Usage: python compute_type_prototypes.py --train_csv <path> --out <path.pt>
"""
import argparse
import torch
import pandas as pd
from transformers import BartphoTokenizer, MBartForConditionalGeneration


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train_csv', default='archive/train_split_original.csv')
    ap.add_argument('--out', default='type_emb_prototypes.pt')
    ap.add_argument('--bartpho_model', default='vinai/bartpho-syllable')
    ap.add_argument('--num_types', type=int, default=4)
    ap.add_argument('--max_q_len', type=int, default=32)
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Loading frozen BARTpho ({args.bartpho_model})...")
    tokenizer = BartphoTokenizer.from_pretrained(args.bartpho_model)
    model = MBartForConditionalGeneration.from_pretrained(args.bartpho_model).to(device)
    encoder = model.get_encoder()
    encoder.eval()

    df = pd.read_csv(args.train_csv)
    assert 'type' in df.columns and 'question' in df.columns, "can cot 'type' va 'question'"

    hidden_dim = model.config.d_model
    print(f"BARTpho d_model={hidden_dim}")

    protos = torch.zeros(args.num_types, hidden_dim)
    counts = [0] * args.num_types

    with torch.no_grad():
        for t in range(args.num_types):
            qs = df[df['type'] == t]['question'].tolist()
            if not qs:
                print(f"  type={t}: KHONG CO cau hoi nao, giu prototype = 0")
                continue
            sum_emb = torch.zeros(hidden_dim, device=device)
            bs = 64
            for i in range(0, len(qs), bs):
                batch = qs[i:i+bs]
                enc = tokenizer(batch, truncation=True, padding=True, max_length=args.max_q_len,
                                 return_tensors='pt').to(device)
                out = encoder(input_ids=enc['input_ids'], attention_mask=enc['attention_mask'])
                mask = enc['attention_mask'].unsqueeze(-1).float()  # [B,L,1]
                pooled = (out.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)  # [B,D]
                sum_emb += pooled.sum(dim=0)
            mean_emb = (sum_emb / len(qs)).cpu()
            protos[t] = mean_emb
            counts[t] = len(qs)
            print(f"  type={t}: n={len(qs)}  norm={mean_emb.norm().item():.4f}")

    # pairwise cosine similarity giua cac prototype -- kiem tra chung co thuc su khac nhau khong
    print("\nCosine similarity giua cac type prototype:")
    normed = torch.nn.functional.normalize(protos, dim=-1)
    sim = normed @ normed.T
    print(sim)

    torch.save({'prototypes': protos, 'counts': counts, 'num_types': args.num_types}, args.out)
    print(f"\nSaved to {args.out}")


if __name__ == '__main__':
    main()
