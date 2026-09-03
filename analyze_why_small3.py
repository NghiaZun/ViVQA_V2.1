"""Kiem lai ket qua B bang NULL DUNG, roi dac ta tap mau bi anh huong on dinh.

LOI CUA NULL TRUOC: gia dinh MOI mau co cung ty le lat 0.0786. Thuc te do kho cua mau
rat khac nhau — mau "sat nguong" lat nhieu o CA HAI mo hinh, lam phan phoi w rong ra
NGAY CA KHI TCVG khong co tac dung gi. Nen z=+46 co the la ao.

NULL DUNG (kiem hoan doi / exchangeability): voi TUNG mau i va TUNG seed s, hoan doi ngau
nhien nhan T0<->T2. Cach nay GIU NGUYEN mau hinh dung/sai theo tung mau va tung seed
(tuc giu nguyen do kho va do "sat nguong"), chi pha bo viec nhan nao la T0 nhan nao la T2.
Duoi H0 "T0 va T2 hoan doi duoc", phan phoi w khong doi.
"""
import csv
import os
import random
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.abspath(__file__))
SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8]


def rows_of(path):
    return list(csv.DictReader(open(path))) if os.path.exists(path) else None


def em_of(rs):
    return [1 if float(r['exact_match']) >= 0.5 else 0 for r in rs]


def mean(x):
    return sum(x) / len(x)


def sd(x):
    m = mean(x)
    return (sum((v - m) ** 2 for v in x) / (len(x) - 1)) ** 0.5


pairs, meta = [], None
for s in SEEDS:
    ra = rows_of(f"{ROOT}/analysis/T0/T0_seed{s}.csv")
    rb = rows_of(f"{ROOT}/beam3fixed/seed{s}_ep40.csv")
    if ra and rb and len(ra) == len(rb):
        pairs.append((s, em_of(ra), em_of(rb)))
        if meta is None:
            meta = ra
N, k = len(pairs[0][1]), len(pairs)
print(f"{k} seed, {N} mau")

w = [0] * N
for _, a, b in pairs:
    for i, (x, y) in enumerate(zip(a, b)):
        if y == 1 and x == 0:
            w[i] += 1
        elif x == 1 and y == 0:
            w[i] -= 1

print()
print("=" * 88)
print("NULL DUNG: hoan doi nhan T0<->T2 trong tung (mau, seed) — giu nguyen do kho tung mau")
print("=" * 88)
rng = random.Random(0)
REP = 400
sim = {th: [] for th in range(2, k + 1)}
sim_var = []
disc = [[(a[i], b[i]) for _, a, b in pairs] for i in range(N)]
for _ in range(REP):
    ws = []
    for i in range(N):
        v = 0
        for x, y in disc[i]:
            if x == y:
                continue
            if rng.random() < 0.5:
                x, y = y, x
            v += 1 if (y == 1 and x == 0) else -1
        ws.append(v)
    sim_var.append(sd(ws) ** 2)
    for th in sim:
        sim[th].append(sum(1 for v in ws if abs(v) >= th))

print(f"phuong sai w: quan sat = {sd(w)**2:.4f}   null = {mean(sim_var):.4f} (sd {sd(sim_var):.4f})"
      f"   z = {(sd(w)**2-mean(sim_var))/max(sd(sim_var),1e-9):+.2f}")
print(f"\n{'|w|>=':>6} {'quan sat':>10} {'null TB':>10} {'sd':>7} {'z':>8}")
for th in range(2, k + 1):
    obs = sum(1 for v in w if abs(v) >= th)
    m, s = mean(sim[th]), sd(sim[th])
    print(f"{th:>6} {obs:>10} {m:>10.1f} {s:>7.2f} {(obs-m)/max(s,1e-9):>+8.2f}")

print()
print("=" * 88)
print("DAC TA: mau T2 SUA duoc on dinh (w>=3) vs mau T2 LAM HONG on dinh (w<=-3)")
print("=" * 88)
fixed = [i for i in range(N) if w[i] >= 3]
broke = [i for i in range(N) if w[i] <= -3]
print(f"T2 sua on dinh (w>=3) : {len(fixed)} mau")
print(f"T2 hong on dinh (w<=-3): {len(broke)} mau")
print(f"chenh RONG            : {len(fixed)-len(broke)} mau  = {(len(fixed)-len(broke))/N*100:+.2f} diem EM")

# dac trung: loai cau hoi, do dai cau, tan suat dap an
import collections
ans_freq = collections.Counter()
try:
    import pandas as pd
    tr = pd.read_csv(f"{ROOT}/archive/train_split.csv")
    ans_freq = collections.Counter(str(a).strip().lower() for a in tr['answer'])
except Exception as e:
    print("(khong doc duoc train_split de tinh tan suat dap an:", e, ")")


def describe(name, idxs):
    if not idxs:
        print(f"{name}: rong")
        return
    types = Counter(meta[i].get('question_type', '?') for i in idxs)
    qlen = mean([len(str(meta[i].get('question', '')).split()) for i in idxs])
    if ans_freq:
        fr = [ans_freq.get(str(meta[i].get('ground_truth', '')).strip().lower(), 0) for i in idxs]
        frs = f"  tan suat dap an trong train: trung vi={sorted(fr)[len(fr)//2]}"
    else:
        frs = ""
    tot = sum(types.values())
    dist = "  ".join(f"{t}={types[t]/tot*100:.0f}%" for t in sorted(types))
    print(f"{name} (n={len(idxs)}): {dist}   do dai cau TB={qlen:.1f}{frs}")


base_types = Counter(r.get('question_type', '?') for r in meta)
tot = sum(base_types.values())
print("\nphan bo loai TOAN TAP TEST: " + "  ".join(
    f"{t}={base_types[t]/tot*100:.0f}%" for t in sorted(base_types)))
describe("T2 SUA on dinh ", fixed)
describe("T2 HONG on dinh", broke)
