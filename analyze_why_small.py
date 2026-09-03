"""TAI SAO Delta(T2-T0) NHO VA KHONG ON DINH — ba phep phan tich chua tung lam.

1. Delta co phai HOI QUY VE TRUNG BINH khong?
   Neu corr(T0_seed, Delta_seed) am manh -> TCVG khong cong them mot luong co dinh, ma
   NANG SAN cho cac seed te. Do la mot dong gop KHAC, va do duoc.

2. TCVG co GIAM PHUONG SAI giua cac seed khong? (kiem dinh F)
   Neu sd(T2) < sd(T0) co y nghia -> "TCVG lam mo hinh on dinh hon voi khoi tao" la mot
   claim that, khac voi claim accuracy.

3. Loi ich co TAP TRUNG vao mot tap mau co dinh khong, hay moi seed thang o mot cho khac?
   Voi moi mau i: w_i = (#seed T2 dung & T0 sai) - (#seed T0 dung & T2 sai).
   Neu TCVG co CO CHE THAT -> phan phoi w_i co duoi phai nang (mot so mau LUON duoc sua).
   Neu chi la nhieu -> w_i doi xung nhu nhi thuc ngau nhien.
   Day la phep phan biet dut khoat giua "co co che yeu" va "khong co co che".
"""
import csv
import os
import math
from collections import Counter

ROOT = os.path.dirname(os.path.abspath(__file__))
SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8]


def mean(x):
    return sum(x) / len(x)


def sd(x):
    m = mean(x)
    return (sum((v - m) ** 2 for v in x) / (len(x) - 1)) ** 0.5


def pearson(x, y):
    mx, my = mean(x), mean(y)
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    den = (sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in y)) ** 0.5
    return num / den if den else float('nan')


# ---------------------------------------------------------------- 1 & 2
rows = list(csv.DictReader(open(f"{ROOT}/analysis/final/vivqa_paired_T0_T2.csv")))
rows = [r for r in rows if int(r['seed']) in SEEDS]
t0 = [float(r['T0_Overall']) for r in rows]
t2 = [float(r['T2_Overall']) for r in rows]
dl = [float(r['d_Overall']) for r in rows]

print("=" * 84)
print("1. DELTA CO PHAI HOI QUY VE TRUNG BINH?")
print("=" * 84)
print(f"{'seed':>5} {'T0':>8} {'T2':>8} {'Delta':>8}")
for r, a, b, d in sorted(zip(rows, t0, t2, dl), key=lambda z: z[1]):
    print(f"{int(r['seed']):>5} {a:>8.2f} {b:>8.2f} {d:>+8.2f}")
r_t0 = pearson(t0, dl)
r_t2 = pearson(t2, dl)
n = len(t0)
tstat = r_t0 * math.sqrt((n - 2) / max(1 - r_t0 ** 2, 1e-12))
print(f"\ncorr(T0, Delta) = {r_t0:+.3f}   t={tstat:+.2f}  (n={n}, df={n-2})")
print(f"corr(T2, Delta) = {r_t2:+.3f}")
print("Doc: corr(T0,Delta) am manh nghia la seed nao T0 TE thi TCVG giup NHIEU")
print("     -> TCVG nang san, khong cong deu. corr(T2,Delta) ~ 0 cung cung chieu do.")

print()
print("=" * 84)
print("2. TCVG CO GIAM PHUONG SAI GIUA CAC SEED KHONG? (kiem dinh F)")
print("=" * 84)
s0, s2 = sd(t0), sd(t2)
print(f"sd(T0) = {s0:.3f}   sd(T2) = {s2:.3f}   ty le phuong sai F = {(s0**2)/(s2**2):.2f}")
print(f"khoang bien thien: T0 [{min(t0):.2f}, {max(t0):.2f}] = {max(t0)-min(t0):.2f}"
      f"   T2 [{min(t2):.2f}, {max(t2):.2f}] = {max(t2)-min(t2):.2f}")
# p-value cho F voi df1=df2=n-1, xap xi bang mo phong hoan vi (khong can scipy)
import random
random.seed(0)
obs = (s0 ** 2) / (s2 ** 2)
pool = list(zip(t0, t2))
cnt = 0
N = 20000
for _ in range(N):
    a, b = [], []
    for x, y in pool:
        if random.random() < 0.5:
            a.append(x); b.append(y)
        else:
            a.append(y); b.append(x)
    if (sd(a) ** 2) / (sd(b) ** 2) >= obs:
        cnt += 1
print(f"p (hoan vi trong tung cap, mot phia) = {cnt/N:.4f}   [{N} lan]")

# ---------------------------------------------------------------- 3
print()
print("=" * 84)
print("3. LOI ICH CO TAP TRUNG VAO TAP MAU CO DINH KHONG?")
print("=" * 84)


def load(path):
    if not os.path.exists(path):
        return None
    return [1 if float(r['exact_match']) >= 0.5 else 0
            for r in csv.DictReader(open(path))]


pairs = []
for s in SEEDS:
    a = load(f"{ROOT}/analysis/T0/T0_seed{s}.csv")
    b = load(f"{ROOT}/beam3fixed/seed{s}_ep40.csv")
    if a and b and len(a) == len(b):
        pairs.append((s, a, b))
print(f"so seed doc duoc: {len(pairs)}")
if pairs:
    N_s = len(pairs[0][1])
    w = [0] * N_s
    for _, a, b in pairs:
        for i, (x, y) in enumerate(zip(a, b)):
            if y == 1 and x == 0:
                w[i] += 1
            elif x == 1 and y == 0:
                w[i] -= 1
    k = len(pairs)
    dist = Counter(w)
    print(f"\nw_i = (#seed T2 sua duoc) - (#seed T2 lam hong), moi mau, {k} seed")
    print(f"{'w':>5} {'so mau':>9}   {'':<4}")
    for v in sorted(dist):
        bar = '#' * min(60, dist[v] * 60 // max(dist.values()))
        print(f"{v:>5} {dist[v]:>9}   {bar}")
    pos = sum(c for v, c in dist.items() if v > 0)
    neg = sum(c for v, c in dist.items() if v < 0)
    zer = dist.get(0, 0)
    print(f"\nw>0: {pos}   w=0: {zer}   w<0: {neg}   tong {N_s}")
    strong_p = sum(c for v, c in dist.items() if v >= k - 2)
    strong_n = sum(c for v, c in dist.items() if v <= -(k - 2))
    print(f"mau T2 sua duoc o >= {k-2}/{k} seed : {strong_p}")
    print(f"mau T2 lam hong o >= {k-2}/{k} seed : {strong_n}")
    print(f"chenh lech ROnG cua duoi manh        : {strong_p - strong_n}")

    # NULL: neu chi la nhieu doc lap, w_i ~ hieu hai nhi thuc doi xung.
    # Uoc luong xac suat lat tu chinh du lieu roi mo phong.
    flips = sum(1 for _, a, b in pairs for x, y in zip(a, b) if x != y)
    p_flip = flips / (len(pairs) * N_s)
    random.seed(1)
    sim_strong_p = sim_strong_n = 0
    for _ in range(N_s):
        ww = 0
        for _k in range(k):
            if random.random() < p_flip:
                ww += 1 if random.random() < 0.5 else -1
        if ww >= k - 2:
            sim_strong_p += 1
        if ww <= -(k - 2):
            sim_strong_n += 1
    print(f"\nNULL (nhieu doc lap, ty le lat = {p_flip:.4f}) tren cung {N_s} mau:")
    print(f"   ky vong duoi phai manh: {sim_strong_p}   duoi trai manh: {sim_strong_n}")
    print("Doc: neu quan sat >> null -> CO tap mau on dinh duoc TCVG sua -> co che that.")
    print("     neu quan sat ~ null -> loi ich khong tap trung, moi seed thang o cho khac.")
