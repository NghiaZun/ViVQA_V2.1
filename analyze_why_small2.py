"""Kiem tra lai hai ket qua o buoc truoc, bang phep kiem KHONG bi nhiem.

A. corr(T0, Delta) = -0.771 co that khong, hay la AO GIAC HOI QUY?
   Delta = T2 - T0 nen T0 nam o CA HAI truc: nhieu do luong trong T0 tu dong tao tuong quan am.
   PHEP KIEM SACH: chia doi tap test. Do T0 tren NUA A, do Delta tren NUA B.
   Hai nua khong chia se nhieu lay mau -> neu van am manh thi hieu ung la THAT.

B. Phan phoi w_i co RONG hon null khong?
   Buoc truoc dat nguong |w| >= 7/9 — nguong do khong co suc manh (null cung ra 0).
   Lam lai: so sanh TOAN BO duoi o moi nguong 2..9, va so sanh phuong sai cua w.
"""
import csv
import os
import math
import random

ROOT = os.path.dirname(os.path.abspath(__file__))
SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8]


def load(path):
    if not os.path.exists(path):
        return None
    return [1 if float(r['exact_match']) >= 0.5 else 0
            for r in csv.DictReader(open(path))]


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


pairs = []
for s in SEEDS:
    a = load(f"{ROOT}/analysis/T0/T0_seed{s}.csv")
    b = load(f"{ROOT}/beam3fixed/seed{s}_ep40.csv")
    if a and b and len(a) == len(b):
        pairs.append((s, a, b))
N = len(pairs[0][1])
print(f"{len(pairs)} seed, {N} mau")

# ------------------------------------------------------------------ A
print()
print("=" * 84)
print("A. corr(T0, Delta) — kiem bang CHIA DOI TAP TEST (khong chia se nhieu lay mau)")
print("=" * 84)
rng = random.Random(0)
rs_naive, rs_split = [], []
for rep in range(200):
    idx = list(range(N))
    rng.shuffle(idx)
    A, B = idx[:N // 2], idx[N // 2:]
    t0A = [mean([a[i] for i in A]) * 100 for _, a, _ in pairs]
    t0B = [mean([a[i] for i in B]) * 100 for _, a, _ in pairs]
    t2B = [mean([b[i] for i in B]) * 100 for _, _, b in pairs]
    dB = [x - y for x, y in zip(t2B, t0B)]
    rs_split.append(pearson(t0A, dB))    # SACH: T0 do tren nua A, Delta tren nua B
    rs_naive.append(pearson(t0B, dB))    # NHIEM: cung nua B
print(f"corr NHIEM   (T0 va Delta cung nua): {mean(rs_naive):+.3f}  (sd {sd(rs_naive):.3f})")
print(f"corr SACH    (T0 nua A, Delta nua B): {mean(rs_split):+.3f}  (sd {sd(rs_split):.3f})")
n = len(pairs)
t = mean(rs_split) * math.sqrt((n - 2) / max(1 - mean(rs_split) ** 2, 1e-12))
print(f"   t = {t:+.2f}, df = {n-2}")
print()
print("Doc: neu corr SACH ~ 0 thi -0.771 la AO GIAC HOI QUY, khong duoc claim.")
print("     neu corr SACH van am ro thi TCVG THUC SU giup nhieu hon o seed yeu.")

# ------------------------------------------------------------------ B
print()
print("=" * 84)
print("B. Phan phoi w_i co RONG hon null khong (moi nguong, khong chi 7/9)")
print("=" * 84)
w = [0] * N
for _, a, b in pairs:
    for i, (x, y) in enumerate(zip(a, b)):
        if y == 1 and x == 0:
            w[i] += 1
        elif x == 1 and y == 0:
            w[i] -= 1
k = len(pairs)
flips = sum(1 for _, a, b in pairs for x, y in zip(a, b) if x != y)
p_flip = flips / (k * N)
# huong lat that (T2 thang nhieu hon mot chut)
wins = sum(1 for _, a, b in pairs for x, y in zip(a, b) if x == 0 and y == 1)
p_win = wins / flips
print(f"ty le lat = {p_flip:.4f}   trong do T2 thang = {p_win:.4f}")

rng2 = random.Random(1)
REP = 300
sim_counts = {th: [] for th in range(2, k + 1)}
sim_var = []
for _ in range(REP):
    ws = []
    for _i in range(N):
        v = 0
        for _j in range(k):
            if rng2.random() < p_flip:
                v += 1 if rng2.random() < p_win else -1
        ws.append(v)
    sim_var.append(sd(ws) ** 2)
    for th in sim_counts:
        sim_counts[th].append(sum(1 for v in ws if abs(v) >= th))

print(f"\nphuong sai cua w: quan sat = {sd(w)**2:.4f}   null = {mean(sim_var):.4f}"
      f" (sd {sd(sim_var):.4f})")
z = (sd(w) ** 2 - mean(sim_var)) / max(sd(sim_var), 1e-9)
print(f"   z = {z:+.2f}")
print(f"\n{'nguong |w|>=':>13} {'quan sat':>10} {'null (TB)':>12} {'sd null':>9} {'z':>7}")
for th in range(2, k + 1):
    obs = sum(1 for v in w if abs(v) >= th)
    m, s = mean(sim_counts[th]), sd(sim_counts[th])
    zz = (obs - m) / max(s, 1e-9)
    print(f"{th:>13} {obs:>10} {m:>12.1f} {s:>9.2f} {zz:>+7.2f}")
print()
print("Doc: neu quan sat ~ null o MOI nguong -> khong co tap mau nao on dinh duoc TCVG sua;")
print("     loi ich la nhieu co dich duong nho, moi seed thang o mot cho khac.")
