"""Kiem tra gia thuyet: gate KHONG bat on ngau nhien, no BIMODAL (hai che do on dinh).

Che do A = COLOR+COUNT co chon loc (within_sample_std > 0.10) -> dung gia thuyet paper.
Che do B = COLOR+COUNT mo hoan toan (alpha ~ 1.0).

Neu che do A cho d_COLOR / d_COUNT lon hon che do B mot cach he thong, thi:
  - tinh chon loc CO tac dung, chi la khong phai seed nao cung tim duoc no
  - +0.24 nho vi no lay trung binh cua mot che do CO ich va mot che do TRUNG TINH
  - va huong di la lam sao vao duoc che do A moi lan
"""
import csv
import os
from itertools import combinations

ROOT = os.path.dirname(os.path.abspath(__file__))
SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 42]
SEL_THRESH = 0.10   # nguong 'chon loc' dung nhat quan voi muc E


def gate_stats(seed):
    """within-sample std cua alpha theo tung loai, tu gate_stats_T2_seed*.csv."""
    p = f"{ROOT}/analysis/measure/gate_stats_T2_seed{seed}.csv"
    if not os.path.exists(p):
        return None
    out = {}
    for r in csv.DictReader(open(p)):
        out[r["question_type"]] = {
            "mean": float(r["mean"]),
            "wstd": float(r["within_sample_std"]),
        }
    return out


def paired():
    p = f"{ROOT}/analysis/final/vivqa_paired_T0_T2.csv"
    return {int(r["seed"]): r for r in csv.DictReader(open(p))}


def mean(xs):
    return sum(xs) / len(xs)


def sd(xs):
    if len(xs) < 2:
        return float("nan")
    m = mean(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


def welch_t(a, b):
    """t va df Welch — khong dung scipy de khoi phu thuoc."""
    na, nb = len(a), len(b)
    va, vb = sd(a) ** 2, sd(b) ** 2
    se = (va / na + vb / nb) ** 0.5
    if se == 0:
        return float("nan"), float("nan")
    t = (mean(a) - mean(b)) / se
    df = (va / na + vb / nb) ** 2 / (
        (va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1)
    )
    return t, df


def perm_p(a, b, n_iter=None):
    """p hai phia chinh xac bang cach lay tat ca cach chia (n nho nen ve duoc het)."""
    obs = abs(mean(a) - mean(b))
    pool = a + b
    k = len(a)
    cnt = tot = 0
    for idx in combinations(range(len(pool)), k):
        s = set(idx)
        ga = [pool[i] for i in idx]
        gb = [pool[i] for i in range(len(pool)) if i not in s]
        tot += 1
        if abs(mean(ga) - mean(gb)) >= obs - 1e-12:
            cnt += 1
    return cnt / tot


# ---- 1. phan loai che do -------------------------------------------------
print("=" * 74)
print("1. PHAN LOAI CHE DO GATE (nguong within-sample std > %.2f = 'chon loc')" % SEL_THRESH)
print("=" * 74)
print(f"{'seed':>5} {'COLOR wstd':>11} {'COUNT wstd':>11} {'LOC wstd':>10} {'OBJ wstd':>10}  che do")
regime = {}
for s in SEEDS:
    g = gate_stats(s)
    if g is None:
        print(f"{s:>5}   (thieu gate_stats)")
        continue
    c = g["COLOR"]["wstd"]
    n = g["COUNT"]["wstd"]
    lo = g["LOCATION"]["wstd"]
    ob = g["OBJECT"]["wstd"]
    r = "A" if (c > SEL_THRESH and n > SEL_THRESH) else ("B" if (c <= SEL_THRESH and n <= SEL_THRESH) else "?")
    regime[s] = r
    print(f"{s:>5} {c:>11.3f} {n:>11.3f} {lo:>10.3f} {ob:>10.3f}  {r}")

A = sorted(k for k, v in regime.items() if v == "A")
B = sorted(k for k, v in regime.items() if v == "B")
M = sorted(k for k, v in regime.items() if v == "?")
print(f"\nChe do A (COLOR+COUNT chon loc): {A}  n={len(A)}")
print(f"Che do B (COLOR+COUNT mo)      : {B}  n={len(B)}")
print(f"Khong khop (chi 1 trong 2)     : {M}  n={len(M)}")
print("\n-> COLOR va COUNT dong pha %d/%d seed. Neu la nhieu ngau nhien thi xac suat"
      % (len(A) + len(B), len(regime)))
print("   dong pha het la 2*0.5^%d = %.4f (nhi thuc, hai phia)."
      % (len(regime), 2 * 0.5 ** len(regime)))

# ---- 2. hieu suat theo che do -------------------------------------------
print()
print("=" * 74)
print("2. DELTA (T2 - T0, ghep cap cung seed) THEO CHE DO")
print("=" * 74)
P = paired()
cols = ["Overall", "OBJECT", "COUNT", "COLOR", "LOCATION"]
print(f"{'phan':>8}", "".join(f"{c:>12}" for c in cols))
for name, grp in (("A", A), ("B", B)):
    grp = [s for s in grp if s in P]
    row = [mean([float(P[s]["d_" + c]) for s in grp]) for c in cols]
    print(f"{'che do '+name:>8}", "".join(f"{v:>+12.2f}" for v in row), f"  (n={len(grp)})")
print()
for c in cols:
    a = [float(P[s]["d_" + c]) for s in A if s in P]
    b = [float(P[s]["d_" + c]) for s in B if s in P]
    t, df = welch_t(a, b)
    p = perm_p(a, b)
    print(f"{c:>9}: A={mean(a):+.2f}+-{sd(a):.2f} (n={len(a)})  "
          f"B={mean(b):+.2f}+-{sd(b):.2f} (n={len(b)})  "
          f"A-B={mean(a)-mean(b):+.2f}  Welch t={t:+.2f}  perm p={p:.4f}")

# ---- 3. tuong quan lien tuc ---------------------------------------------
print()
print("=" * 74)
print("3. TUONG QUAN LIEN TUC: do chon loc cua alpha vs delta, TUNG LOAI")
print("=" * 74)


def pearson(x, y):
    n = len(x)
    mx, my = mean(x), mean(y)
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    den = (sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in y)) ** 0.5
    return num / den if den else float("nan")


for c in ["COLOR", "COUNT", "LOCATION", "OBJECT"]:
    xs, ys = [], []
    for s in SEEDS:
        g = gate_stats(s)
        if g is None or s not in P:
            continue
        xs.append(g[c]["wstd"])
        ys.append(float(P[s]["d_" + c]))
    print(f"{c:>9}: r(wstd, delta) = {pearson(xs, ys):+.3f}   n={len(xs)}")
