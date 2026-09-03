"""Delta(T2-T0) co cung DAU o MOI seed khong — xet tung loai cau hoi.

Yeu cau cua user la 'T2>T1>T0 o moi seed'. Cau hoi thuc te: menh de nao TRONG SO CAC MENH DE
DA CO da dat duoc dieu do roi? Overall thi khong. Nhung tung loai thi chua ai kiem.
"""
import csv
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
COLS = ["Overall", "OBJECT", "COUNT", "COLOR", "LOCATION"]


def mean(xs):
    return sum(xs) / len(xs)


def sd(xs):
    m = mean(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


rows = list(csv.DictReader(open(f"{ROOT}/analysis/final/vivqa_paired_T0_T2.csv")))
print("=" * 84)
print("DAU CUA Delta(T2 - T0) THEO TUNG SEED VA TUNG LOAI")
print("=" * 84)
hdr = f"{'seed':>5}" + "".join(f"{c:>12}" for c in COLS)
print(hdr)
for r in rows:
    line = f"{int(r['seed']):>5}"
    for c in COLS:
        d = float(r["d_" + c])
        mark = "+" if d > 0 else ("0" if d == 0 else "-")
        line += f"{d:>+10.2f}{mark:>2}"
    print(line)

print()
print("=" * 84)
print("TONG HOP TINH NHAT QUAN VE DAU")
print("=" * 84)
print(f"{'loai':>10} {'mean':>8} {'sd':>7} {'so seed >0':>11} {'>=0':>6} {'<0':>5}   {'ket luan':<34}")
n = len(rows)
for c in COLS:
    ds = [float(r["d_" + c]) for r in rows]
    pos = sum(1 for d in ds if d > 0)
    nonneg = sum(1 for d in ds if d >= 0)
    neg = sum(1 for d in ds if d < 0)
    if neg == 0:
        verdict = "KHONG BAO GIO AM tren moi seed"
    elif neg <= 1:
        verdict = f"am o {neg}/{n} seed"
    else:
        verdict = f"am o {neg}/{n} seed — khong nhat quan"
    print(f"{c:>10} {mean(ds):>+8.2f} {sd(ds):>7.2f} {pos:>7}/{n} {nonneg:>5}/{n} {neg:>4}/{n}   {verdict:<34}")

print()
print("Nguong de 'moi seed deu duong' la ben vung (3 x sd giua cac seed):")
for c in COLS:
    ds = [float(r["d_" + c]) for r in rows]
    need = 3 * sd(ds)
    got = mean(ds)
    status = "DA DAT" if got >= need else f"con thieu {need-got:+.2f}"
    print(f"  {c:>10}: can Delta >= {need:+.2f}, dang co {got:+.2f}   -> {status}")
