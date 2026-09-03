"""TAI SAO 'moi seed moi khac', va CAN HIEU UNG BAO NHIEU de T2>T0 dung o MOI seed.

Y tuong: tach phuong sai cua Delta(T2-T0) tung seed thanh hai phan
  (a) NHIEU LAY MAU TAP TEST — T0 va T2 khong dong y tren mot so mau; voi n=3001 mau, ngay ca
      hai model CO CHAT LUONG Y HET cung cho Delta khac 0. Phan nay KHONG THE giam bang cach
      train ky hon; chi giam bang cach tang so mau test.
  (b) NHIEU QUY TRINH HUAN LUYEN — khoi tao khac + quy dao toi uu hoa phan ky.

Neu (a) da chiem phan lon thi 'T2>T0 o moi seed' bi chan boi ban than tap test, va khong co
can thiep kien truc nao cuu duoc — phai bao cao bang thong ke ghep cap, khong bang 'moi seed'.

Cong thuc SE ghep cap (McNemar): voi b = so mau T2 dung/T0 sai, c = so mau T2 sai/T0 dung,
  Delta = (b - c)/n,  Var(Delta) ~ (b + c)/n^2  ->  SE = sqrt(b+c)/n
"""
import csv
import glob
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8]


def load_em(path):
    """tra ve list 0/1 theo thu tu dong."""
    if not os.path.exists(path):
        return None
    out = []
    for r in csv.DictReader(open(path)):
        v = r.get('exact_match')
        if v is None:
            return None
        out.append(1 if float(v) >= 0.5 else 0)
    return out


def mean(xs):
    return sum(xs) / len(xs)


def sd(xs):
    m = mean(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


print("=" * 92)
print("NGAN SACH NHIEU CUA Delta(T2 - T0) TUNG SEED")
print("=" * 92)
print(f"{'seed':>5} {'n':>6} {'T0':>7} {'T2':>7} {'Delta':>8} {'b':>5} {'c':>5} {'SE(ghep cap)':>13} {'Delta/SE':>9}")
rows = []
for s in SEEDS:
    t0 = load_em(f"{ROOT}/analysis/T0/T0_seed{s}.csv")
    t2 = load_em(f"{ROOT}/beam3fixed/seed{s}_ep40.csv")
    if t0 is None or t2 is None or len(t0) != len(t2):
        print(f"{s:>5}  (thieu du lieu: T0={t0 is not None} T2={t2 is not None}"
              + (f" len {len(t0)} vs {len(t2)}" if t0 and t2 else "") + ")")
        continue
    n = len(t0)
    b = sum(1 for x, y in zip(t0, t2) if x == 0 and y == 1)   # T2 dung, T0 sai
    c = sum(1 for x, y in zip(t0, t2) if x == 1 and y == 0)   # T2 sai, T0 dung
    d = (b - c) / n * 100
    se = (b + c) ** 0.5 / n * 100
    rows.append((s, n, mean(t0) * 100, mean(t2) * 100, d, b, c, se))
    print(f"{s:>5} {n:>6} {mean(t0)*100:>7.2f} {mean(t2)*100:>7.2f} {d:>+8.2f} {b:>5} {c:>5} {se:>13.2f} {d/se:>+9.2f}")

if not rows:
    raise SystemExit("khong doc duoc du lieu ghep cap")

deltas = [r[4] for r in rows]
ses = [r[7] for r in rows]
n_samp = rows[0][1]
print()
print(f"Delta qua {len(rows)} seed        : mean = {mean(deltas):+.2f}   sd giua cac seed = {sd(deltas):.2f}")
print(f"SE lay mau tap test (TB)     : {mean(ses):.2f}   <- KHONG giam duoc bang cach train khac")
print(f"So mau khong dong y (b+c) TB : {mean([r[5]+r[6] for r in rows]):.0f} / {n_samp}"
      f"  ({mean([r[5]+r[6] for r in rows])/n_samp*100:.1f}% mau)")

var_between = sd(deltas) ** 2
var_sampling = mean(ses) ** 2
print()
print("PHAN RA PHUONG SAI")
print(f"  phuong sai quan sat giua cac seed : {var_between:.3f}")
print(f"  phuong sai do lay mau tap test    : {var_sampling:.3f}"
      f"   ({min(100.0, var_sampling/var_between*100):.0f}% cua tong quan sat)")
resid = var_between - var_sampling
print(f"  phan con lai (quy trinh huan luyen): {resid:+.3f}"
      + ("   <- ~0 hoac AM: nhieu gan nhu HOAN TOAN do lay mau tap test" if resid <= 0.02 else ""))

print()
print("=" * 92)
print("CAN HIEU UNG BAO NHIEU DE Delta > 0 O MOI SEED?")
print("=" * 92)
se_eff = max(sd(deltas), mean(ses))
for k, lbl in ((1.65, "90% moi seed duong"), (2.33, "99%"), (3.0, "99.9% (~10/10 seed an toan)")):
    print(f"  can Delta >= {k:.2f} x SE = {k*se_eff:+.2f} diem   ({lbl})")
print(f"\n  Delta hien tai = {mean(deltas):+.2f}  ->  can lon hon khoang "
      f"{3.0*se_eff/max(mean(deltas),1e-9):.1f} lan")
print("\nKet luan doc duoc: neu phuong sai chu yeu do LAY MAU TAP TEST thi khong can thiep")
print("kien truc nao lam 'moi seed deu dung' duoc — phai tang hieu ung len nguong tren,")
print("hoac bao cao bang kiem dinh ghep cap thay vi bang 'dung o moi seed'.")
