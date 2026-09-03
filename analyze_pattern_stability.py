"""Bien the nao lam KHUON MAU GATING NHAT QUAN qua cac seed?

Tat ca cac bien the TCVG truoc day chi duoc danh gia bang ACCURACY (muc H: deu trung tinh
71.5-71.9 nen bi ket luan la 'khong co tac dung'). Nhung cau hoi cua user la khac:
co bien the nao lam khuon mau gating GIONG NHAU qua cac seed khong?
Do la thu chua bao gio duoc do. File nay do no.

Do bat on goc (comment trong model.py): type_embedding init N(0,1), hidden=1024 -> norm ~32,
va do duoc chi xoay 0.3-0.6 do sau 21320 buoc -> 'loai nao bi gating' bi DONG BANG o gia tri
khoi tao ngau nhien theo seed.
"""
import csv
import os
import glob

ROOT = os.path.dirname(os.path.abspath(__file__))
TYPES = ["COLOR", "COUNT", "LOCATION", "OBJECT"]
SEL = 0.10   # nguong 'chon loc', nhat quan voi muc E

# (ten hien thi, mo ta can thiep, glob de tim file theo seed)
VARIANTS = [
    ("baseline T2",   "khong can thiep",                 "analysis/measure/gate_stats_T2_seed{s}.csv"),
    ("init02",        "type_emb init std=0.02",          "analysis/tcvginit/gate_stats_init02_seed{s}.csv"),
    ("temblr",        "LR type_emb x100",                "analysis/temblr/gate_stats_lr100_seed{s}.csv"),
    ("tcvgB",         "chuan hoa type_emb (huong+bien do)", "analysis/*/gate_stats_tcvgB_s{s}.csv"),
    ("tcvgA",         "type_null offset theo loai",      "analysis/*/gate_stats_tcvgA_s{s}.csv"),
    ("tcvgAB",        "ca A lan B",                      "analysis/*/gate_stats_tcvgAB_s{s}.csv"),
]
SEEDS = [42, 0, 1]


def load(pattern, seed):
    hits = glob.glob(os.path.join(ROOT, pattern.format(s=seed)))
    if not hits:
        return None
    out = {}
    for r in csv.DictReader(open(hits[0])):
        out[r["question_type"]] = (float(r["mean"]), float(r["within_sample_std"]))
    return out if all(t in out for t in TYPES) else None


def sig(stats):
    """Chu ky nhi phan: loai nao chon loc."""
    return "".join("1" if stats[t][1] > SEL else "0" for t in TYPES)


print("=" * 88)
print("KHUON MAU GATING QUA SEED  (chu ky = COLOR/COUNT/LOCATION/OBJECT, 1 = chon loc)")
print("=" * 88)
rows = []
for name, desc, pat in VARIANTS:
    per = {}
    for s in SEEDS:
        st = load(pat, s)
        if st:
            per[s] = st
    if len(per) < 2:
        print(f"\n{name:12s} — thieu du lieu ({len(per)}/3 seed)")
        continue
    sigs = {s: sig(per[s]) for s in per}
    uniq = len(set(sigs.values()))
    print(f"\n{name:12s} ({desc})")
    print(f"  {'seed':>5} {'COLOR':>14} {'COUNT':>14} {'LOCATION':>14} {'OBJECT':>14}   chu ky")
    for s in SEEDS:
        if s not in per:
            continue
        cells = "".join(f"{per[s][t][0]:>7.2f}/{per[s][t][1]:<6.3f}" for t in TYPES)
        print(f"  {s:>5} {cells}   {sigs[s]}")
    verdict = "NHAT QUAN" if uniq == 1 else f"khac nhau ({uniq} chu ky tren {len(per)} seed)"
    print(f"  -> {verdict}")
    rows.append((name, uniq, len(per), sigs))

print()
print("=" * 88)
print("TONG HOP")
print("=" * 88)
print(f"{'bien the':14s} {'so chu ky':>10} {'so seed':>8}   ket luan")
for name, uniq, n, sigs in rows:
    v = "NHAT QUAN qua seed" if uniq == 1 else "van phu thuoc seed"
    print(f"{name:14s} {uniq:>10} {n:>8}   {v}   {list(sigs.values())}")

print()
print("Luu y doc ket qua: chu ky nhat quan MA accuracy khong doi (muc H: moi bien the 71.5-71.9)")
print("van la ket qua co gia tri — no bien 'gate bat on theo seed' thanh 'gate on dinh',")
print("tuc sua duoc diem yeu kho bao ve nhat cua khoa luan ma khong mat gi.")
