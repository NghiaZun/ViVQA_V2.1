"""Kiem dinh NULL CO DIEU KIEN cho claim per-type.

Van de: per-type delta nghich the nhau (OBJECT-COLOR r=-0.58) vi tong EM gan nhu bao toan.
Nen "COLOR tut 1.36" tu no KHONG noi len gi — phai hoi: cho truoc cu dich TONG THE cung co,
pattern per-type nay co bat thuong so voi pattern ma CHI DOI SEED tao ra khong?

Null: 45 cap seed T2 cung cau hinh. Hoi quy Delta_type ~ beta*Delta_tong (qua goc toa do).
Phan du = chu ky rieng cua module. So phan du quan sat voi phan phoi phan du cua null.
"""
import numpy as np, pandas as pd, itertools, os, sys
T=['OBJECT','COUNT','COLOR','LOCATION']; S=[0,1,2,3,4,5,6,7,8,42]
E={}; Q=None
for s in S:
    d=pd.read_csv(f'beam3fixed/seed{s}_ep40.csv'); Q=d.question_type.values
    E[s]=d.exact_match.values>.5
P=pd.DataFrame({s:{t:100*E[s][Q==t].mean() for t in T} for s in S}).T
P['ALL']=[100*E[s].mean() for s in S]

# --- dung null tu 90 cap co huong (ca hai chieu) ---
N=[]
for a,b in itertools.permutations(S,2):
    r={'ALL':P.loc[b,'ALL']-P.loc[a,'ALL']}
    for t in T: r[t]=P.loc[b,t]-P.loc[a,t]
    N.append(r)
N=pd.DataFrame(N)
print(f'null: {len(N)} cap seed cung cau hinh')
print(f'  Delta tong: sd {N.ALL.std():.3f}, khoang [{N.ALL.min():+.2f}, {N.ALL.max():+.2f}]')

BETA={}
print(f'\n{"loai":<10} {"beta":>7} {"y nghia":<44} {"sd phan du":>11}')
for t in T:
    b=float((N[t]*N.ALL).sum()/(N.ALL**2).sum())      # hoi quy qua goc
    res=N[t]-b*N.ALL; BETA[t]=(b,res.std())
    share={'OBJECT':41.6,'COUNT':14.8,'COLOR':20.8,'LOCATION':22.8}[t]
    print(f'{t:<10} {b:>7.3f} {f"1pp tong -> {b:+.2f}pp loai nay (share {share}%)":<44} {res.std():>11.3f}')

# --- do arm quan sat ---
arm=sys.argv[1] if len(sys.argv)>1 else 'gd13'
seeds=[int(x) for x in sys.argv[2].split(',')] if len(sys.argv)>2 else [0,1]
rows=[]
for s in seeds:
    f=f'analysis/{arm}/s{s}.csv'
    if not os.path.exists(f): continue
    d=pd.read_csv(f); o=d.exact_match.values>.5; q=d.question_type.values
    r={'seed':s,'ALL':100*o.mean()-P.loc[s,'ALL']}
    for t in T: r[t]=100*o[q==t].mean()-P.loc[s,t]
    rows.append(r)
O=pd.DataFrame(rows)
if not len(O): print(f'\nkhong co du lieu cho {arm}'); sys.exit()
print(f'\n=== {arm}, n={len(O)} seed ===')
print(O.to_string(index=False,float_format=lambda x:f'{x:+.2f}'))
dall=O.ALL.mean()
print(f'\nDelta tong trung binh: {dall:+.3f}')
print(f'\n{"loai":<10} {"quan sat":>9} {"ky vong":>8} {"phan du":>8} {"z":>6} {"p 2 phia":>9} {"ket luan":<32}')
for t in T:
    b,sd=BETA[t]; obs=O[t].mean(); exp=b*dall; res=obs-exp
    se=sd/np.sqrt(len(O)); z=res/se
    from scipy import stats as st
    p=2*(1-st.norm.cdf(abs(z)))
    verdict='BAT THUONG (rieng cua module)' if p<0.05 else 'trong dai nhieu seed'
    print(f'{t:<10} {obs:>+9.2f} {exp:>+8.2f} {res:>+8.2f} {z:>6.2f} {p:>9.4f} {verdict:<32}')
print(f'\nDoc: "ky vong" = phan cu dich tong the tu nhien roi vao loai do khi CHI doi seed.')
print(f'     Chi "phan du" moi la chu ky rieng cua can thiep.')
