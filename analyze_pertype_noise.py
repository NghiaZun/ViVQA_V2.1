"""Hieu chinh tieu chi "ca 4 loai deu tot" theo NHIEU SEED.
Dung 10 seed T2 CUNG CAU HINH: neu chi doi seed, bao nhieu lan ca 4 loai cung di len?
Neu ty le nen da cao thi tieu chi de dat mot cach vo nghia; neu rat thap thi no la tieu chi manh.
"""
import numpy as np, pandas as pd, itertools
T=['OBJECT','COUNT','COLOR','LOCATION']; S=[0,1,2,3,4,5,6,7,8,42]
E={}; Q=None
for s in S:
    d=pd.read_csv(f'beam3fixed/seed{s}_ep40.csv')
    Q=d.question_type.values; E[s]=d.exact_match.values>.5
P=pd.DataFrame({s:{t:100*E[s][Q==t].mean() for t in T} for s in S}).T
P['ALL']=[100*E[s].mean() for s in S]
print('EM theo loai, 10 seed CUNG cau hinh:')
print(P.to_string(float_format=lambda x:f'{x:.2f}'))
print('\nSD qua seed: ' + '  '.join(f'{t} {P[t].std():.2f}' for t in T) + f'  | ALL {P.ALL.std():.2f}')
print(f'SE voi n=6:  ' + '  '.join(f'{t} {P[t].std()/np.sqrt(6):.2f}' for t in T))

print('\n=== ty le nen: doi seed thoi thi bao nhieu lan CA 4 LOAI cung di len? ===')
pairs=list(itertools.combinations(S,2)); cnt4=0; cnt3=0; rows=[]
for a,b in pairs:
    d=[P.loc[b,t]-P.loc[a,t] for t in T]
    up=sum(x>0 for x in d); rows.append(up)
    cnt4+= (up==4); cnt3+= (up>=3)
rows=np.array(rows)
print(f'{len(pairs)} cap co huong (dung 2x vi doi chieu): ')
for k in range(5):
    n=(rows==k).sum()
    print(f'  {k}/4 loai di len: {n:3d} cap ({100*n/len(pairs):5.1f}%)  '
          f'| chieu nguoc lai = {4-k}/4')
print(f'\n  P(ca 4 cung len) tren mot chieu = {cnt4/len(pairs):.3f}; '
      f'ke ca chieu nguoc = {2*cnt4/len(pairs):.3f}')
print(f'  P(>=3/4 len) = {cnt3/len(pairs):.3f}')

print('\n=== tuong quan per-type qua seed (cung cau hinh) ===')
C=P[T].corr()
print(C.to_string(float_format=lambda x:f'{x:+.2f}'))
print('\n-> cap AM manh nghia la khi mot loai len thi loai kia CO XU HUONG xuong,')
print('   ngay ca khi KHONG doi gi ve kien truc. Do la tinh chat cua DU LIEU.')

print('\n=== nguong "khong doi" cho loai KHONG bi dropout (COLOR, OBJECT) ===')
for t in ['COLOR','OBJECT']:
    se=P[t].std()/np.sqrt(6)
    print(f'  {t}: SD={P[t].std():.2f}, SE(n=6)={se:.2f} -> "khong doi" = |delta| <= {2*se:.2f} (2 SE)')
print('\n=== do nhay: n=6 phat hien duoc delta bao nhieu o moi loai (power 80%, alpha .05)? ===')
for t in T:
    print(f'  {t}: can |delta| >= {2.9*P[t].std()/np.sqrt(6):.2f} moi phat hien duoc')
