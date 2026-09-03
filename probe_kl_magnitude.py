"""DO DO LON KL truoc khi chon lambda — bai hoc R-Drop (mac dinh 0.1 suyt bien so hang thanh no-op).

Y tuong can thiep: giu NGUYEN nang luc decoder nhung PHAT do lech khoi BARTpho pretrained:
    loss = CE + lambda * KL( p_finetuned(.|ctx) || p_pretrained(.|ctx) )
tren cac vi tri token DAP AN. Moc tham chieu la mot ban decoder+lm_head pretrained DONG BANG,
an cung encoder_hidden_states (cong thuc chuan, giong KL-to-SFT trong RLHF).

Khac PMI da that bai: PMI sua LUC SUY LUAN tren mot phan bo DA sup; KL ngan no sup LUC TRAIN.

Script nay KHONG train va KHONG sua src/. No chi tra loi: lambda bao nhieu thi KL chiem 10-30%
tong loss? Do o CA HAI che do: model da hoi tu (run87) va model chua train.
"""
import sys, os, torch, torch.nn.functional as F, pandas as pd
sys.path.insert(0,'src')
from probe_evidence_readout import build_model
from dataset import VQAGenDataset
from torch.utils.data import DataLoader
from transformers import AutoProcessor, AutoModelForSeq2SeqLM

DEV='cuda'; FRESH=bool(os.environ.get('FRESH'))
if FRESH:
    from model import DeterministicVQA
    sa={'vision_model':'google/siglip-base-patch16-224'}
    m=DeterministicVQA(vision_model_name=sa['vision_model'],bartpho_model_name='vinai/bartpho-syllable',
      num_fusion_layers=2,fusion_type='text2vision',use_text_lora=True,text_lora_r=16,
      text_lora_alpha=32,use_vision_gate=True,vision_gate_init=1.0,vision_gate_min_alpha=0.0,
      use_type_task=True,use_siglip_pooler=True).to(DEV)
else:
    m,sa=build_model('checkpoints_run87/best_model.pt',DEV)
m.eval()

# --- moc tham chieu: decoder + lm_head BARTpho pretrained, DONG BANG ---
ref=AutoModelForSeq2SeqLM.from_pretrained('vinai/bartpho-syllable').to(DEV).eval()
ref_dec, ref_head = ref.model.decoder, ref.lm_head
for p in ref.parameters(): p.requires_grad_(False)
print(f'moc tham chieu: BARTpho pretrained decoder ({sum(p.numel() for p in ref_dec.parameters()):,} tham so)')

vp=AutoProcessor.from_pretrained(sa.get('vision_model'))
ds=VQAGenDataset(csv_path='archive/train_split.csv',image_folder='archive/data/images/train',
  vision_processor=vp,tokenizer_name='vinai/bartpho-syllable',max_q_len=32,max_a_len=10,
  include_question_type=True,auto_detect_type=False)
dl=DataLoader(ds,batch_size=12,shuffle=True,num_workers=2)

CE=KL=H_ft=H_ref=0.0; NB=10; n=0
hook={}
def cap(mod,i,o): hook['ehs']=i[0] if i else None
for i,b in enumerate(dl):
    if i>=NB: break
    lab=b['labels'].to(DEV)
    with torch.no_grad():
        o=m(pixel_values=b['pixel_values'].to(DEV),input_ids=b['input_ids'].to(DEV),
            attention_mask=b['attention_mask'].to(DEV),labels=lab,
            question_types=b['question_type'].to(DEV).long())
        lg=o.answer_logits.float()
        # dung CHINH encoder_hidden_states cua model cho moc tham chieu (cong thuc chuan)
        from model import shift_tokens_right
        dii=shift_tokens_right(lab.clone().masked_fill(lab==-100,m.tokenizer.pad_token_id),
                               m.tokenizer.pad_token_id, m.decoder.config.decoder_start_token_id) \
            if hasattr(m.decoder.config,'decoder_start_token_id') else None
        if dii is None: break
        ehs=getattr(m,'_last_ehs',None)
        if ehs is None:      # khong luu -> tinh lai qua duong text-only cho moc
            ehs=m.encoder(input_ids=b['input_ids'].to(DEV),
                          attention_mask=b['attention_mask'].to(DEV)).last_hidden_state
            emask=b['attention_mask'].to(DEV)
        else:
            emask=None
        rd=ref_dec(input_ids=dii,encoder_hidden_states=ehs,encoder_attention_mask=emask)
        rl=ref_head(rd.last_hidden_state).float()
    msk=(lab!=-100)
    p_ft=F.log_softmax(lg,dim=-1); p_rf=F.log_softmax(rl,dim=-1)
    kl=(p_ft.exp()*(p_ft-p_rf)).sum(-1)[msk].mean()
    ce=F.cross_entropy(lg.reshape(-1,lg.size(-1)),lab.reshape(-1),ignore_index=-100)
    CE+=float(ce); KL+=float(kl)
    H_ft+=float(-(p_ft.exp()*p_ft).sum(-1)[msk].mean())
    H_ref+=float(-(p_rf.exp()*p_rf).sum(-1)[msk].mean()); n+=1

CE/=n; KL/=n; H_ft/=n; H_ref/=n
print(f'\n=== che do: {"CHUA TRAIN" if FRESH else "DA HOI TU (run87)"} | {n} batch x 12 ===')
print(f'  CE (loss nhiem vu)                 = {CE:.4f}')
print(f'  KL(finetuned || pretrained)        = {KL:.4f}')
print(f'  entropy phan bo dau ra, finetuned  = {H_ft:.4f} nat')
print(f'  entropy phan bo dau ra, pretrained = {H_ref:.4f} nat')
print(f'  -> do SUP DO entropy: {H_ref-H_ft:+.4f} nat ({100*(1-H_ft/max(H_ref,1e-9)):.1f}% hep hon)')
print(f'\n{"lambda":>8} {"lam*KL":>10} {"ty trong trong tong loss":>26}')
for lam in (0.01,0.1,0.5,1.0,5.0,20.0):
    v=lam*KL; print(f'{lam:>8} {v:>10.4f} {100*v/(CE+v):>25.1f}%')
print('\nMUC TIEU da dang ky: chon lambda sao cho KL chiem 10-30% tong loss.')
