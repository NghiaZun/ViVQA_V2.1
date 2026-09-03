import sys, torch
sys.path.insert(0,'src')
from model import DeterministicVQA
which = sys.argv[1]  # T0 | T2
gate = dict(use_vision_gate=True, vision_gate_init=1.0, use_type_task=True, type_loss_weight=0.2) if which=='T2' else dict(use_vision_gate=False, use_type_task=False)
torch.cuda.reset_peak_memory_stats()
m = DeterministicVQA(num_fusion_layers=2, fusion_type='text2vision', use_text_lora=True,
                     text_lora_r=16, text_lora_alpha=32, use_siglip_pooler=True, **gate).cuda()
def run(B):
    px=torch.randn(B,3,224,224).cuda(); ii=torch.randint(0,1000,(B,12)).cuda()
    am=torch.ones(B,12).long().cuda(); lab=torch.randint(0,1000,(B,8)).cuda()
    qt=torch.tensor([0,1,2,3]*((B+3)//4))[:B].cuda()
    torch.cuda.reset_peak_memory_stats()
    m.train(); out=m(pixel_values=px,input_ids=ii,attention_mask=am,labels=lab,question_types=qt if which=='T2' else None); out.total_loss.backward()
    tr=torch.cuda.max_memory_allocated()/1e6
    torch.cuda.reset_peak_memory_stats(); m.eval()
    with torch.no_grad(): m.generate(pixel_values=px[:1],input_ids=ii[:1],attention_mask=am[:1],max_length=10,num_beams=3)
    inf=torch.cuda.max_memory_allocated()/1e6
    return tr, inf
tr12, _ = run(12); _, inf1 = run(1)
print(f"[vram] {which}: train peak (batch12+backward) = {tr12:.0f} MB | inference peak (batch1 beam3) = {inf1:.0f} MB")
