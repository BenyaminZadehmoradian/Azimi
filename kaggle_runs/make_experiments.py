"""Create one Kaggle notebook per experiment from INFO01: exactly one switch changed each."""
import json, copy
base=json.load(open('INFO01_TIMING_FIXED.ipynb'))
EXPS={
 'E1_ENDPOINT2_PLATEAU':("EXP_ENDPOINT2_PROTOCOL='reference'","EXP_ENDPOINT2_PROTOCOL='plateau'"),
 'E2_S10_RELATIVE_SCALE':("EXP_S10_RELATIVE_SCALE=False","EXP_S10_RELATIVE_SCALE=True"),
 'E3_BN_DIRECT_BIAS':("BN_DIRECT_BIAS=False","BN_DIRECT_BIAS=True"),
}
for name,(old,new) in EXPS.items():
    nb=copy.deepcopy(base); c=nb['cells'][1]; s=''.join(c['source'])
    assert s.count(old)==1,(name,s.count(old)); c['source']=s.replace(old,new).splitlines(keepends=True)
    md=nb['cells'][0]; md['source']=[f'# {name}\n\nفقط یک تغییر نسبت به INFO01: `{new}`. GPU T4، Restart & Run All، سپس ZIP نتایج را بفرستید.\n\n']+md['source']
    json.dump(nb,open(f'{name}.ipynb','w'),ensure_ascii=False,indent=1); print('wrote',name)
