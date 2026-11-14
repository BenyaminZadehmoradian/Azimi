"""CPU smoke test of the INFO01 notebook's real code (cells 1-20 executed unchanged).

Checks:
 1. warm-up + bit-exact restore -> classification phase gives identical weights/loss
    to the same phase without warm-up (training trajectory unchanged);
 2. same for one local-matching stage;
 3. restored-validation check matches the selected metric;
 4. timer buckets are filled and do not overlap.
Synthetic data: this checks mechanics, not accuracy.
"""
import json, sys, time, numpy as np

nb = json.load(open(sys.argv[1]))
code = {i: "".join(c["source"]) for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code"}
G = {"__name__": "__main__"}
t0 = time.time()
for i in sorted(code):
    if i > 20:
        break
    exec(compile(code[i], f"cell{i}", "exec"), G)
    print(f"cell {i} ok ({time.time()-t0:.0f}s)", flush=True)

exec(r'''
import numpy as np, tensorflow as tf
x_train=x_train[:72];y_train=y_train[:72];x_val=x_val[:64];y_val=y_val[:64]
backbone=VGGPHAXBackbone()
for i in range(1,10):
    if not backbone.bn(i).built:backbone.bn(i).build((None,backbone.bn(i).channels))
suffix=ProgressiveDigitalSuffix();_=suffix.call_from(2,tf.zeros([1,16,16,64]),training=False)
calibrate_stages(backbone,x_cal[:32],(1,2),'front/initial_test')
hybrid=ProgressiveHybridClassifier(backbone,suffix);_=hybrid(tf.convert_to_tensor(x_val[:1]),training=False)
set_physical_trainability(backbone,(1,2));set_suffix_trainability(suffix,True)
W0=[v.numpy().copy() for v in hybrid.variables]

def run(flag,label):
    global TIMING_WARMUP_BEFORE_PHASE
    TIMING_WARMUP_BEFORE_PHASE=flag
    for v,a in zip(hybrid.variables,W0):v.assign(a)
    opt=fresh_optimizer(hybrid,1e-3)
    m=run_phase_v111(hybrid,backbone,opt,label,1e-3,5,2,SEED+1,save_best=False)
    return m,[v.numpy().copy() for v in hybrid.variables],[r for r in EVAL_ROWS if r['phase']==label]

wb=TIMING['compile_warmup_sec']
mA,WA,rA=run(True,'smokeA');warm=TIMING['compile_warmup_sec']-wb
mB,WB,rB=run(False,'smokeB')
same=all(np.array_equal(a,b) for a,b in zip(WA,WB))
loss_same=[ (a['train_loss'],b['train_loss']) for a,b in zip(rA,rB)]
print('CLASSIFICATION identical weights with/without warm-up:',same)
print('per-epoch train loss A vs B:',loss_same)
print('selected',mA,mB)
print('restored check A:',PHASE_ROWS[-2]['restored_validation_check'],' B:',PHASE_ROWS[-1]['restored_validation_check'])
print('warm-up sec (phase A):',round(warm,2),' epoch secs A:',[round(r['train_epoch_sec'],2) for r in rA],' B:',[round(r['train_epoch_sec'],2) for r in rB])
assert same and all(a==b for a,b in loss_same)
assert PHASE_ROWS[-1]['restored_validation_check']['matches_selected']
V12_LOCAL_TRAIN_SAMPLES=48;INFORMATION_LOCAL_HOLDOUT_IMAGES=16
''', G)

# build_local_cache asserts teacher<=0.45 which needs teacher normalization over x_match;
# run the real normalization on the small match set.
exec(r'''
x_match=x_match[:64]
normalize_teacher_interface(backbone,suffix,hybrid,2,3)
initialize_physical_from_digital(backbone,suffix,3)
calibrate_stage(backbone.stage(3),stage_input_for(backbone,x_cal[:32],3),'smoke/stage3')
xin,yt=build_local_cache(backbone,suffix,2,3,x_match)
L=backbone.stage(3);B=backbone.bn(3);S0=[v.numpy().copy() for v in L.variables+B.variables]
V123_LOCAL_MAX_EPOCHS[3]=2
def lrun(flag):
    global TIMING_WARMUP_BEFORE_PHASE
    TIMING_WARMUP_BEFORE_PHASE=flag
    for v,a in zip(L.variables+B.variables,S0):v.assign(a)
    best,sec=_local_match_without_information(backbone,3,xin,yt)
    return best,sec,[v.numpy().copy() for v in L.variables+B.variables]
bA,sA,LA=lrun(True);bB,sB,LB=lrun(False)
lsame=all(np.array_equal(a,b) for a,b in zip(LA,LB))
print('LOCAL identical state with/without warm-up:',lsame,' best loss',bA['loss'],bB['loss'])
print('LOCAL timing rows:',LOCAL_TIMING_ROWS[-2:])
assert lsame and bA['loss']==bB['loss']
print('TIMING buckets:',{k:round(v,2) for k,v in TIMING.items()})
assert TIMING['local_matching_to_best_sec']<=TIMING['local_matching_sec']+1e-9
print('SMOKE TEST PASSED')
''', G)
