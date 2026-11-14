"""Baseline with the same INFO01 timing instruments + E0 schedule-control switch."""
import json, sys

SRC, DST = sys.argv[1], sys.argv[2]
nb = json.load(open(SRC))
cells = nb["cells"]


def src(i):
    return "".join(cells[i]["source"])


def put(i, text):
    cells[i]["source"] = text.splitlines(keepends=True)
    if cells[i]["cell_type"] == "code":
        cells[i]["outputs"] = []
        cells[i]["execution_count"] = None


def edit(i, old, new):
    s = src(i)
    n = s.count(old)
    if n != 1:
        raise SystemExit(f"cell {i}: expected one match, found {n}:\n{old[:200]}")
    put(i, s.replace(old, new))


put(0, src(0) + """

**INFO01 timing + کنترل E0.** ابزار زمان مانند INFO01 است: warm-up/کامپایل روی دیتاست جداگانه با بازگردانی بیت‌به‌بیت، ساخت `tf.data` بیرون از تایمر، همگام‌سازی GPU با خواندن accumulator، و محاسبهٔ «زمان مسیر منتخب» (مجموع زمان خالص هر فاز تا بهترین ایپاک) داخل خود نوت‌بوک.

`PHASES_MODE='reference'` همان بیس‌لاین قبلی است. `PHASES_MODE='phax_endpoint2_matched'` کنترل E0 است: همان برنامهٔ آموزش endpoint2 در PHAX (batch=32، دوازده ایپاک ثابت با 1e-3، یک ایپاک با 2e-4، حداکثر دو ایپاک با 5e-5 و patience=2، بهینه‌ساز تازه در هر فاز). اگر این کنترل هم حدود ۹۰٫۹٪ Validation بدهد، فاصله از برنامهٔ آموزش است، نه از بخش فیزیکی.
""")

edit(1, """BATCH_SIZE=16; EVAL_BATCH_SIZE=64; STEPS_PER_EXECUTION=16
VAL_SIZE=5000; TOTAL_EPOCH_CAP=65
BN_MOMENTUM=.90; BN_EPSILON=.001
CIFAR_ROOT=None  # optional explicit folder containing data_batch_1 ... test_batch
HW_REFERENCE_PATH=None  # optional summary.json or diagnostics ZIP
PHASES=(('joint',5e-4,4,SEED+10000),('live_bn_low',2.5e-4,3,SEED+12500),('live_bn_lower',1.25e-4,3,SEED+14000))
""", """PHASES_MODE='reference'   # 'reference' | 'phax_endpoint2_matched' (control E0)
TIMING_WARMUP_BEFORE_PHASE=True; TIMING_VERIFY_EXACT_RESTORE=True
assert PHASES_MODE in ('reference','phax_endpoint2_matched')
BATCH_SIZE=16 if PHASES_MODE=='reference' else 32; EVAL_BATCH_SIZE=64; STEPS_PER_EXECUTION=16
VAL_SIZE=5000; TOTAL_EPOCH_CAP=65
BN_MOMENTUM=.90; BN_EPSILON=.001
CIFAR_ROOT=None  # optional explicit folder containing data_batch_1 ... test_batch
HW_REFERENCE_PATH=None  # optional summary.json or diagnostics ZIP
if PHASES_MODE=='reference':
    # (label, lr, patience, seed, max_epochs or None=remaining cap); one shared Adam.
    PHASES=(('joint',5e-4,4,SEED+10000,None),('live_bn_low',2.5e-4,3,SEED+12500,None),('live_bn_lower',1.25e-4,3,SEED+14000,None))
    FRESH_OPTIMIZER_PER_PHASE=False
else:
    # PHAX INFO00 endpoint2: 12 fixed @1e-3, 1 @2e-4, <=2 @5e-5 (patience 2); fresh Adam per phase.
    PHASES=(('joint_fixed12',1e-3,12,SEED+12020,12),('refine',2e-4,2,SEED+12070,1),('low',5e-5,2,SEED+12120,2))
    FRESH_OPTIMIZER_PER_PHASE=True
""")
edit(1, """OUT_DIR=Path('/kaggle/working')/f'phax_digital_baseline_{RUN_ID}';OUT_DIR.mkdir(parents=True,exist_ok=True)
""", """OUT_DIR=Path('/kaggle/working')/f'phax_digital_baseline_{PHASES_MODE}_{RUN_ID}';OUT_DIR.mkdir(parents=True,exist_ok=True)
""")
edit(1, """TIMING={k:0. for k in ('data_setup_sec','train_loop_sec','validation_selection_sec','checkpoint_sec','inference_benchmark_sec','final_test_sec','final_train_eval_sec')}
""", """TIMING={k:0. for k in ('data_setup_sec','train_loop_sec','validation_selection_sec','checkpoint_sec','inference_benchmark_sec','final_test_sec','final_train_eval_sec',
                        'compile_warmup_sec','input_pipeline_build_sec')}
""")

edit(2, """def make_train_ds(epoch_seed,batch):
    ds=tf.data.Dataset.from_tensor_slices((x_train,y_train)).shuffle(len(x_train),seed=epoch_seed,reshuffle_each_iteration=False)
    ds=ds.enumerate().map(lambda i,xy: augment_stateless(xy[0],xy[1],tf.stack([tf.cast(epoch_seed,tf.int32),tf.cast(i,tf.int32)])),num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch).prefetch(tf.data.AUTOTUNE)
""", """def _train_pipeline(x,y,epoch_seed,batch):
    ds=tf.data.Dataset.from_tensor_slices((x,y)).shuffle(len(x),seed=epoch_seed,reshuffle_each_iteration=False)
    ds=ds.enumerate().map(lambda i,xy: augment_stateless(xy[0],xy[1],tf.stack([tf.cast(epoch_seed,tf.int32),tf.cast(i,tf.int32)])),num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch).prefetch(tf.data.AUTOTUNE)

def make_train_ds(epoch_seed,batch):
    return _train_pipeline(x_train,y_train,epoch_seed,batch)
""")

edit(3, """        'baseline_logit_temperature':1,'environment':ENVIRONMENT,'front':False,'ADC_DAC':False}
""", """        'baseline_logit_temperature':1,'environment':ENVIRONMENT,'front':False,'ADC_DAC':False,
        'phases_mode':PHASES_MODE,'fresh_optimizer_per_phase':FRESH_OPTIMIZER_PER_PHASE,
        'timing_warmup_before_phase':TIMING_WARMUP_BEFORE_PHASE}
""")

edit(5, """def _make_eval(model):
""", """def train_epoch_timed(model,step,seed):
    t0=time.perf_counter()
    step.accumulator.assign(tf.zeros([3])); ds=make_train_ds(seed,BATCH_SIZE); it=iter(ds)
    count=int(math.ceil(len(x_train)/BATCH_SIZE)); build=time.perf_counter()-t0
    t1=time.perf_counter()
    for j in range(0,count,STEPS_PER_EXECUTION):
        step.group(it,tf.constant(min(STEPS_PER_EXECUTION,count-j),tf.int32))
    ls,correct,n=step.accumulator.numpy().astype(np.float64)   # host read = GPU sync
    return {"loss":float(ls/n),"accuracy":float(correct/n)},{"pipeline_build_sec":build,"train_sec":time.perf_counter()-t1}

def warmup_classification_step(model,opt,step):
    if not TIMING_WARMUP_BEFORE_PHASE: return
    mv=list(model.variables); ov=list(opt.variables)
    sm=[v.numpy().copy() for v in mv]; so=[v.numpy().copy() for v in ov]; sa=step.accumulator.numpy().copy()
    rem=len(x_train)%BATCH_SIZE; n=BATCH_SIZE+rem
    step.group(iter(_train_pipeline(x_train[:n],y_train[:n],0,BATCH_SIZE)),tf.constant(2 if rem else 1,tf.int32)); _=step.accumulator.numpy()
    for v,a in zip(mv,sm): v.assign(a)
    for v,a in zip(ov,so): v.assign(a)
    step.accumulator.assign(sa)
    if TIMING_VERIFY_EXACT_RESTORE and not (all(np.array_equal(v.numpy(),a) for v,a in zip(mv,sm)) and all(np.array_equal(v.numpy(),a) for v,a in zip(ov,so))):
        raise RuntimeError('warm-up restore is not bit-exact')

def _make_eval(model):
""")
edit(5, """    optimizer.learning_rate.assign(lr);step=make_train_step(model,optimizer)
    best=timed('validation_selection_sec',evaluate,model,x_val,y_val);state=snapshot(model,optimizer);wait=0;used=0
    for epoch in range(1,max_epochs+1):
        t=time.perf_counter();tr=train_epoch(model,step,seed+epoch);train_sec=time.perf_counter()-t;TIMING['train_loop_sec']+=train_sec
""", """    optimizer.learning_rate.assign(lr);step=make_train_step(model,optimizer)
    best=timed('validation_selection_sec',evaluate,model,x_val,y_val);state=snapshot(model,optimizer);wait=0;used=0
    best_epoch=0;prefix=0.;best_prefix=0.
    if max_epochs>0:timed('compile_warmup_sec',warmup_classification_step,model,optimizer,step)
    for epoch in range(1,max_epochs+1):
        tr,ti=train_epoch_timed(model,step,seed+epoch);train_sec=ti['train_sec'];TIMING['train_loop_sec']+=train_sec
        TIMING['input_pipeline_build_sec']+=ti['pipeline_build_sec'];prefix+=train_sec
""")
edit(5, """        if better(va,best):best=dict(va);state=snapshot(model,optimizer);wait=0;save_checkpoint(label+'_best',best)
        else:wait+=1
        if wait>=patience:break
    restore(model,optimizer,state);save_checkpoint(label+'_best',best)
    PHASE_ROWS.append({'phase':label,'epochs_executed':used,**best});return best
""", """        if better(va,best):best=dict(va);state=snapshot(model,optimizer);wait=0;best_epoch=epoch;best_prefix=prefix;save_checkpoint(label+'_best',best)
        else:wait+=1
        if wait>=patience:break
    restore(model,optimizer,state);save_checkpoint(label+'_best',best)
    PHASE_ROWS.append({'phase':label,'epochs_executed':used,'best_local_epoch':best_epoch,
                       'train_to_best_sec':best_prefix,'actual_training_sec':prefix,**best});return best
""")

edit(6, """for label,lr,patience,seed in PHASES:
    remaining=TOTAL_EPOCH_CAP-CUM_EPOCH
    if remaining<=0:break
    run_digital_phase(label,lr,patience,seed,remaining)
""", """for label,lr,patience,seed,cap in PHASES:
    remaining=TOTAL_EPOCH_CAP-CUM_EPOCH
    if remaining<=0:break
    if FRESH_OPTIMIZER_PER_PHASE:
        optimizer=tf.keras.optimizers.Adam(lr);optimizer.build(model.trainable_variables)
    run_digital_phase(label,lr,patience,seed,remaining if cap is None else min(cap,remaining))
""")

edit(7, """summary={'artifact':'PHAX_DIGITAL_BASELINE_FP32_VGG9_CIFAR10','run_id':RUN_ID,'dataset':'CIFAR-10','seed':SEED,
""", """selected_path_train_sec=float(sum(r['train_to_best_sec'] for r in PHASE_ROWS))
summary={'artifact':'PHAX_DIGITAL_BASELINE_FP32_VGG9_CIFAR10','run_id':RUN_ID,'dataset':'CIFAR-10','seed':SEED,
         'phases_mode':PHASES_MODE,'selected_path_train_sec':selected_path_train_sec,
         'actual_training_sec':float(TIMING['train_loop_sec']),
         'timing_definition':'pure train-loop sec to each phase best epoch; warm-up/compile, pipeline build, validation excluded',
         'legacy_reference_selected_path_train_sec':2072.356839279,
""")

nb.setdefault("metadata", {})["info01_patch"] = "baseline with INFO01 timing and E0 control (patch_base.py)"
json.dump(nb, open(DST, "w"), ensure_ascii=False, indent=1)
print("wrote", DST)
