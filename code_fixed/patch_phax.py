"""Build INFO01 from the unchanged INFO00 notebook by exact, single-occurrence edits.

Definite fixes (timing instrumentation, no change to the training trajectory) are
always applied. Controlled experiments are behind EXP_* flags whose defaults
reproduce INFO00's training path.
"""
import json, sys, copy

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
        raise SystemExit(f"cell {i}: expected exactly one match, found {n}:\n{old[:200]}")
    put(i, s.replace(old, new))


# ---------------------------------------------------------------- cell 0
put(0, """# INFO01_TIMING_FIXED (from INFO00_BN05_CONTROL_DIAGNOSTICS)

**تغییرات قطعی (مسیر آموزش را عوض نمی‌کنند):** ابزار اندازه‌گیری زمان اصلاح شده است.
- تایمر برازش محلی فقط حلقهٔ forward/backward/update را می‌شمارد؛ ساخت `tf.data`، ارزیابی محلی، ثبت پارامترهای BN، نوشتن CSV و کپی بهترین وزن‌ها بیرون از تایمرند.
- پیش از هر فاز طبقه‌بندی و هر برازش محلی، گراف آموزش روی یک دیتاست جداگانهٔ ۱ تا ۲ batchی (اندازهٔ کامل و باقی‌مانده) trace و کامپایل می‌شود و سپس تمام متغیرهای مدل و بهینه‌ساز **بیت‌به‌بیت** بازگردانده و بررسی می‌شوند. زمان این کار در `compile_warmup_sec` جدا ثبت می‌شود.
- پایان هر بازهٔ زمانی با خواندن یک مقدار از GPU همگام می‌شود.
- زمان منتخب = Σ زمان خالص فازهای طبقه‌بندی تا بهترین ایپاک + Σ زمان خالص برازش محلی تا بهترین ایپاک محلی.
- پس از بازگردانی بهترین وضعیت هر فاز، Validation دوباره سنجیده و با مقدار ثبت‌شده مقایسه می‌شود.

**آزمایش‌های کنترل‌شده (پیش‌فرض همه خاموش = همان مسیر آموزش INFO00):** `EXP_ENDPOINT2_PROTOCOL`، `EXP_S10_RELATIVE_SCALE`، `EXP_S10_HEAD_EPOCHS`، `EXP_RECOVERY_MAX_EPOCHS`، `EXP_LOCAL_MAX_EPOCHS` و پرچم موجود `BN_DIRECT_BIAS`. در هر اجرا فقط یک پرچم را تغییر دهید.

Test فقط پس از انتخاب نهایی ارزیابی می‌شود. LUT و دادهٔ CIFAR-10 را مانند قبل به Input اضافه کنید؛ GPU T4 و Restart & Run All.
""")

# ---------------------------------------------------------------- cell 1
edit(1, """BN_SCALE_UPDATE_MULTIPLIER=20.0
""", """BN_SCALE_UPDATE_MULTIPLIER=20.0

# ---------------- INFO01: measurement switches (definite fixes) ----------------
# Warm-up = one full + one remainder batch from a SEPARATE tiny dataset, then a
# bit-exact restore of every model/optimizer variable. Training batches/order unchanged.
TIMING_WARMUP_BEFORE_PHASE=True
TIMING_VERIFY_EXACT_RESTORE=True
VERIFY_RESTORED_VALIDATION=True   # re-evaluate validation after each best-state restore
# ---------------- INFO01: controlled experiments (defaults = INFO00 path) -------
# Change ONE switch per run. Selection stays validation-only.
EXP_ENDPOINT2_PROTOCOL='reference'   # 'reference' (12@1e-3,1@2e-4) | 'plateau' (baseline-like)
EXP_E2_PLATEAU=dict(lr=1.0e-3,patience=4,max_epochs=40,
                    low_lr=5.0e-4,low_patience=3,low_max_epochs=10,
                    lower_lr=2.5e-4,lower_patience=3,lower_max_epochs=10)
EXP_S10_RELATIVE_SCALE=False   # S10: one weight peak + one chain gain for all 10 classes
EXP_S10_HEAD_EPOCHS=None       # None keeps INFO00 (0 epochs)
EXP_RECOVERY_MAX_EPOCHS=None   # e.g. {6:1,9:1}; None keeps INFO00
EXP_LOCAL_MAX_EPOCHS=None      # e.g. {8:12}; None keeps INFO00
""")
edit(1, """ARTIFACT_VERSION="INFO00_BN05_CONTROL_DIAGNOSTICS"
EXPERIMENT_NAME="INFO00_BN05_CONTROL_DIAGNOSTICS"
RUN_ID=time.strftime("%Y%m%d_%H%M%S")
OUT_DIR=Path("/kaggle/working")/f"INFO00_BN05_CONTROL_DIAGNOSTICS_{RUN_ID}"
""", """ARTIFACT_VERSION="INFO01_TIMING_FIXED"
_EXP_TAGS=[]
if EXP_ENDPOINT2_PROTOCOL!='reference':_EXP_TAGS.append('E2'+EXP_ENDPOINT2_PROTOCOL.upper())
if EXP_S10_RELATIVE_SCALE:_EXP_TAGS.append('S10REL')
if EXP_S10_HEAD_EPOCHS:_EXP_TAGS.append(f'HEAD{int(EXP_S10_HEAD_EPOCHS)}')
if EXP_RECOVERY_MAX_EPOCHS:_EXP_TAGS.append('RECOV')
if EXP_LOCAL_MAX_EPOCHS:_EXP_TAGS.append('LOCAL')
if BN_DIRECT_BIAS:_EXP_TAGS.append('BNDIRECTBIAS')
assert EXP_ENDPOINT2_PROTOCOL in ('reference','plateau')
EXPERIMENT_NAME="INFO01_TIMING_FIXED"+("__"+"_".join(_EXP_TAGS) if _EXP_TAGS else "")
RUN_ID=time.strftime("%Y%m%d_%H%M%S")
OUT_DIR=Path("/kaggle/working")/f"{EXPERIMENT_NAME}_{RUN_ID}"
""")
edit(1, """for k in ('compile_warmup_sec','setup_sec','benchmark_sec','suffix_training_sec'):
    TIMING[k]=0.0
""", """for k in ('compile_warmup_sec','setup_sec','benchmark_sec','suffix_training_sec',
          'input_pipeline_build_sec','local_eval_sec','local_matching_to_best_sec',
          'restore_verification_sec'):
    TIMING[k]=0.0
""")
edit(1, """print('V11.0 CURRICULUM',PROGRESSIVE_PLAN)
""", """print('V11.0 CURRICULUM',PROGRESSIVE_PLAN)
CONFIG['INFO01']={k:globals()[k] for k in (
    'TIMING_WARMUP_BEFORE_PHASE','TIMING_VERIFY_EXACT_RESTORE','VERIFY_RESTORED_VALIDATION',
    'EXP_ENDPOINT2_PROTOCOL','EXP_E2_PLATEAU','EXP_S10_RELATIVE_SCALE','EXP_S10_HEAD_EPOCHS',
    'EXP_RECOVERY_MAX_EPOCHS','EXP_LOCAL_MAX_EPOCHS')}
CONFIG['INFO01']['parent']='INFO00_BN05_CONTROL_DIAGNOSTICS'
(OUT_DIR/'config.json').write_text(json.dumps(CONFIG,indent=2,default=str))
""")

# ---------------------------------------------------------------- cell 5 (LUT checks + benchmark)
s5 = src(5)
put(5, "_t_bench=time.perf_counter()\n" + s5 + "\nTIMING['benchmark_sec']+=time.perf_counter()-_t_bench\n")

# ---------------------------------------------------------------- cell 6 (data)
edit(6, """def make_train_ds(epoch_seed,batch):
    ds=tf.data.Dataset.from_tensor_slices((x_train,y_train)).shuffle(len(x_train),seed=epoch_seed,reshuffle_each_iteration=False)
    ds=ds.enumerate().map(lambda i,xy: augment_stateless(xy[0],xy[1],tf.stack([tf.cast(epoch_seed,tf.int32),tf.cast(i,tf.int32)])),num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch).prefetch(tf.data.AUTOTUNE)
""", """def _train_pipeline(x,y,epoch_seed,batch):
    ds=tf.data.Dataset.from_tensor_slices((x,y)).shuffle(len(x),seed=epoch_seed,reshuffle_each_iteration=False)
    ds=ds.enumerate().map(lambda i,xy: augment_stateless(xy[0],xy[1],tf.stack([tf.cast(epoch_seed,tf.int32),tf.cast(i,tf.int32)])),num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch).prefetch(tf.data.AUTOTUNE)

def make_train_ds(epoch_seed,batch):
    # Identical op sequence to INFO00; the shared pipeline also builds warm-up batches.
    return _train_pipeline(x_train,y_train,epoch_seed,batch)
""")
edit(6, """x_all,y_all=load_cifar10_train_only()
""", """_t_data=time.perf_counter()
x_all,y_all=load_cifar10_train_only()
""")
edit(6, """x_probe,y_probe=x_train[probe_local],y_train[probe_local]
""", """x_probe,y_probe=x_train[probe_local],y_train[probe_local]
TIMING['setup_sec']+=time.perf_counter()-_t_data
""")

# ---------------------------------------------------------------- cell 8 (S10 relative scale)
edit(8, """    robust=bool(str(tag).startswith("full/initial") and layer.stage not in ("stage1","stage2"))
""", """    robust=bool(str(tag).startswith("full/initial") and layer.stage not in ("stage1","stage2"))
    s10_common=bool(EXP_S10_RELATIVE_SCALE and layer.stage=="stage10")
    if s10_common:
        # EXPERIMENT: identical chain for all classes -> identical per-class gain,
        # so the relative logit scale of the mapped head is preserved.
        maxv=np.full_like(maxv,float(np.max(maxv)))
""")
edit(8, """        pc=synthesize_common_depth_per_column(postmax[None,:],tag+"/post",ALL_INVERTER_DEPTHS,robust_initial=robust)
""", """        if s10_common: postmax=np.full_like(postmax,float(np.max(postmax)))
        pc=synthesize_common_depth_per_column(postmax[None,:],tag+"/post",ALL_INVERTER_DEPTHS,robust_initial=robust)
""")

# ---------------------------------------------------------------- cell 11 (timing helpers)
edit(11, """def _make_eval(model):
""", """# ---------------- INFO01 timing helpers ----------------
def _state_arrays(variables): return [v.numpy().copy() for v in variables]
def _assign_arrays(variables,arrays):
    for v,a in zip(variables,arrays): v.assign(a)
def _arrays_equal(variables,arrays):
    return all(np.array_equal(v.numpy(),a) for v,a in zip(variables,arrays))
def _sync_on(variables):
    # Host read of values written by the last queued update: waits for the GPU stream.
    vs=[v for v in variables if tf.as_dtype(v.dtype).is_floating]   # Keras 3 dtype may be a str
    if vs: float(tf.add_n([tf.reduce_sum(tf.cast(v,tf.float32)) for v in vs]).numpy())

def train_epoch_timed(model,step,seed):
    \"\"\"Same batches and order as train_epoch. The training interval starts after
    the tf.data object and iterator exist and ends after a host read of the on-device
    accumulator, i.e. after every queued forward/backward/update has finished.\"\"\"
    t0=time.perf_counter()
    step.accumulator.assign(tf.zeros([3])); ds=make_train_ds(seed,BATCH_SIZE); it=iter(ds)
    count=int(math.ceil(len(x_train)/BATCH_SIZE))
    build=time.perf_counter()-t0
    t1=time.perf_counter()
    for j in range(0,count,STEPS_PER_EXECUTION):
        step.group(it,tf.constant(min(STEPS_PER_EXECUTION,count-j),tf.int32))
    ls,correct,n=step.accumulator.numpy().astype(np.float64)
    train=time.perf_counter()-t1
    return {"loss":float(ls/n),"accuracy":float(correct/n)},{"pipeline_build_sec":build,"train_sec":train}

def warmup_classification_step(model,opt,step):
    \"\"\"Trace and compile this phase's train graph for the full and the remainder batch
    shape on a separate tiny dataset, then restore all model/optimizer variables
    bit-exactly. The real epoch iterator and data order are untouched.\"\"\"
    if not TIMING_WARMUP_BEFORE_PHASE: return
    mv=list(model.variables); ov=list(opt.variables)
    saved_m=_state_arrays(mv); saved_o=_state_arrays(ov); saved_acc=step.accumulator.numpy().copy()
    rem=len(x_train)%BATCH_SIZE; n=BATCH_SIZE+rem
    ds=_train_pipeline(x_train[:n],y_train[:n],0,BATCH_SIZE)
    step.group(iter(ds),tf.constant(2 if rem else 1,tf.int32)); _=step.accumulator.numpy()
    _assign_arrays(mv,saved_m); _assign_arrays(ov,saved_o); step.accumulator.assign(saved_acc)
    if TIMING_VERIFY_EXACT_RESTORE and not (_arrays_equal(mv,saved_m) and _arrays_equal(ov,saved_o)):
        raise RuntimeError('warm-up restore is not bit-exact')

def _make_eval(model):
""")

# ---------------------------------------------------------------- cell 15 (run_phase_v111)
edit(15, """    best_epoch=0;best_prefix_sec=0.0;prefix_sec=0.0;wait=0;rows=[]
    kind=phase_checkpoint_kind(model,label)
    if save_best:save_checkpoint(model,backbone,label+'_epoch0',best_metric,opt,kind)
    for epoch in range(1,int(max_epochs)+1):
        t=time.perf_counter()
        try:tr=train_epoch(model,step,seed_base+epoch)
        except tf.errors.InvalidArgumentError as exc:
            restore_state(model,backbone,best_state,opt)
            record_event('training_assertion_restored_best',phase=label,reason=str(exc)[:1000])
            raise RuntimeError('training assertion failed; best valid state restored') from exc
        sec=time.perf_counter()-t;TIMING['train_forward_backward_sec']+=sec
""", """    best_epoch=0;best_prefix_sec=0.0;prefix_sec=0.0;wait=0;rows=[]
    kind=phase_checkpoint_kind(model,label)
    if save_best:save_checkpoint(model,backbone,label+'_epoch0',best_metric,opt,kind)
    warm_t=time.perf_counter()
    if int(max_epochs)>0:warmup_classification_step(model,opt,step)
    warmup_sec=time.perf_counter()-warm_t;TIMING['compile_warmup_sec']+=warmup_sec
    build_total=0.0
    for epoch in range(1,int(max_epochs)+1):
        try:tr,ti=train_epoch_timed(model,step,seed_base+epoch)
        except tf.errors.InvalidArgumentError as exc:
            restore_state(model,backbone,best_state,opt)
            record_event('training_assertion_restored_best',phase=label,reason=str(exc)[:1000])
            raise RuntimeError('training assertion failed; best valid state restored') from exc
        sec=ti['train_sec'];TIMING['train_forward_backward_sec']+=sec
        TIMING['input_pipeline_build_sec']+=ti['pipeline_build_sec'];build_total+=ti['pipeline_build_sec']
""")
edit(15, """             'train_epoch_sec':sec,'train_prefix_sec':prefix_sec,
""", """             'train_epoch_sec':sec,'pipeline_build_sec':ti['pipeline_build_sec'],'train_prefix_sec':prefix_sec,
""")
edit(15, """    restore_state(model,backbone,best_state,opt)
    summary={'phase':label,'initial_validation':initial,'validation':best_metric,
             'epochs_executed':len(rows),'best_local_epoch':best_epoch,
             'train_to_best_sec':best_prefix_sec,'actual_training_sec':prefix_sec,
             'lr':float(lr),'model_kind':kind}
""", """    restore_state(model,backbone,best_state,opt)
    restored_check=None
    if VERIFY_RESTORED_VALIDATION and len(rows)>0:
        chk=timed('restore_verification_sec',evaluate,model,x_val,y_val)
        restored_check={'accuracy':chk['accuracy'],'loss':chk['loss'],
                        'matches_selected':bool(abs(chk['accuracy']-best_metric['accuracy'])<1e-12)}
        if not restored_check['matches_selected']:
            record_event('restored_validation_mismatch',phase=label,selected=best_metric,restored=chk)
            warnings.warn(f'{label}: restored state validation {chk} != selected {best_metric}')
    summary={'phase':label,'initial_validation':initial,'validation':best_metric,
             'epochs_executed':len(rows),'best_local_epoch':best_epoch,
             'train_to_best_sec':best_prefix_sec,'actual_training_sec':prefix_sec,
             'compile_warmup_sec':warmup_sec,'pipeline_build_sec':build_total,
             'restored_validation_check':restored_check,
             'lr':float(lr),'model_kind':kind}
""")

# ---------------------------------------------------------------- cell 16 (experiment overrides)
edit(16, """for k in ('local_cache_sec','local_matching_sec','function_preserving_sec'):
    TIMING[k]=0.0
""", """# INFO01 experiment overrides (no effect with default switches).
if EXP_RECOVERY_MAX_EPOCHS:V123_RECOVERY_MAX_EPOCHS.update({int(k):int(v) for k,v in EXP_RECOVERY_MAX_EPOCHS.items()})
if EXP_LOCAL_MAX_EPOCHS:V123_LOCAL_MAX_EPOCHS.update({int(k):int(v) for k,v in EXP_LOCAL_MAX_EPOCHS.items()})
if EXP_S10_HEAD_EPOCHS is not None:
    V12_HEAD_EPOCHS=int(EXP_S10_HEAD_EPOCHS);INFORMATION_HEAD_CONFIG['head_epochs']=V12_HEAD_EPOCHS

for k in ('local_cache_sec','local_matching_sec','function_preserving_sec'):
    TIMING[k]=0.0
LOCAL_TIMING_ROWS=[]
""")

# ---------------------------------------------------------------- cell 20 (local matching timer)
edit(20, """    best=local_eval();best_weights=[v.numpy().copy() for v in state_variables];start=time.perf_counter()
    max_epochs=int(V123_LOCAL_MAX_EPOCHS.get(stage,V12_LOCAL_EPOCHS));wait=0
    for epoch in range(1,max_epochs+1):
        ds=tf.data.Dataset.from_tensor_slices((xtr,ytr)).shuffle(len(xtr),seed=SEED+stage*100+epoch,reshuffle_each_iteration=False).batch(V12_LOCAL_BATCH).prefetch(tf.data.AUTOTUNE)
        for xb,yb in ds:step(xb,yb)
        bn_record_parameters(backbone,f'local_stage{stage}_epoch{epoch}')
        va=local_eval();row={'stage':stage,'epoch':epoch,'val_local_loss':va['loss'],'val_local_mse':va['mse'],
                             'val_cosine_loss_sign':va['cosine_loss_sign'],'val_channel_nmse':va['channel_nmse'],
                             'max_epochs':max_epochs}
        LOCAL_ROWS.append(row);pd.DataFrame(LOCAL_ROWS).to_csv(OUT_DIR/'local_matching_history.csv',index=False)
        print('LOCAL',row)
        if va['loss']<best['loss']-1e-12:
            best=va;best_weights=[v.numpy().copy() for v in state_variables];wait=0
        else:
            wait+=1
            if wait>=V123_LOCAL_PATIENCE:
                print('LOCAL EARLY STOP',stage,'at',epoch,'best_loss',best['loss']);break
    for v,a in zip(state_variables,best_weights):v.assign(a)
    sec=time.perf_counter()-start;TIMING['local_matching_sec']+=sec
    bn.set_stats_frozen(True);bump_hardware_revision('local_match_selected_stage_'+str(stage))
    return best,sec
""", """    def local_warmup():
        # Compile the local step for the full and remainder batch, then restore bit-exactly.
        if not TIMING_WARMUP_BEFORE_PHASE: return
        ov=list(opt.variables);sm=_state_arrays(state_variables);so=_state_arrays(ov)
        b=V12_LOCAL_BATCH;rem=len(xtr)%b
        out=step(tf.convert_to_tensor(xtr[:b]),tf.convert_to_tensor(ytr[:b]))
        if rem:out=step(tf.convert_to_tensor(xtr[:rem]),tf.convert_to_tensor(ytr[:rem]))
        _=float(out[0].numpy())
        _assign_arrays(state_variables,sm);_assign_arrays(ov,so)
        if TIMING_VERIFY_EXACT_RESTORE and not (_arrays_equal(state_variables,sm) and _arrays_equal(ov,so)):
            raise RuntimeError(f'local warm-up restore is not bit-exact S{stage}')

    best=timed('local_eval_sec',local_eval);best_weights=[v.numpy().copy() for v in state_variables]
    max_epochs=int(V123_LOCAL_MAX_EPOCHS.get(stage,V12_LOCAL_EPOCHS));wait=0;best_epoch=0;epoch_secs=[]
    timed('compile_warmup_sec',local_warmup)
    for epoch in range(1,max_epochs+1):
        tb=time.perf_counter()
        ds=tf.data.Dataset.from_tensor_slices((xtr,ytr)).shuffle(len(xtr),seed=SEED+stage*100+epoch,reshuffle_each_iteration=False).batch(V12_LOCAL_BATCH).prefetch(tf.data.AUTOTUNE)
        TIMING['input_pipeline_build_sec']+=time.perf_counter()-tb
        t0=time.perf_counter()
        for xb,yb in ds:step(xb,yb)
        _sync_on(variables)
        epoch_secs.append(time.perf_counter()-t0)
        # Everything below is outside the local-optimization timer.
        timed('bn_experiment_diagnostic_sec',bn_record_parameters,backbone,f'local_stage{stage}_epoch{epoch}')
        va=timed('local_eval_sec',local_eval);row={'stage':stage,'epoch':epoch,'val_local_loss':va['loss'],'val_local_mse':va['mse'],
                             'val_cosine_loss_sign':va['cosine_loss_sign'],'val_channel_nmse':va['channel_nmse'],
                             'max_epochs':max_epochs,'train_epoch_sec':epoch_secs[-1]}
        LOCAL_ROWS.append(row);pd.DataFrame(LOCAL_ROWS).to_csv(OUT_DIR/'local_matching_history.csv',index=False)
        print('LOCAL',row)
        if va['loss']<best['loss']-1e-12:
            best=va;best_weights=[v.numpy().copy() for v in state_variables];wait=0;best_epoch=epoch
        else:
            wait+=1
            if wait>=V123_LOCAL_PATIENCE:
                print('LOCAL EARLY STOP',stage,'at',epoch,'best_loss',best['loss']);break
    for v,a in zip(state_variables,best_weights):v.assign(a)
    sec_all=float(sum(epoch_secs));sec_best=float(sum(epoch_secs[:best_epoch]))
    TIMING['local_matching_sec']+=sec_all;TIMING['local_matching_to_best_sec']+=sec_best
    LOCAL_TIMING_ROWS.append({'stage':stage,'epochs_executed':len(epoch_secs),'best_local_epoch':best_epoch,
                              'train_all_sec':sec_all,'train_to_best_sec':sec_best,
                              'epoch_secs':[round(s,4) for s in epoch_secs]})
    pd.DataFrame(LOCAL_TIMING_ROWS).to_csv(OUT_DIR/'local_timing.csv',index=False)
    bn.set_stats_frozen(True);bump_hardware_revision('local_match_selected_stage_'+str(stage))
    return best,sec_best
""")

# ---------------------------------------------------------------- cell 22 (endpoint2 protocol)
edit(22, """set_physical_trainability(backbone,(1,2));set_suffix_trainability(suffix,True)
opt=fresh_optimizer(hybrid,PROGRESSIVE_PLAN[2]['lr'])
endpoint2_float=run_phase_v111(hybrid,backbone,opt,'v12_endpoint2_float',PROGRESSIVE_PLAN[2]['lr'],V123_ENDPOINT2_PATIENCE,V123_ENDPOINT2_MAX_EPOCHS,SEED+12020,
                                save_best=False,diagnose_gradient=False)
eval_and_record(hybrid,'endpoint2_float_selected','endpoint')

if PHASE_ROWS[-1]['epochs_executed'] != V123_ENDPOINT2_MAX_EPOCHS:
    raise RuntimeError('endpoint2 schedule guard: all 12 high-LR epochs must execute')
""", """set_physical_trainability(backbone,(1,2));set_suffix_trainability(suffix,True)
if EXP_ENDPOINT2_PROTOCOL=='reference':
    opt=fresh_optimizer(hybrid,PROGRESSIVE_PLAN[2]['lr'])
    endpoint2_float=run_phase_v111(hybrid,backbone,opt,'v12_endpoint2_float',PROGRESSIVE_PLAN[2]['lr'],V123_ENDPOINT2_PATIENCE,V123_ENDPOINT2_MAX_EPOCHS,SEED+12020,
                                    save_best=False,diagnose_gradient=False)
else:
    # EXPERIMENT E1: baseline-like plateau protocol (1e-3 at batch 32 = 5e-4 at batch 16
    # under linear scaling); patience-driven, validation-only selection.
    P=EXP_E2_PLATEAU
    opt=fresh_optimizer(hybrid,P['lr'])
    endpoint2_float=run_phase_v111(hybrid,backbone,opt,'v12_endpoint2_float',P['lr'],P['patience'],P['max_epochs'],SEED+12020,
                                    save_best=False,diagnose_gradient=False)
eval_and_record(hybrid,'endpoint2_float_selected','endpoint')

if EXP_ENDPOINT2_PROTOCOL=='reference' and PHASE_ROWS[-1]['epochs_executed'] != V123_ENDPOINT2_MAX_EPOCHS:
    raise RuntimeError('endpoint2 schedule guard: all 12 high-LR epochs must execute')
""")
edit(22, """set_physical_trainability(backbone,(1,2));set_suffix_trainability(suffix,True)
opt=fresh_optimizer(hybrid,V123_ENDPOINT2_REFINE_LR)
endpoint2_refined=run_phase_v111(hybrid,backbone,opt,'v12_endpoint2_float_refine',V123_ENDPOINT2_REFINE_LR,
                                 V123_ENDPOINT2_REFINE_PATIENCE,V123_ENDPOINT2_REFINE_MAX_EPOCHS,
                                 SEED+12070,save_best=False,diagnose_gradient=False)
""", """set_physical_trainability(backbone,(1,2));set_suffix_trainability(suffix,True)
if EXP_ENDPOINT2_PROTOCOL=='reference':
    opt=fresh_optimizer(hybrid,V123_ENDPOINT2_REFINE_LR)
    endpoint2_refined=run_phase_v111(hybrid,backbone,opt,'v12_endpoint2_float_refine',V123_ENDPOINT2_REFINE_LR,
                                     V123_ENDPOINT2_REFINE_PATIENCE,V123_ENDPOINT2_REFINE_MAX_EPOCHS,
                                     SEED+12070,save_best=False,diagnose_gradient=False)
else:
    opt=fresh_optimizer(hybrid,P['low_lr'])
    run_phase_v111(hybrid,backbone,opt,'v12_endpoint2_float_low',P['low_lr'],P['low_patience'],P['low_max_epochs'],
                   SEED+12040,save_best=False,diagnose_gradient=False)
    opt=fresh_optimizer(hybrid,P['lower_lr'])
    endpoint2_refined=run_phase_v111(hybrid,backbone,opt,'v12_endpoint2_float_refine',P['lower_lr'],
                                     P['lower_patience'],P['lower_max_epochs'],
                                     SEED+12070,save_best=False,diagnose_gradient=False)
""")

# ---------------------------------------------------------------- cell 26 (S10 mapping)
edit(26, """peak=np.max(np.abs(w),axis=(0,1),keepdims=True);wn=w/np.maximum(peak,1e-9)
""", """# INFO00 normalizes every class column by its own peak (relative logit scale lost).
# EXPERIMENT EXP_S10_RELATIVE_SCALE: one global peak keeps Gp-Gn proportional to w
# across classes; the column denominator Gp+Gn is identical for every column.
peak=(np.max(np.abs(w)) if EXP_S10_RELATIVE_SCALE else np.max(np.abs(w),axis=(0,1),keepdims=True));wn=w/np.maximum(peak,1e-9)
""")

# ---------------------------------------------------------------- cell 29 (timing summary)
edit(29, """selected_full_phase_sec=float(sum(r['train_to_best_sec'] for r in PHASE_ROWS))
# Local matching is a required sequential part of the deployed path. With only
# adaptive local matching, count its complete measured cost instead of hiding it.
selected_path_train_sec=selected_full_phase_sec+float(TIMING['local_matching_sec'])
actual_all_training_sec=float(TIMING['train_forward_backward_sec']+TIMING['local_matching_sec'])
""", """# INFO01 definition (same concept as INFO00, cleaner instruments):
#   classification: pure train-loop seconds up to each phase's selected epoch (0 if epoch 0)
#   local matching: pure optimization seconds up to each stage's selected local epoch
#   compile/warm-up, data/pipeline construction, caches, calibration, R/depth selection,
#   teacher scaling, validation/test, diagnostics, logging and checkpoints are excluded.
selected_full_phase_sec=float(sum(r['train_to_best_sec'] for r in PHASE_ROWS))
selected_local_matching_sec=float(TIMING['local_matching_to_best_sec'])
selected_path_train_sec=selected_full_phase_sec+selected_local_matching_sec
actual_all_training_sec=float(TIMING['train_forward_backward_sec']+TIMING['local_matching_sec'])
if selected_path_train_sec>actual_all_training_sec+1e-6:
    raise RuntimeError('selected path cannot exceed all executed training')
TIMING_DEFINITION={'selected_path':'sum(train_to_best_sec over classification phases)+sum(local to-best optimization sec)',
    'excluded':['compile_warmup_sec','input_pipeline_build_sec','local_cache_sec','hardware_codesign_sec',
                'function_preserving_sec','validation_selection_sec','local_eval_sec','restore_verification_sec',
                'final_test_sec','train_diagnostic_sec','information_diagnostic_sec','bn_experiment_diagnostic_sec',
                'physical_audit_sec','diagnostic_sec','checkpoint_sec','benchmark_sec','setup_sec'],
    'gpu_sync':'host read of on-device accumulator / trained-variable sum at interval end',
    'warmup':'separate 1-2 batch dataset; bit-exact restore verified' if TIMING_WARMUP_BEFORE_PHASE else 'disabled',
    'INFO00_legacy_reference':{'selected_full_phase_sec':3307.070204288,'local_matching_sec':358.19750007599987,
        'selected_path_train_sec':3665.2677043639997,
        'note':'legacy local timer included local eval, BN logging, CSV writes, dataset construction and compilation; '
               'legacy phase timers included first-epoch tracing/compilation'}}
pd.DataFrame([{k:v for k,v in r.items() if k not in ('initial_validation','validation','restored_validation_check')}
              for r in PHASE_ROWS]).to_csv(OUT_DIR/'timing_phase_breakdown.csv',index=False)
""")
edit(29, """identified=sum(v for k,v in TIMING.items() if k not in ('total_wall_sec','suffix_training_sec'))
TIMING['unattributed_session_sec']=TIMING['total_wall_sec']-identified
""", """# suffix_training_sec and local_matching_to_best_sec are subsets of other buckets.
identified=sum(v for k,v in TIMING.items() if k not in ('total_wall_sec','suffix_training_sec','local_matching_to_best_sec','unattributed_session_sec'))
TIMING['unattributed_session_sec']=TIMING['total_wall_sec']-identified
if TIMING['unattributed_session_sec']<-1.0:
    raise RuntimeError(f"timer overlap: exclusive buckets exceed wall time by {-TIMING['unattributed_session_sec']:.2f}s")
""")
edit(29, """         'selected_local_matching_sec':float(TIMING['local_matching_sec']),
         'actual_all_training_sec_including_local':actual_all_training_sec,
""", """         'selected_local_matching_sec':selected_local_matching_sec,
         'local_matching_all_executed_sec':float(TIMING['local_matching_sec']),
         'actual_all_training_sec_including_local':actual_all_training_sec,
         'compile_warmup_sec':float(TIMING['compile_warmup_sec']),
         'total_wall_sec':float(TIMING['total_wall_sec']),
         'timing_definition':TIMING_DEFINITION,'local_timing':LOCAL_TIMING_ROWS,
         'info01_config':CONFIG['INFO01'],
""")

nb.setdefault("metadata", {})["info01_patch"] = "built from INFO00_BN05_CONTROL_DIAGNOSTICS by patch_phax.py"
json.dump(nb, open(DST, "w"), ensure_ascii=False, indent=1)
print("wrote", DST)
