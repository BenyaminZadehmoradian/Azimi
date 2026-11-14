import json,sys,numpy as np
nb=json.load(open(sys.argv[1])); code={i:''.join(c['source']) for i,c in enumerate(nb['cells']) if c['cell_type']=='code'}
for mode in ('reference','phax_endpoint2_matched'):
    G={'__name__':'__main__'}
    exec(code[1].replace("PHASES_MODE='reference'",f"PHASES_MODE='{mode}'"),G)
    exec(code[2].split('t=time.perf_counter()\nx_all')[0],G)   # functions only
    rng=np.random.default_rng(0)
    G.update(x_train=rng.uniform(-.5,.5,(72,32,32,3)).astype(np.float32),y_train=np.tile(np.arange(10),8)[:72].astype(np.int64),
             x_val=rng.uniform(-.5,.5,(32,32,32,3)).astype(np.float32),y_val=np.tile(np.arange(10),4)[:32].astype(np.int64),CONFIG={})
    G['TOTAL_EPOCH_CAP']=4 if mode=='reference' else 15
    exec(code[4].replace('widths=(64,64,128,128,256,256,512,512)','widths=(8,8,8,8,8,8,8,8)'),G)
    exec(code[5],G)
    G['PHASES']=tuple((l,lr,1 if mode=='reference' else p,s,c if c is None else min(c,2)) for l,lr,p,s,c in G['PHASES'])
    exec(code[6],G)
    rows=G['PHASE_ROWS']; sel=sum(r['train_to_best_sec'] for r in rows)
    print(mode,'batch',G['BATCH_SIZE'],'fresh_opt',G['FRESH_OPTIMIZER_PER_PHASE'],'phases',[(r['phase'],r['epochs_executed'],r['best_local_epoch'],round(r['train_to_best_sec'],2)) for r in rows],
          'selected',round(sel,2),'all',round(G['TIMING']['train_loop_sec'],2),'warmup',round(G['TIMING']['compile_warmup_sec'],2))
    assert sel<=G['TIMING']['train_loop_sec']+1e-9
print('BASELINE SMOKE PASSED')
