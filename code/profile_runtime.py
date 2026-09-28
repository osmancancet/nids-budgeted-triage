import time, numpy as np, run_experiments as R
df,feats=R.load('CSE-CIC-IDS2018'); X,y=df[feats].to_numpy(np.float32),df.label.to_numpy(); pt=R.split(df,0)
for m in ['LGBM','RF','LR','MLP']:
    t=time.time(); F=R.Fitted(m,0,X[pt==0],y[pt==0],X[pt==1],y[pt==1]); t1=time.time()-t
    t=time.time(); d,n=F.scores(X[pt==2]); t2=time.time()-t
    t=time.time(); R.detection_metrics(d,F.cals,y[pt==2],F.tau); t3=time.time()-t
    print(m,'fit %.1f score %.1f metrics %.1f'%(t1,t2,t3),flush=True)
