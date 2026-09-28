import numpy as np, run_experiments as R
from nidslib import *
from sklearn.ensemble import IsolationForest
for name,fams in [('ACI-IoT-2023',['Ping Sweep','SYN Flood','Dictionary Attack','Slowloris']),('UNSW-NB15',['Fuzzers','Reconnaissance','Shellcode']),('CIC-IDS2017',['Bot','SSH-Patator','Web Attack - XSS'])]:
    df,feats=R.load(name); X,y,fam=df[feats].to_numpy(np.float32),df.label.to_numpy(),df.family.to_numpy(); pt=R.split(df,0)
    for f in fams:
        keep=fam!=f; tr=(pt==0)&keep; te=pt==2
        m=te&((y==0)|(fam==f)); yy=y[m]
        a=KNNNovelty().fit(X[tr]).score(X[m]); b=KNNNovelty().fit(X[tr&(y==0)]).score(X[m])
        sc=LogScaler().fit(X[tr&(y==0)]); iso=IsolationForest(n_estimators=200,random_state=0,n_jobs=4).fit(sc.transform(X[tr&(y==0)]))
        c=-iso.score_samples(sc.transform(X[m]))
        print(name,f,'all-ref %.3f benign-ref %.3f iforest-benign %.3f'%(safe_auc(yy,a),safe_auc(yy,b),safe_auc(yy,c)),flush=True)
