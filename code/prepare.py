"""Harmonise raw NIDS datasets into analysis-ready parquet files.

Output: data/processed/<name>.parquet with float32 feature columns plus
  label  (0 benign / 1 attack), family (original attack label),
  group  (harmonised attack group, CIC-family datasets only).

Steps per dataset: drop identifiers (flow id, IPs, ports, timestamps),
replace +-inf with NaN and NaN with 0, remove exact duplicates over
features+family, then cap benign rows at BENIGN_CAP and each attack family at
FAMILY_CAP by uniform random sampling (seed 0).  Counts before/after are
written to results/data_summary.csv.
"""
import os
import sys
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")
OUT = os.path.join(ROOT, "data", "processed")
os.makedirs(OUT, exist_ok=True)
BENIGN_CAP, FAMILY_CAP, SEED = 200_000, 30_000, 0

# 2018 naming (CICFlowMeter-V3) is canonical for the shared CIC feature space.
CIC_FEATURES = [
    'Flow Duration', 'Tot Fwd Pkts', 'Tot Bwd Pkts', 'TotLen Fwd Pkts', 'TotLen Bwd Pkts',
    'Fwd Pkt Len Max', 'Fwd Pkt Len Min', 'Fwd Pkt Len Mean', 'Fwd Pkt Len Std',
    'Bwd Pkt Len Max', 'Bwd Pkt Len Min', 'Bwd Pkt Len Mean', 'Bwd Pkt Len Std',
    'Flow Byts/s', 'Flow Pkts/s', 'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min',
    'Fwd IAT Tot', 'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min',
    'Bwd IAT Tot', 'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min',
    'Fwd PSH Flags', 'Bwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags', 'Fwd Header Len',
    'Bwd Header Len', 'Fwd Pkts/s', 'Bwd Pkts/s', 'Pkt Len Min', 'Pkt Len Max', 'Pkt Len Mean',
    'Pkt Len Std', 'Pkt Len Var', 'FIN Flag Cnt', 'SYN Flag Cnt', 'RST Flag Cnt', 'PSH Flag Cnt',
    'ACK Flag Cnt', 'URG Flag Cnt', 'CWE Flag Count', 'ECE Flag Cnt', 'Down/Up Ratio',
    'Pkt Size Avg', 'Fwd Seg Size Avg', 'Bwd Seg Size Avg', 'Fwd Byts/b Avg', 'Fwd Pkts/b Avg',
    'Fwd Blk Rate Avg', 'Bwd Byts/b Avg', 'Bwd Pkts/b Avg', 'Bwd Blk Rate Avg',
    'Subflow Fwd Pkts', 'Subflow Fwd Byts', 'Subflow Bwd Pkts', 'Subflow Bwd Byts',
    'Init Fwd Win Byts', 'Init Bwd Win Byts', 'Fwd Act Data Pkts', 'Fwd Seg Size Min',
    'Active Mean', 'Active Std', 'Active Max', 'Active Min', 'Idle Mean', 'Idle Std',
    'Idle Max', 'Idle Min']
assert len(CIC_FEATURES) == 76

GROUPS = {
    # CIC-IDS2017
    'BENIGN': 'Benign', 'DoS Hulk': 'DoS', 'DoS GoldenEye': 'DoS', 'DoS slowloris': 'DoS',
    'DoS Slowhttptest': 'DoS', 'DDoS': 'DDoS', 'PortScan': 'Scan', 'FTP-Patator': 'BruteForce',
    'SSH-Patator': 'BruteForce', 'Web Attack - Brute Force': 'WebAttack',
    'Web Attack - XSS': 'WebAttack', 'Web Attack - Sql Injection': 'WebAttack', 'Bot': 'Botnet',
    'Infiltration': 'Infiltration', 'Heartbleed': 'Other',
    # CSE-CIC-IDS2018
    'Benign': 'Benign', 'DoS attacks-Hulk': 'DoS', 'DoS attacks-GoldenEye': 'DoS',
    'DoS attacks-Slowloris': 'DoS', 'DoS attacks-SlowHTTPTest': 'DoS',
    'DDOS attack-HOIC': 'DDoS', 'DDoS attacks-LOIC-HTTP': 'DDoS', 'DDOS attack-LOIC-UDP': 'DDoS',
    'FTP-BruteForce': 'BruteForce', 'SSH-Bruteforce': 'BruteForce',
    'Brute Force -Web': 'WebAttack', 'Brute Force -XSS': 'WebAttack',
    'SQL Injection': 'WebAttack', 'Infilteration': 'Infiltration',
    # ACI-IoT-2023
    'Port Scan': 'Scan', 'OS Scan': 'Scan', 'Vulnerability Scan': 'Scan', 'Ping Sweep': 'Scan',
    'ICMP Flood': 'DoS', 'SYN Flood': 'DoS', 'UDP Flood': 'DoS', 'DNS Flood': 'DoS',
    'Slowloris': 'DoS', 'Dictionary Attack': 'BruteForce', 'ARP Spoofing': 'Other',
}


def _norm(s):
    return ''.join(ch for ch in s.lower() if ch.isalnum())


SYN = {'tot': 'total', 'totlen': 'total length', 'pkt': 'packet', 'pkts': 'packet',
       'packets': 'packet', 'len': 'length', 'byts': 'byte', 'bytes': 'byte', 'b': 'bulk',
       'blk': 'bulk', 'cnt': 'count', 'seg': 'segment', 'avg': 'average', 'fwd': 'forward',
       'bwd': 'backward', 'var': 'variance', 'cwr': 'cwe', 'act': 'active', 'of': ''}


def _canon(s):
    toks = s.lower().replace('/', ' ').replace('_', ' ').split()
    out = []
    for t in toks:
        out += SYN.get(t, t).split()
    return sorted(out)


def check_alignment(cols):
    """Positional mapping sanity check: same bag of canonical name tokens."""
    for a, b in zip(cols, CIC_FEATURES):
        assert _canon(a) == _canon(b), (a, b)


def finish(df, name, feats, stats):
    X = df[feats].apply(pd.to_numeric, errors='coerce').astype(np.float64)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    X = X.astype(np.float32)
    X['family'] = df['family'].astype(str).values
    df = X
    n0 = len(df)
    df = df.drop_duplicates()
    n1 = len(df)
    rng = np.random.default_rng(SEED)
    parts = []
    for fam, g in df.groupby('family'):
        cap = BENIGN_CAP if GROUPS.get(fam, fam) in ('Benign',) or fam.lower() in ('benign', 'normal') else FAMILY_CAP
        if len(g) > cap:
            g = g.iloc[rng.choice(len(g), cap, replace=False)]
        parts.append(g)
    df = pd.concat(parts).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    df['label'] = (~df['family'].str.lower().isin(['benign', 'normal'])).astype(np.int8)
    df['group'] = df['family'].map(GROUPS).fillna(df['family'])
    df.to_parquet(os.path.join(OUT, f'{name}.parquet'), index=False)
    for fam, n in df['family'].value_counts().items():
        stats.append(dict(dataset=name, family=fam, group=GROUPS.get(fam, fam), n_final=n))
    stats.append(dict(dataset=name, family='__rows_raw__', n_final=n0))
    stats.append(dict(dataset=name, family='__rows_dedup__', n_final=n1))
    print(name, 'raw', n0, 'dedup', n1, 'final', len(df), flush=True)


def precap(df, rng, benign_cap=3 * BENIGN_CAP, fam_cap=3 * FAMILY_CAP):
    """Memory guard while streaming large files (final caps applied later)."""
    parts = []
    for fam, g in df.groupby('family'):
        cap = benign_cap if fam.lower() in ('benign',) else fam_cap
        if len(g) > cap:
            g = g.iloc[rng.choice(len(g), cap, replace=False)]
        parts.append(g)
    return pd.concat(parts)


def cic2017(stats):
    d = os.path.join(RAW, 'CIC-IDS2017')
    df = pd.concat([pd.read_parquet(os.path.join(d, f)) for f in
                    ('train_set.parquet', 'crossval_set.parquet', 'test_set.parquet')])
    df.columns = [c.strip() for c in df.columns]
    df = df.drop(columns=['Fwd Header Length.1'])
    cols = df.columns[df.columns.get_loc('Flow Duration'):df.columns.get_loc('Idle Min') + 1]
    check_alignment(list(cols))
    df = df.rename(columns=dict(zip(cols, CIC_FEATURES)))
    df['family'] = df['Label'].str.replace('\x96', '-', regex=False).str.replace('–', '-', regex=False)
    finish(df, 'CIC-IDS2017', CIC_FEATURES + ['Protocol'], stats)


def cic2018(stats):
    """Streamed: benign rows are Bernoulli-subsampled at 5% per chunk (13.5M benign
    rows in total) and features converted to float32 immediately to bound memory."""
    d = os.path.join(RAW, 'CSE-CIC-IDS2018')
    rng = np.random.default_rng(SEED)
    parts = []
    for f in sorted(os.listdir(d)):
        if not f.endswith('.csv'):
            continue
        for ch in pd.read_csv(os.path.join(d, f), chunksize=250_000, dtype=str,
                              usecols=CIC_FEATURES + ['Protocol', 'Label']):
            ch = ch[ch['Label'] != 'Label']
            keep = (ch['Label'] != 'Benign').to_numpy() | (rng.random(len(ch)) < 0.05)
            ch = ch[keep]
            X = ch[CIC_FEATURES + ['Protocol']].apply(pd.to_numeric, errors='coerce').astype(np.float32)
            X['family'] = ch['Label'].to_numpy()
            parts.append(X)
        print('read', f, flush=True)
    df = pd.concat(parts, ignore_index=True)
    finish(df, 'CSE-CIC-IDS2018', CIC_FEATURES + ['Protocol'], stats)


def aci(stats):
    df = pd.read_csv(os.path.join(RAW, 'ACI-IoT-2023', 'ACI-IoT-2023.csv'), low_memory=False)
    cols = df.columns[df.columns.get_loc('Flow Duration'):df.columns.get_loc('Idle Min') + 1]
    check_alignment(list(cols))
    df = df.rename(columns=dict(zip(cols, CIC_FEATURES)))
    df['family'] = df['Label']
    finish(df, 'ACI-IoT-2023', CIC_FEATURES + ['Protocol'], stats)


def onehot_topk(df, cols, k=15):
    out = []
    for c in cols:
        v = df[c].astype(str)
        top = v.value_counts().index[:k]
        for t in top:
            out.append(pd.Series((v == t).astype(np.float32), name=f'{c}={t}'))
    return pd.concat(out, axis=1)


def unsw(stats):
    d = os.path.join(RAW, 'UNSW-NB15')
    df = pd.concat([pd.read_csv(os.path.join(d, f), encoding='utf-8-sig') for f in
                    ('UNSW_NB15_training-set.csv', 'UNSW_NB15_testing-set.csv')], ignore_index=True)
    cat = ['proto', 'service', 'state']
    num = [c for c in df.columns if c not in cat + ['id', 'attack_cat', 'label']]
    oh = onehot_topk(df, cat)
    df = pd.concat([df[num].reset_index(drop=True), oh], axis=1).assign(family=df['attack_cat'].values)
    finish(df, 'UNSW-NB15', num + list(oh.columns), stats)


def ton(stats):
    df = pd.read_csv(os.path.join(RAW, 'ToN-IoT-network', 'Train_Test_Network.csv'),
                     encoding='utf-8-sig', low_memory=False)
    drop = ['src_ip', 'dst_ip', 'src_port', 'dst_port', 'dns_query', 'ssl_subject', 'ssl_issuer',
            'http_uri', 'http_user_agent', 'weird_addl', 'label', 'type']
    rest = [c for c in df.columns if c not in drop]
    num, cat = [], []
    for c in rest:
        (num if pd.to_numeric(df[c], errors='coerce').notna().mean() > 0.99 else cat).append(c)
    oh = onehot_topk(df, cat)
    df2 = pd.concat([df[num].apply(pd.to_numeric, errors='coerce').reset_index(drop=True), oh], axis=1)
    df2['family'] = df['type'].values
    finish(df2, 'ToN-IoT', num + list(oh.columns), stats)


if __name__ == '__main__':
    which = sys.argv[1:] or ['cic2017', 'aci', 'unsw', 'ton', 'cic2018']
    stats = []
    for w in which:
        globals()[w](stats)
        pd.DataFrame(stats).to_csv(os.path.join(ROOT, 'results', f'data_summary_{w}.csv'), index=False)
        stats = []
