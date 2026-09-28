# Raw NIDS datasets: provenance and summary

Collected 2026-09-27. Stats were computed with pandas (the CSVs were read as strings). "exact dup" counts rows that are identical across all columns, including IDs, IPs and timestamps where present. MD5 checksums are in `md5.txt` in this folder.

Network note: Kaggle, HuggingFace, Zenodo, UQ eSpace/itee, UNB, figshare and Git LFS (media.githubusercontent.com and the LFS batch API) are all unreachable from this container. GitHub raw/clone, GitHub release assets and **AWS S3** are reachable.

---

## 1. NF-ToN-IoT-v2 (NetFlow v2, 43 features): partial, 20% split
- **Source:** GitHub release asset `https://github.com/awize68/test/releases/download/v0.2-data/NF-ToN-IoT-v2-test.csv` (repo awize68/test, pre-release tag `v0.2-data`, published around 17 Sep 2026; the repo has no documentation). The file seems to match the `NF-ToN-IoT-v2-test.csv` split on HF (Nora9029/NF-ToN-IoT-v2).
- **File:** `NF-ToN-IoT-v2/NF-ToN-IoT-v2-test.csv`, 531,923,548 B, **3,388,100 rows** x 45 cols. There are 0 exact duplicates and no repeated headers.
- **Columns:** IPV4_SRC_ADDR, L4_SRC_PORT, IPV4_DST_ADDR, L4_DST_PORT, PROTOCOL, L7_PROTO, IN_BYTES, IN_PKTS, OUT_BYTES, OUT_PKTS, TCP_FLAGS, CLIENT_TCP_FLAGS, SERVER_TCP_FLAGS, FLOW_DURATION_MILLISECONDS, DURATION_IN, DURATION_OUT, MIN_TTL, MAX_TTL, LONGEST_FLOW_PKT, SHORTEST_FLOW_PKT, MIN_IP_PKT_LEN, MAX_IP_PKT_LEN, SRC_TO_DST_SECOND_BYTES, DST_TO_SRC_SECOND_BYTES, RETRANSMITTED_IN_BYTES, RETRANSMITTED_IN_PKTS, RETRANSMITTED_OUT_BYTES, RETRANSMITTED_OUT_PKTS, SRC_TO_DST_AVG_THROUGHPUT, DST_TO_SRC_AVG_THROUGHPUT, NUM_PKTS_UP_TO_128_BYTES, NUM_PKTS_128_TO_256_BYTES, NUM_PKTS_256_TO_512_BYTES, NUM_PKTS_512_TO_1024_BYTES, NUM_PKTS_1024_TO_1514_BYTES, TCP_WIN_MAX_IN, TCP_WIN_MAX_OUT, ICMP_TYPE, ICMP_IPV4_TYPE, DNS_QUERY_ID, DNS_QUERY_TYPE, DNS_TTL_ANSWER, FTP_COMMAND_RET_CODE, **Label**, **Attack**
- **Labels:** `Label` has 1 = 2,169,107 and 0 = 1,218,993. `Attack` has these counts: Benign 1,218,993; scanning 756,869; xss 491,442; ddos 405,014; password 230,994; dos 142,532; injection 136,609; backdoor 3,364; mitm 1,595; ransomware 688.
- **Notes:** The row count is exactly 20% of the full dataset (16,940,496). The class proportions match the full dataset, which suggests a stratified random 20% sample with shuffled rows. `L7_PROTO` is written as a float (for example `7.0`). The matching train split (80%) could not be found.

## 2. CSE-CIC-IDS2018 (CICFlowMeter, official "Processed Traffic Data for ML Algorithms"): complete
- **Source:** the official AWS Open Data bucket `https://cse-cic-ids2018.s3.ca-central-1.amazonaws.com/Processed%20Traffic%20Data%20for%20ML%20Algorithms/<file>`, downloaded unmodified.
- **Files:** 10 CSVs in `CSE-CIC-IDS2018/`, **16,233,002 rows** in total. Nine files have 80 cols. `Thuesday-20-02-2018` has 84 cols because it adds `Flow ID, Src IP, Src Port, Dst IP`.

| file | bytes | rows | label counts | exact dup |
|---|---|---|---|---|
| Wednesday-14-02-2018 | 358,223,333 | 1,048,575 | Benign 667,626; FTP-BruteForce 193,360; SSH-Bruteforce 187,589 | 225,628 |
| Thursday-15-02-2018 | 375,945,899 | 1,048,575 | Benign 996,077; DoS attacks-GoldenEye 41,508; DoS attacks-Slowloris 10,990 | 2,421 |
| Friday-16-02-2018 | 333,723,605 | 1,048,575 | DoS attacks-Hulk 461,912; Benign 446,772; DoS attacks-SlowHTTPTest 139,890; (1 repeated header row) | 147,586 |
| Thuesday-20-02-2018 | 4,054,925,350 | 7,948,748 | Benign 7,372,557; DDoS attacks-LOIC-HTTP 576,191 | 2 |
| Wednesday-21-02-2018 | 328,893,673 | 1,048,575 | DDOS attack-HOIC 686,012; Benign 360,833; DDOS attack-LOIC-UDP 1,730 | 17,557 |
| Thursday-22-02-2018 | 382,636,202 | 1,048,575 | Benign 1,048,213; Brute Force -Web 249; Brute Force -XSS 79; SQL Injection 34 | 3,278 |
| Friday-23-02-2018 | 382,840,456 | 1,048,575 | Benign 1,048,009; Brute Force -Web 362; Brute Force -XSS 151; SQL Injection 53 | 2,614 |
| Wednesday-28-02-2018 | 209,249,758 | 613,104 | Benign 544,200; Infilteration 68,871; (33 repeated header rows) | 6,121 |
| Thursday-01-03-2018 | 107,842,858 | 331,125 | Benign 238,037; Infilteration 93,063; (25 repeated header rows) | 97 |
| Friday-02-03-2018 | 352,368,373 | 1,048,575 | Benign 762,384; Bot 286,191 | 5,459 |

- **Totals:** Benign 13,484,708; DDOS attack-HOIC 686,012; DDoS attacks-LOIC-HTTP 576,191; DoS attacks-Hulk 461,912; Bot 286,191; FTP-BruteForce 193,360; SSH-Bruteforce 187,589; Infilteration 161,934; DoS attacks-SlowHTTPTest 139,890; DoS attacks-GoldenEye 41,508; DoS attacks-Slowloris 10,990; DDOS attack-LOIC-UDP 1,730; Brute Force -Web 611; Brute Force -XSS 230; SQL Injection 87. There are also 59 repeated header rows with the value "Label".
- **Columns (80):** Dst Port, Protocol, Timestamp, Flow Duration, Tot Fwd Pkts, Tot Bwd Pkts, TotLen Fwd Pkts, TotLen Bwd Pkts, Fwd Pkt Len Max/Min/Mean/Std, Bwd Pkt Len Max/Min/Mean/Std, Flow Byts/s, Flow Pkts/s, Flow IAT Mean/Std/Max/Min, Fwd IAT Tot/Mean/Std/Max/Min, Bwd IAT Tot/Mean/Std/Max/Min, Fwd PSH Flags, Bwd PSH Flags, Fwd URG Flags, Bwd URG Flags, Fwd Header Len, Bwd Header Len, Fwd Pkts/s, Bwd Pkts/s, Pkt Len Min/Max/Mean/Std/Var, FIN/SYN/RST/PSH/ACK/URG Flag Cnt, CWE Flag Count, ECE Flag Cnt, Down/Up Ratio, Pkt Size Avg, Fwd Seg Size Avg, Bwd Seg Size Avg, Fwd Byts/b Avg, Fwd Pkts/b Avg, Fwd Blk Rate Avg, Bwd Byts/b Avg, Bwd Pkts/b Avg, Bwd Blk Rate Avg, Subflow Fwd Pkts/Byts, Subflow Bwd Pkts/Byts, Init Fwd Win Byts, Init Bwd Win Byts, Fwd Act Data Pkts, Fwd Seg Size Min, Active Mean/Std/Max/Min, Idle Mean/Std/Max/Min, **Label**
- **Known quirks, preserved as-is:**
  - Seven files stop at exactly 1,048,575 rows (the Excel row limit), so the official release is truncated there.
  - Some files contain repeated header rows. Drop rows where `Label=="Label"`.
  - `Thuesday-20-02-2018` has 4 extra identifier columns (the misspelled "Thuesday" is the official filename).
  - The data contains Inf/NaN values in `Flow Byts/s` and `Flow Pkts/s` (not checked here, but documented for the original release).
  - There are many exact duplicate rows, especially on 14-02 and 16-02.
  - The column names use the CICFlowMeter-V3 abbreviated style, so they need mapping to match the CIC-IDS2017 names.

## 3. CIC-IDS2017 (CICFlowMeter, GeneratedLabelledFlows variant): subsample, all classes present
- **Source:** `git clone https://github.com/ArnaudRosay/mlp4nids` at commit `5167e608e21ec85362cb88c6515a0b66161fd39b`, path `cicids2017/datasets/ids2017-run-20190910154834/{train_set,test_set,crossval_set}.parquet` (plain git blobs, not LFS).
- **Files:** `CIC-IDS2017/train_set.parquet` (90,549,639 B, 556,548 rows), `test_set.parquet` (46,130,886 B, 278,270 rows) and `crossval_set.parquet` (45,945,066 B, 278,270 rows). That is **1,113,088 rows** in total, each with 85 cols (80 float64 and 5 string).
- **Columns:** the original GeneratedLabelledFlows header, including the leading spaces (for example `' Label'`, `' Source IP'`). The columns are Flow ID, Source IP, Source Port, Destination IP, Destination Port, Protocol, Timestamp, then the 77 CICFlowMeter features (Flow Duration ... Idle Min, including the duplicated `Fwd Header Length.1`), then **Label**.
- **Labels (train / test / crossval):**

| Label | train / test / crossval |
|---|---|
| BENIGN | 278,274 / 139,135 / 139,135 |
| DoS Hulk | 115,062 / 57,531 / 57,531 |
| PortScan | 79,402 / 39,701 / 39,701 |
| DDoS | 64,012 / 32,006 / 32,006 |
| DoS GoldenEye | 5,146 / 2,573 / 2,573 |
| FTP-Patator | 3,967 / 1,983 / 1,983 |
| SSH-Patator | 2,948 / 1,474 / 1,474 |
| DoS slowloris | 2,898 / 1,449 / 1,449 |
| DoS Slowhttptest | 2,749 / 1,374 / 1,374 |
| Bot | 978 / 489 / 489 |
| Web Attack \x96 Brute Force | 753 / 376 / 376 |
| Web Attack \x96 XSS | 326 / 163 / 163 |
| Infiltration | 18 / 9 / 9 |
| Web Attack \x96 Sql Injection | 10 / 5 / 5 |
| Heartbleed | 5 / 2 / 2 |

- **Notes, including preprocessing by the repo author:**
  - Each split is 50% BENIGN and 50% attack. In total there are about 556k benign rows, against about 2.27M in the original. Attack rows look nearly complete: for example DoS Hulk has 230,124 here against 231,073 originally.
  - The splits look stratified 50/25/25. Overlap between them is tiny: 2 to 3 identical rows.
  - Features are already float and the original Label strings are kept. The Web Attack labels contain the cp1252 en-dash byte `\x96`.
  - The Timestamp strings use mixed formats (for example `5/7/2017 10:33` and `03/07/2017 09:49:12`).
  - This is **not** the full MachineLearningCVE set. A full set could not be found: the day-CSVs on GitHub are all Git LFS pointers except `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv`, for example in SHUBHAMRAWAT-25/DDOS-attack-detection-using-random-forest-algorithm and StarterArcher/CICIDS2017. That file was not downloaded.

## 4. UNSW-NB15 official training/testing partitions: complete
- **Source:** `https://raw.githubusercontent.com/shahariar-shibli/Feature-Selection-using-Metaheuristic-Approaches-for-Intrusion-Detection-System/72f6e2019e225104a20f8d53423e743d9d4c39a0/datasets/UNSW_NB15_{training,testing}-set.csv`
- **Files:** `UNSW-NB15/UNSW_NB15_training-set.csv` (32,117,676 B, **175,341 rows**) and `UNSW_NB15_testing-set.csv` (15,298,467 B, **82,332 rows**). Both have 45 cols and a UTF-8 BOM, so the first column reads as `﻿id` unless you pass `encoding='utf-8-sig'`.
- **Columns:** id, dur, proto, service, state, spkts, dpkts, sbytes, dbytes, rate, sttl, dttl, sload, dload, sloss, dloss, sinpkt, dinpkt, sjit, djit, swin, stcpb, dtcpb, dwin, tcprtt, synack, ackdat, smean, dmean, trans_depth, response_body_len, ct_srv_src, ct_state_ttl, ct_dst_ltm, ct_src_dport_ltm, ct_dst_sport_ltm, ct_dst_src_ltm, is_ftp_login, ct_ftp_cmd, ct_flw_http_mthd, ct_src_ltm, ct_srv_dst, is_sm_ips_ports, **attack_cat**, **label**
- **Training labels (`attack_cat`):** Normal 56,000; Generic 40,000; Exploits 33,393; Fuzzers 18,184; DoS 12,264; Reconnaissance 10,491; Analysis 2,000; Backdoor 1,746; Shellcode 1,133; Worms 130. `label` has 1 = 119,341 and 0 = 56,000.
- **Testing labels (`attack_cat`):** Normal 37,000; Generic 18,871; Exploits 11,132; Fuzzers 6,062; DoS 4,089; Reconnaissance 3,496; Analysis 677; Backdoor 583; Shellcode 378; Worms 44. `label` has 1 = 45,332 and 0 = 37,000.
- **Notes:**
  - The counts match the official partitions.
  - Duplicates after excluding `id`: 67,601 in train and 26,387 in test.
  - Some GitHub copies are mislabeled or altered. The `training-set.csv` in ushukkla/nospammers is really the 82,332-row testing set. InitRoot/UNSW_NB15 has swapped file names and no `attack_cat` column. The `testing-set.csv` in ishaak15/UNSW-IDS-Feature-Selection has identical values but different formatting.
  - The full 2.54M-record UNSW-NB15_1..4.csv (jamshaid120/UNSW_NB15-Complete-dataset) is only Git LFS pointers, so it was not obtained.

## 5. ToN-IoT network (Train_Test_Network, Zeek features): benign subsampled
- **Source:** `https://raw.githubusercontent.com/thawatchai2799/SecurityArchitectureSmartCityIoT/98c97ba14f8226158ce8082d33ee5aa0deaf45ef/data/TON_IoT_Train_Test_Network.csv`
- **File:** `ToN-IoT-network/Train_Test_Network.csv`, 29,691,731 B, **211,043 rows** x 44 cols, with a UTF-8 BOM.
- **Columns:** src_ip, src_port, dst_ip, dst_port, proto, service, duration, src_bytes, dst_bytes, conn_state, missed_bytes, src_pkts, src_ip_bytes, dst_pkts, dst_ip_bytes, dns_query, dns_qclass, dns_qtype, dns_rcode, dns_AA, dns_RD, dns_RA, dns_rejected, ssl_version, ssl_cipher, ssl_resumed, ssl_established, ssl_subject, ssl_issuer, http_trans_depth, http_method, http_uri, http_version, http_request_body_len, http_response_body_len, http_status_code, http_user_agent, http_orig_mime_types, http_resp_mime_types, weird_name, weird_addl, weird_notice, **label**, **type**
- **Labels (`type`):** normal 50,000; backdoor, ddos, dos, injection, password, ransomware, scanning and xss 20,000 each; mitm 1,043. `label` has 1 = 161,043 and 0 = 50,000.
- **Notes:**
  - The official Train_Test_Network.csv has 461,043 rows with 300,000 normal. **Only 50,000 normal rows are present here**, so the repo author (or an upstream source) subsampled the normal class. The attack classes are complete.
  - There are 20,569 exact duplicate rows.

## 6. ACI-IoT-2023 (Army Cyber Institute IoT, CICFlowMeter features): complete
- **Source:** GitHub release asset `https://github.com/awize68/test/releases/download/v0.1-data/ACI-IoT-2023.csv` (tag `v0.1-data`, pre-release, around 15 Sep 2026).
- **File:** `ACI-IoT-2023/ACI-IoT-2023.csv`, 625,901,428 B, **1,231,411 rows** x 85 cols.
- **Columns:** CICFlowMeter-V4 naming: Flow ID, Src IP, Src Port, Dst IP, Dst Port, Protocol, Timestamp, Flow Duration, Total Fwd Packet, Total Bwd packets, ... , Idle Min, **Label**, **Connection Type**. Apart from the naming, the features are the same as CIC-IDS2017/2018.
- **Labels (`Label`):** Port Scan 441,282; Benign 329,295; ICMP Flood 225,234; Ping Sweep 71,928; DNS Flood 46,935; Vulnerability Scan 39,537; OS Scan 37,524; Slowloris 18,643; SYN Flood 13,857; Dictionary Attack 6,380; UDP Flood 791; ARP Spoofing 5. `Connection Type` has wireless 742,758 and wired 488,653.
- **Notes:** There are 56,189 exact duplicate rows. There is no binary label column.

---

## Not obtained
- **NF-UNSW-NB15(-v2), NF-BoT-IoT(-v2), NF-CSE-CIC-IDS2018(-v2), NF-ToN-IoT v1, NF-UQ-NIDS(-v2), NetFlow v3:**
  - The only hosts are Kaggle, HF, Zenodo and UQ eSpace, and all of them are blocked.
  - The only GitHub copy found, zih19/Network-Intrusion-Detection `NF-UQ-NIDS-v2.csv`, is a 772 MB Git LFS pointer. The LFS batch API is denied by the git proxy.
  - About 15 searches plus repo listings found no other non-LFS copies or release assets.
- **NF-ToN-IoT-v2 train split (80%):** not found. Only the test split above is available.
- **Full CIC-IDS2017 MachineLearningCVE:** only LFS pointers were found. The subsample above was used instead.
- **CIC-DDoS2019, CIC-IoT2023:** not searched in depth.
