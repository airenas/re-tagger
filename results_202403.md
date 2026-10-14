# Results

On CONLLU_24_01_30.zip test set


| No   | Method                                                                                | Acc          | Err count (on test) | Err count (ignore some non critical tts errors) | Date        | Test details             |
|------|---------------------------------------------------------------------------------------|--------------|-|----------|-------------|--------------------------|
| o2   | Oracle                                                                                | `0.99235`    | `338`     | 
| f2   | Fotonija morph                                                                        | `0.95808`    | `1854`    |
|      |
|      | **CRF** *context: 2 words before, 2 after*                                            
| c2   | Word features                                                                         | `0.97411`    | `1145`    |
|      | **BiLSTM-CRF**                                                                        | 
| l2   | embedding (e0), 300 hidden neurons                                | `0.97750`    | `995`     | |            | data_v5.3                
| l21  | embedding (e1), crf(bilstm(300)), cuddn compatible lstm          | `0.97773`    | `985`     | | 2024-11-30  | data_v6, model tm-0.6    |
| l22  | embedding (e0), crf(lstm(300,bilstm(300))), cuddn compatible lstm | `0.97915` | `922`   | | 2024-12-02  | data_v72, model tm-0.7   |
| l23  | embedding (e1), crf(lstm(300,bilstm(300))), cuddn compatible lstm | `0.97940` | `911`  | | 2024-12-04 | data_v73, model-vol v73, (finalfusion format) | 
| l24  | embedding (e1), crf(lstm(300,bilstm(300))), cuddn compatible lstm | `0.97734` | `1002`  | | 2024-12-04 | data_v74 | 
| [l080](#l080) | embedding (e1), crf(lstm(300,bilstm(300)) + dropout), cuddn compatible lstm | `0.98260` | `769`  | `403` | 2026-10-14 | data_v071 | 
| [l082](#l082) | embedding (e2), crf(lstm(300,bilstm(300)) + dropout), cuddn compatible lstm | `0.98346` | `731`  | `357` | 2026-10-14 | data_v071 | 
| [l088](#l088) | embedding (e3), crf(lstm(300,bilstm(300)) + dropout), cuddn compatible lstm | `0.98350` | `729`  | `376` | 2026-10-14 | data_v071 | 
|      | **Transformer**                                                                        | 
| [t010](#t010)  | [./egs/transformer](./egs/transformer) https://huggingface.co/VSSA-SDSA/LT-MLKM-modernBERT + MLP | `0.984728` | `675`  | `323` | 2026-10-14 | data_v071 | 

## Embeddings

| No   | Corpus  | Type | Dim | Using through | Vocab | Info | Text cleaning |
|------|---|-|-|-|-|-|-|
| e0   | Delfi | fasttext cbow | 150 | fasttext | 890k | training?? from Darius |
| e1   | Delfi | fasttext cbow | 150 | finalfusion (normalized) | 890k | training?? from Darius |
| e2   | https://huggingface.co/datasets/VSSA-SDSA/LT_AI_BLKT | fasttext cbow | 150 | finalfusion (normalized) | 2522k | [./egs/fasttext](./egs/fasttext): bucket=5000000, min_count=10, sampling=0.0001  | `Kaunas, 1934.3.27 NAŠLAITIS PABODO man šis tuščias namas Ir ilgos rūsčios dienos.` |
| e3   | https://huggingface.co/datasets/VSSA-SDSA/LT_AI_BLKT | fasttext cbow | 150 | finalfusion (normalized) | 1701k | [./egs/fasttext](./egs/fasttext): bucket=5000000, min_count=10, sampling=0.0001  | `Kaunas , 1934 . 3 . 27 NAŠLAITIS PABODO man šis tuščias namas Ir ilgos rūsčios dienos .`
 |




## Top errors

### f2

```txt
    184 Ncfpnn-	Ncfsgn-
     84 Ncfsin-	Ncfsnn-
     70 Vgps-pmpnnnn-p	Vgi-----n--n--
     61 Pgfpgn	Pgmpgn
     57 Agpfpnn	Agpfsgn
     52 Np---n-	X-
     45 Agpfpgn	Agpmpgn
     35 Qg	Cg
     31 Npmsnn-	X-
     30 Y-	X-
     30 Pgfpnn	Pgfsgn
     29 Vgps--npnn-n-p	Vgps-sfpnnnn-p
     29 Vgpp-sfpnnnn-p	Vgpp--npnn-n-p
     25 Ncfsgn-	Ncfpnn-
     23 Pgn--n	Cg
     21 Rgc	Rgp
     19 Sg-	Rgp
     19 Agpmpgn	Agpfpgn
     17 Npmsgn-	X-
     16 Vgps-sfpnngn-p	Vgps-pfpnnnn-p
```

### c2
```txt
     59 Ncfpnn-	Ncfsgn-
     52 Np---n-	X-
     52 Ncfsin-	Ncfsnn-
     43 Ncfsgn-	Ncfpnn-
     37 Agpn--n	Agpfsnn
     32 Pgfpgn	Pgmpgn
     31 Npmsnn-	X-
     30 Y-	X-
     27 Vgps-pmpnnnn-p	Vgi-----n--n--
     19 Agpfpnn	Agpfsgn
     18 Pgmpgn	Pgfpgn
     17 Vgpp-sfpnnnn-p	Vgpp--npnn-n-p
     17 Npmsgn-	X-
     15 Agpfsgn	Agpfpnn
     14 Agpfpgn	Agpmpgn
     13 Vgpp--npnn-n-p	Vgpp-sfpnnnn-p
     11 Vgps--npnn-n-p	Vgps-sfpnnnn-p
     11 Qg	Cg
     10 Vgps-sfpnngn-p	Vgps-pfpnnnn-p
     10 Qg	Rgp
```

### l2
```txt
     57 Ncfsgn-	Ncfpnn-
     52 Np---n-	X-
     41 Pgmpgn	Pgfpgn
     35 Agpn--n	Agpfsnn
     33 Ncfsin-	Ncfsnn-
     31 Npmsnn-	X-
     30 Y-	X-
     26 Ncfpnn-	Ncfsgn-
     17 Vgps-pmpnnnn-p	Vgi-----n--n--
     17 Npmsgn-	X-
     12 Pgfpgn	Pgmpgn
     12 Agpfpnn	Agpfsgn
     10 Vgpp-sfpnnnn-p	Vgpp--npnn-n-p
     10 Pgfsgn	Pgfpnn
      9 Qg	Rgp
      9 Ncfsnn-	Ncfsin-
      9 Agsn--n	Agsfsnn
      8 Vgps--npnn-n-p	Vgps-sfpnnnn-p
      8 Ncmsnn-	X-
      8 Ncmpnn-	X-
      7 Y-	Ig
```

### l21
```txt
     51 Np---n-	X-
     48 Ncfsgn-	Ncfpnn-
     39 Pgmpgn	Pgfpgn
     33 Ncfsin-	Ncfsnn-
     31 Npmsnn-	X-
     31 Ncfpnn-	Ncfsgn-
     30 Y-	X-
     16 Vgps-pmpnnnn-p	Vgi-----n--n--
     16 Pgfpgn	Pgmpgn
     16 Npmsgn-	X-
     13 Rgp	Cg
     13 Agpfpnn	Agpfsgn
     11 Ncfsnn-	Ncfsin-
      9 Vgpp-sfpnnnn-p	Vgpp--npnn-n-p
      9 Vgi-----n--n--	Vgps-pmpnnnn-p
      8 Ncmsnn-	X-
      8 Ncmpnn-	X-
      7 Y-	Ig
      7 Vgmp3---n--ni-	Ncmsgn-
      7 Qg	Rgp
      7 Cg	Qg
```

### l22
```txt
     51 Np---n-	X-
     38 Ncfpnn-	Ncfsgn-
     35 Ncfsgn-	Ncfpnn-
     33 Pgmpgn	Pgfpgn
     31 Npmsnn-	X-
     30 Y-	X-
     29 Ncfsin-	Ncfsnn-
     16 Npmsgn-	X-
     13 Vgpp-sfpnnnn-p	Vgpp--npnn-n-p
     12 Vgps-pmpnnnn-p	Vgi-----n--n--
     12 Pgfpgn	Pgmpgn
     12 Agpfpnn	Agpfsgn
     11 Agpfsgn	Agpfpnn
      9 Rgp	Cg
      8 Vgi-----n--n--	Vgps-pmpnnnn-p
      8 Ncmsnn-	X-
      8 Ncmpnn-	X-
      7 Y-	Ig
      7 Vgpp--npnn-n-p	Vgpp-sfpnnnn-p
      7 Ncfsnn-	Ncfsin-
      7 Agpfpgn	Agpmpgn
```

### l23
```txt
     51 Np---n-	X-
     40 Ncfsgn-	Ncfpnn-
     38 Pgmpgn	Pgfpgn
     38 Ncfsin-	Ncfsnn-
     31 Npmsnn-	X-
     30 Y-	X-
     26 Ncfpnn-	Ncfsgn-
     17 Vgps-pmpnnnn-p	Vgi-----n--n--
     16 Npmsgn-	X-
     12 Vgpp-sfpnnnn-p	Vgpp--npnn-n-p
     12 Agpfpnn	Agpfsgn
     11 Rgp	Cg
     10 Ncfsnn-	Ncfsin-
     10 Agpfpgn	Agpmpgn
      8 Pgfpgn	Pgmpgn
      8 Ncmsnn-	X-
      8 Ncmpnn-	X-
      7 Y-	Ig
      7 Qg	Rgp
      6 X-	Ig
      6 Vgmp3---n--ni-	Ncmsgn-
```

### l24
```txt
     51 Np---n-	X-
     35 Ncfsgn-	Ncfpnn-
     31 Npmsnn-	X-
     30 Y-	X-
     29 Ncfsin-	Ncfsnn-
     23 Ncfpnn-	Ncfsgn-
     20 Pgmpgn	Pgfpgn
     19 Agpfpgn	Agpmpgn
     17 Vgps-pmpnnnn-p	Vgi-----n--n--
     17 Pgfpgn	Pgmpgn
     16 Npmsgn-	X-
     16 Ncfsnn-	Ncfsin-
     14 Vgpp-sfpnnnn-p	Vgpp--npnn-n-p
     11 Agpmpgn	Agpfpgn
      9 Agpfpnn	Agpfsgn
      8 Rgp	Cg
      8 Ncmsnn-	X-
      8 Ncmpnn-	X-
      8 Cg	Rgp
      7 Y-	Ig
      6 X-	Ig
      6 Vgpp-sfanynn-p	Vgpp-sfanyvn-p
      6 Vgpp--npnn-n-p	Vgpp-sfpnnnn-p
      6 Qg	Rgp
      6 Pgfsgn	Pgfpnn
      6 Pgfpnn	Pgfsgn
```

### l080
Important

```txt
     23 Ncfpnn-	Ncfsgn-
     22 Ncfsin-	Ncfsnn-
     20 Ncfsnn-	Ncfsin-
     17 Ncfsgn-	Ncfpnn-
     10 Vgps-pmpnnnn-p	Vgi-----n--n--
      9 Vgpp-sfpnnnn-p	Vgpp--npnn-n-p
      6 Vgmp3---n--ni-	Ncmsgn-
      6 Np---n-	Npc--n-
      5 Pgmdnn	Mcm-vl-
      5 Pgfsgn	Pgfpnn
      5 Npmsgn-	Ncmsgn-
      5 Ncfpgn-	Ncmpgn-
      5 Agpfpnn	Agpfsgn
      4 Vgps-sfpnnnn-p	Vgps--npnn-n-p
      4 Vgpp-sfpnngn-p	Vgpp-pfpnnnn-p
      4 Vgpp--npnn-n-p	Vgpp-sfpnnnn-p

```

### l082
Important

```txt
     22 Ncfpnn-	Ncfsgn-
     16 Ncfsgn-	Ncfpnn-
     13 Ncfsnn-	Ncfsin-
     13 Ncfsin-	Ncfsnn-
      8 Vgps-pmpnnnn-p	Vgi-----n--n--
      7 Vgpp-sfpnnnn-p	Vgpp--npnn-n-p
      7 Vgi-----n--n--	Vgps-pmpnnnn-p
      7 Agpfpnn	Agpfsgn
      6 Np---n-	Npc--n-
      6 Ncfpgn-	Ncmpgn-
      5 Vgps--npnn-n-p	Vgps-sfpnnnn-p
      5 Vgmp3---n--ni-	Ncmsgn-
      5 Pgn--n	Cg
      5 Pgmdnn	Mcm-vl-
      5 Pgfsgn	Pgfpnn
      5 Npmsgn-	Ncmsgn-
```

### l088
Important

```txt
     24 Ncfsgn-	Ncfpnn-
     23 Ncfpnn-	Ncfsgn-
     21 Ncfsnn-	Ncfsin-
     12 Ncfsin-	Ncfsnn-
     11 Vgpp-sfpnnnn-p	Vgpp--npnn-n-p
      9 Vgps-pmpnnnn-p	Vgi-----n--n--
      6 Vgpp-sfpnngn-p	Vgpp-pfpnnnn-p
      6 Vgmp3---n--ni-	Ncmsgn-
      6 Np---n-	Npc--n-
      6 Agpfpnn	Agpfsgn
      5 Pgmdnn	Mcm-vl-
      5 Npmsgn-	Ncmsgn-
      5 Ncfpgn-	Ncmpgn-
      4 Vgi-----n--n--	Vgps-pmpnnnn-p
      4 Pg--nn	Qg
      4 Pgfsgn	Pgfpnn
```

### t010
Important

```txt
     16 Ncfsnn-	Ncfsin-
     16 Ncfpnn-	Ncfsgn-
     15 Ncfsgn-	Ncfpnn-
     10 Ncfsin-	Ncfsnn-
      7 Vgpp-sfpnnnn-p	Vgpp--npnn-n-p
      6 Vgps-pmpnnnn-p	Vgi-----n--n--
      6 Np---n-	Npc--n-
      6 Agpfpnn	Agpfsgn
      5 Vgi-----n--n--	Vgps-pmpnnnn-p
      5 Pgmdnn	Mcm-vl-
      5 Npmsgn-	Ncmsgn-
      5 Agpmpnn	Agpmpvn
      4 Vgpp--npnn-n-p	Vgpp-sfpnnnn-p
      4 Npmsnn-	Ncmsnn-
      4 Npfsnn-	Ncfsnn-
      4 Ncmpgn-	Vgpp-pmpnygn-p

```