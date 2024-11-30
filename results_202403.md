# Results

On CONLLU_24_01_30.zip test set


| No | Method                                                                        | Acc | Err count | Date       | Test details     |
|-|-------------------------------------------------------------------------------|-|-----------|------------|-----------------------|
| o2 | Oracle                                                                        | `0.99235` | `338`     |
| f2 | Fotonija morph                                                                | `0.95808` | `1854`    |
||
| | **CRF** *context: 2 words before, 2 after*                                    
| c2 | Word features                                                                 | `0.97411` | `1145`    |
| | **BiLSTM-CRF**                                                                | 
| l2 | embedding pretrained fasttext cbow, 300 hidden neurons                        | `0.97750` | `995`     | | data_v5.3             
| l21 | embedding pretrained fasttext cbow, 300 hidden neurons, cuddn compatible lstm | `0.97773` | `985`    | 2024-11-30 | data_v6, model tm-0.6 | 


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
