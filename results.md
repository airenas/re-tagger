# Results

On MATAS v1.0 test set


| No | Method| Acc | Err count | Acc no punct | Err count no punct |
|-|---|-|-|-|-|
| f1 | Fotonija morph | `0.95338` | `2062` | `0.94181` | `2060` |
||
| | **CRF** *context: 2 words before, 2 after*
| c1 | Word features | `0.96444` | `1573` | `0.95556` | `1573` |
| c2 | With morh features added | `0.96521` | `1539` | `0.95652` | `1539` |
||
| | **CRF** *trained without punctuation*
| c3 | Word features | | | `0.95426` | `1619` |
| c4 | With morh features added | || `0.95599` | `1558` |
||
| | **BiLSTM-CRF** | 
| l1 | embedding pretrained fasttext cbow, 300 hidden neurons | `0.96785` | `1422` | `0.95983` | `1422` |

## Top errors

### f1

```txt
    181 Ncfpnn-	Ncfsgn-
    150 Qg	Cg
     84 Ncfsin-	Ncfsnn-
     69 Vgps-pmpnnnn-p	Vgi-----n--n--
     61 Pgfpgn	Pgmpgn
     55 Agpfpnn	Agpfsgn
     53 Xf	X-
     45 Agpfpgn	Agpmpgn
     37 Np---ns	X-
     30 Vgps--npnn-n-p	Vgps-sfpnnnn-p
     30 Pgfpnn	Pgfsgn
     27 Vgpp-sfpnnnn-p	Vgpp--npnn-n-p
     26 Ncfsgn-	Ncfpnn-
     24 Npmsnns	X-
     23 Pgn--n	Cg
     21 Rgc	Rgp
     19 Agpmpgn	Agpfpgn
     17 Ya	X-
     16 Vgps-sfpnngn-p	Vgps-pfpnnnn-p
     16 Sga	Rgp
```

### c1
```txt
     92 Sga	Sgg
     61 Qg	Cg
     60 Rgc	Rgp
     53 Xf	X-
     52 Ncfsgn-	Ncfpnn-
     50 Ncfpnn-	Ncfsgn-
     42 Npmsgng	Npmsgns
     42 Ncfsin-	Ncfsnn-
     41 Npmsnnf	Npmsnns
     39 Agpn--n	Agpfsnn
     37 Np---ns	X-
     35 Cg	Qg
     32 Pgfpgn	Pgmpgn
     25 Vgps-pmpnnnn-p	Vgi-----n--n--
     24 Npmsnns	X-
     24 Agpfpnn	Agpfsgn
     21 Npmsgnf	Npmsgns
     19 Agpfpgn	Agpmpgn
     18 Agpfsgn	Agpfpnn
     17 Ya	X-
```

### l1
```txt
     92 Sga	Sgg
     60 Rgc	Rgp
     53 Xf	X-
     44 Ncfsin-	Ncfsnn-
     44 Ncfpnn-	Ncfsgn-
     42 Npmsgng	Npmsgns
     41 Npmsnnf	Npmsnns
     41 Ncfsgn-	Ncfpnn-
     40 Agpn--n	Agpfsnn
     37 Np---ns	X-
     32 Pgmpgn	Pgfpgn
     30 Qg	Cg
     29 Cg	Qg
     24 Npmsnns	X-
     21 Npmsgnf	Npmsgns
     17 Ya	X-
     17 Vgps-pmpnnnn-p	Vgi-----n--n--
     17 Pgfpgn	Pgmpgn
     17 Npmsnng	Npmsnns
     12 Agpfpgn	Agpmpgn
```