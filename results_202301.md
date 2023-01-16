# Results

On MATAS v1.0 test set


| No | Method| Acc | Err count  |
|-|---|-|-|
| o2 | Oracle | `0.99235` | `338` |
| f2 | Fotonija morph | `0.95813` | `1852` |
||
| | **CRF** *context: 2 words before, 2 after*
| c2 | Word features | `0.97409` | `1146` |
| | **BiLSTM-CRF** | 
| l2 (en_04) | embedding pretrained fasttext cbow, 300 hidden neurons | `0.97635` | `1046` |

## Top errors

### f2

```txt
    181 Ncfpnn-	Ncfsgn-
     84 Ncfsin-	Ncfsnn-
     69 Vgps-pmpnnnn-p	Vgi-----n--n--
     61 Pgfpgn	Pgmpgn
     55 Agpfpnn	Agpfsgn
     52 Np---n-	X-
     45 Agpfpgn	Agpmpgn
     35 Qg	Cg
     31 Npmsnn-	X-
     30 Y-	X-
     30 Vgps--npnn-n-p	Vgps-sfpnnnn-p
     30 Pgfpnn	Pgfsgn
     27 Vgpp-sfpnnnn-p	Vgpp--npnn-n-p
     26 Ncfsgn-	Ncfpnn-
     23 Pgn--n	Cg
     21 Rgc	Rgp
     19 Sg-	Rgp
     19 Agpmpgn	Agpfpgn
     17 Npmsgn-	X-
     16 Vgps-sfpnngn-p	Vgps-pfpnnnn-p
```

### c2
```txt
     52 Np---n-	X-
     52 Ncfsgn-	Ncfpnn-
     50 Ncfpnn-	Ncfsgn-
     42 Ncfsin-	Ncfsnn-
     37 Agpn--n	Agpfsnn
     32 Pgfpgn	Pgmpgn
     31 Npmsnn-	X-
     30 Y-	X-
     25 Vgps-pmpnnnn-p	Vgi-----n--n--
     18 Agpfsgn	Agpfpnn
     18 Agpfpnn	Agpfsgn
     17 Pgmpgn	Pgfpgn
     17 Npmsgn-	X-
     16 Vgpp-sfpnnnn-p	Vgpp--npnn-n-p
     15 Agpfpgn	Agpmpgn
     14 Ncfsnn-	Ncfsin-
     13 Vgpp--npnn-n-p	Vgpp-sfpnnnn-p
     12 Vgps--npnn-n-p	Vgps-sfpnnnn-p
     11 Vgps-sfpnngn-p	Vgps-pfpnnnn-p
     11 Qg	Cg

```

### l2
```txt
     58 Ncfsgn-	Ncfpnn-
     52 Np---n-	X-
     37 Agpn--n	Agpfsnn
     33 Ncfsin-	Ncfsnn-
     31 Pgmpgn	Pgfpgn
     31 Npmsnn-	X-
     30 Y-	X-
     30 Ncfpnn-	Ncfsgn-
     17 Vgi-----n--n--	Vgps-pmpnnnn-p
     17 Npmsgn-	X-
     15 Vgps-pmpnnnn-p	Vgi-----n--n--
     15 Pgfpgn	Pgmpgn
     12 Agpfsgn	Agpfpnn
     10 Vgpp-sfpnnnn-p	Vgpp--npnn-n-p
     10 Vgpp-sfpnngn-p	Vgpp-pfpnnnn-p
     10 Ncfsnn-	Ncfsin-
     10 Agpfpgn	Agpmpgn
      9 Vgps-sfpnnnn-p	Vgps--npnn-n-p
      9 Agsn--n	Agsfsnn
      8 Vgps--npnn-n-p	Vgps-sfpnnnn-p

```