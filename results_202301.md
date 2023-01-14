# Results

On MATAS v1.0 test set


| No | Method| Acc | Err count  |
|-|---|-|-|
| o2 | Oracle | `0.99235` | `338` |
| f2 | Fotonija morph | `0.95813` | `1852` |
||
| | **CRF** *context: 2 words before, 2 after*
| c2 | Word features | `---` | `---` |
| | **BiLSTM-CRF** | 
| l2 | embedding pretrained fasttext cbow, 300 hidden neurons | `---` | `---` |

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

```

### l2
```txt

```