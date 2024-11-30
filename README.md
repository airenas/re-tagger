# re-tagger

Scripts for creating/testing POS tagger

## Prepare data

```bash
cd egs/test_data
make cfg=xxx start/morph &
make cfg=xxx start/lemma &
make cfg=xxx build
```

## Retrain bilstm_crf

```bash
cd egs/bilstm_crf
make cfg=xxx train
## show results on test set
make cfg=xxx show/err/test
# pack for inference
make cfg=xxx pack
```

