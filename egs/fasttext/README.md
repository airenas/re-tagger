# Fasttext embedding modelio paruošimas

## Data

https://huggingface.co/datasets/VSSA-SDSA/LT_AI_BLKT

Split to sentences: https://github.com/airenas/icefall/tree/master/egs/cc-100/LM

## Train

Run `make train` to download and prepare the sentence corpus, then train a
FastText CBOW model. The native model is saved as
`$(work_dir)/data/$(name).$(dim).bin` 
