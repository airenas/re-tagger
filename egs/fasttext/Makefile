###################################################################
cfg?=Makefile.options
-include $(cfg)
###################################################################
log?=INFO
python_cmd=PYTHONPATH=./../../../ LOGLEVEL=$(log) python3

limit?=10000000000
sentences_dir?=$(work_dir)/corpus/sentences

dim?=150
window?=5
min_count?=5
minn?=3
maxn?=6
bucket?=2000000
neg?=5
lr?=0.05
sampling?=0.0001
epochs?=5
threads?=12
###################################################################
info:
	@echo "work_dir: $(work_dir)"
	@echo "corpus_bucket: $(corpus_bucket)"
	@echo "words limit: $(limit)"
	@echo "train window: $(window)"
	@echo "vocab min_count: $(min_count)"
	@echo "char ngrams: $(minn)-$(maxn), bucket: $(bucket)"
	@echo "negative samples: $(neg), learning rate: $(lr), sampling: $(sampling)"
	@echo "train epochs: $(epochs)"
	@echo "train threads: $(threads)"
	@echo "name: $(name)"
	@echo "dim: $(dim)"
.PHONY: info
###################################################################
build: $(work_dir)/data/$(name).$(dim).fifu
.PHONY: build
##################################################################
###################################################################
$(work_dir) $(work_dir)/corpus/sentences $(work_dir)/data:
	mkdir -p $@

###################################################################
# mcli (minio) or rc = https://github.com/rustfs/cli
s3_cmd?=mcli cp --recursive
# s3_cmd?=rc cp --recursive --retry-attempts 3
###################################################################
$(work_dir)/.dwn: | $(work_dir) $(sentences_dir)
	$(s3_cmd) $(corpus_bucket) $(sentences_dir)/
	touch $@
$(work_dir)/data/sentences.txt: $(work_dir)/.dwn | $(work_dir)/data
	$(python_cmd) ./local/to_sentences_from_parquet.py --input $(sentences_dir) --output $@_ --limit $(limit)
	mv $@_ $@
$(work_dir)/data/sentences.clean.txt: $(work_dir)/data/sentences.txt
	$(python_cmd) ./local/clean.py --input $^ --output $@_
	mv $@_ $@    
$(work_dir)/data/$(name).$(dim).bin: $(work_dir)/data/sentences.clean.txt
	$(python_cmd) ./local/train_fasttext.py --input $< --output $@ --dim $(dim) \
        --window $(window) --min-count $(min_count) --minn $(minn) --maxn $(maxn) \
        --bucket $(bucket) --neg $(neg) --lr $(lr) --sampling $(sampling) \
        --epochs $(epochs) --threads $(threads)
###################################################################
train: $(work_dir)/data/$(name).$(dim).bin
.PHONY: train
##################################################################
install/finalfusion:
	cargo install finalfusion-utils --features openblas --locked
###################################################################
$(work_dir)/data/%.fifu: $(work_dir)/data/%.bin 
	finalfusion convert -f fasttext $^ $@_
	mv $@_ $@
$(work_dir)/data/%.q.fifu: $(work_dir)/data/%.fifu
	finalfusion quantize -f finalfusion -a 1 -q opq -b 8 $^ $@_
	mv $@_ $@
quantize: $(work_dir)/data/$(name).$(dim).q.fifu
.PHONY: quantize
###################################################################
test/similarity: $(work_dir)/data/$(name).$(dim).fifu
	@echo "Testing similarity - enter word"
	finalfusion similar -k 10 $<
.PHONY: test/similarity
###################################################################
clean:
	rm -rf $(work_dir)
.PHONY: clean
###################################################################
clean/data:
	rm -rf $(work_dir)/data
.PHONY: clean/data
###################################################################
.EXPORT_ALL_VARIABLES:
