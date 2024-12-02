############################################
cfg?=Makefile.options
-include $(cfg)
############################################
python_dir=${CURDIR}/../../src/utils
log?=INFO
python_cmd=PYTHONPATH=../../ LOG_LEVEL=$(log) python

data_dir?=../../data
out_dir?=data

emb-ff?=delfi_cbow
emb-fb-ff?=cc.lt

############################################
install/finalfusion:
	cargo install finalfusion-utils --features openblas
############################################
${out_dir}:
	mkdir -p $@
$(out_dir)/%.fifu: $(data_dir)/%.bin | $(out_dir)
	finalfusion convert -f fasttext $^ $@
$(out_dir)/%.fifu: $(out_dir)/%.bin
	finalfusion convert -f fasttext $^ $@	
$(out_dir)/%.150.bin: $(data_dir)/%.300.bin
	$(python_cmd) $(python_dir)/reduce_fasttext_o.py --input $^ --output $@ --dim 150
convert/delfi: $(out_dir)/$(emb-ff).150.fifu
convert/facebook: $(out_dir)/$(emb-fb-ff).150.fifu
############################################
$(out_dir)/%.q.fifu: $(out_dir)/%.fifu
	finalfusion quantize -f finalfusion -a 1 -q opq -b 8 $^ $@
quantize/delfi: $(out_dir)/$(emb-ff).150.q.fifu
quantize/facebook: $(out_dir)/$(emb-fb-ff).150.q.fifu
############################################
