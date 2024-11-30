cfg?=Makefile.options
-include $(cfg)
############################################
take_n?=100000000000000
############################################
data_dir?=../test_data/data
out_dir?=data
model?=${out_dir}/models/morph/1
out_model?=tm-0.1
out_model_path?=${out_dir}/$(out_model)
python_dir=${CURDIR}/../../src/utils
nn_params?=--use_ends
use_ft?=1
ft_model?=../../data/delfi_cbow.bin
cuda=
ifeq ($(no_cuda),1)
	cuda=CUDA_VISIBLE_DEVICES=""
endif
batch?=16
hidden?=300
log?=INFO
python_cmd=PYTHONPATH=../../ $(cuda) LOG_LEVEL=$(log) python
############################################
${out_dir} ${out_model_path}:
	mkdir -p $@	
############################################
${out_dir}/%: $(data_dir)/% | ${out_dir}
	cp $^ $@
${out_dir}/train.txt: $(data_dir)/train.txt | ${out_dir}
	cat $^ | head -n $(take_n) > $@
${out_dir}/%_f.txt: ${out_dir}/%.txt
	$(python_cmd) ../crf/local/fix_data.py --input $^ > $@_
	mv $@_ $@
${out_dir}/%_wanted.txt: ${out_dir}/%_f.txt
	$(python_cmd) ../test_data/local/to_plain.py --input $^ > $@_
	mv $@_ $@	
############################################
train: ${model}
############################################
${model}: ${out_dir}/train_f.txt ${out_dir}/words ${out_dir}/tags
ifeq ($(use_ft),1)
	$(python_cmd) local/train_ft.py --input ${out_dir}/train_f.txt --in_ft ${ft_model} --in_t ${out_dir}/tags  \
	    --out $@ --batch $(batch) --hidden $(hidden)
else
	$(python_cmd) local/train.py --input ${out_dir}/train_f.txt --in_v ${out_dir}/words --in_t ${out_dir}/tags \
		$(nn_params) --out $@ --batch $(batch) --hidden $(hidden)
endif
############################################
${out_dir}/words ${out_dir}/tags: ${out_dir}/train_f.txt
	$(python_cmd) local/prepare_vocab.py --input $^ --out_v ${out_dir}/words --out_t ${out_dir}/tags
############################################
start/tf-serving/cpu:
	docker run -it -p 8501:8501 -v $(CURDIR)/${out_dir}/models/morph:/models/morph -e MODEL_NAME=morph tensorflow/serving:latest
start/tf-serving/gpu:
	docker run -it -p 8501:8501 --gpus all -v $(CURDIR)/${out_dir}/models/morph:/models/morph -e MODEL_NAME=morph tensorflow/serving:latest-gpu	
test/serving-api:
	curl -X POST http://localhost:8501/v1/models/morph:predict -d '{"inputs": [[485,4]]}'	
test/test: ${out_dir}/test_pred.txt
############################################
${out_dir}/%_pred.txt: ${model} ${out_dir}/%_f.txt
ifeq ($(use_ft),1)
	$(python_cmd) local/test_ds_ft.py --input ${out_dir}/$*_f.txt --model ${model} \
		--in_ft ${ft_model} --in_t ${out_dir}/tags --out $@_
else
	$(python_cmd) local/test_ds.py --input ${out_dir}/$*_f.txt --model ${model} \
	    --in_v ${out_dir}/words --in_t ${out_dir}/tags $(nn_params) --out $@_
endif
	mv $@_ $@
.PRECIOUS: ${out_dir}/train_pred.txt ${out_dir}/test_pred.txt
############################################
${out_dir}/%_pred_restore.txt: ${out_dir}/%_pred.txt ${out_dir}/%_lemma_tags.txt ${out_dir}/tags.txt
	$(python_cmd) ../crf/local/restore.py --lemmas ${out_dir}/$*_lemma_tags.txt --pred ${out_dir}/$*_pred.txt \
	--tags ${out_dir}/tags.txt > $@_
	mv $@_ $@
${out_dir}/%_final_calc.txt: ${out_dir}/%_pred_restore.txt ${out_dir}/%_plain.txt
	$(python_cmd) $(python_dir)/compare.py --f1 ${out_dir}/$*_plain.txt --f2 ${out_dir}/$*_pred_restore.txt > $@_
	mv $@_ $@
${out_dir}/%_pred_calc.txt: ${out_dir}/%_pred.txt ${out_dir}/%_wanted.txt
	$(python_cmd) $(python_dir)/compare.py --f1 ${out_dir}/$*_wanted.txt --f2 ${out_dir}/$*_pred.txt > $@_
	mv $@_ $@	
calc/final: ${out_dir}/train_final_calc.txt ${out_dir}/test_final_calc.txt ${out_dir}/test_plain.txt ${out_dir}/test_pred_restore.txt
	$(python_cmd) $(python_dir)/compare.py --f1 ${out_dir}/test_plain.txt --f2 ${out_dir}/test_pred_restore.txt > /dev/null
calc/pred: ${out_dir}/train_pred_calc.txt ${out_dir}/test_pred_calc.txt
calc/test: ${out_dir}/test_pred_restore.txt ${out_dir}/test_final_calc.txt
show/err/test: ${out_dir}/test_final_calc.txt
	cat $^ | grep "<--diff-->" | cut -f2,3 | sort | uniq -c | sort -k1 -nr
############################################
${out_dir}/test_pred_ext.txt: ${out_dir}/test_f.txt
	$(python_cmd) ../test_data/local/bilstm_run.py --input $^ > $@_
	mv $@_ $@
${out_dir}/test_final_calc_ext.txt: ${out_dir}/test_pred_ext.txt ${out_dir}/test_plain.txt
	$(python_cmd) $(python_dir)/compare.py --f1 ${out_dir}/test_plain.txt --f2 ${out_dir}/test_pred_ext.txt > $@_
	mv $@_ $@
show/err/test/ext: ${out_dir}/test_final_calc_ext.txt
	cat $^ | grep "<--diff-->" | cut -f2,3 | sort | uniq -c | sort -k1 -nr | head -n 50
############################################
export/onnx: ${out_dir}/model.onnx
${out_dir}/model.onnx:
	$(python_cmd) local/export_onnx.py --model ${model} --out $@

############################################
pack: ${out_model_path}/model.onnx ${out_model_path}/tags ${out_model_path}/delfi_cbow.bin ${out_model_path}/tags_freq \
	${out_model_path}/clitics 
${out_model_path}/model.onnx: ${out_dir}/model.onnx | ${out_model_path}
	cp $^ $@
${out_model_path}/tags: ${out_dir}/tags | ${out_model_path}
	cp $^ $@
${out_model_path}/tags_freq: ${out_dir}/full_tags.txt | ${out_model_path}
	cp $^ $@
${out_model_path}/clitics: ${out_dir}/clitics_lemma.txt | ${out_model_path}
	cp $^ $@		
${out_model_path}/delfi_cbow.bin: ${ft_model} | ${out_model_path}
	cp $^ $@
pack/gz: ${out_dir}/${out_model}.tar.gz
${out_dir}/${out_model}.tar.gz: pack
	tar -czvf $@_ -C ${out_dir}/ ./${out_model}
	mv $@_ $@


############################################
build: test/train test/test
.PHONY: build
############################################
clean:
	rm -rf $(out_dir)
.PHONY: clean
.EXPORT_ALL_VARIABLES:
