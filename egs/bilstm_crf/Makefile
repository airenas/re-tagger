############################################
-include Makefile.options
take_n?=100000000000000
############################################
data_dir?=../../data
out_dir?=data
model=${out_dir}/models/morph/1
python_dir=${CURDIR}/../../src/utils
python_cmd=PYTHONPATH=../../ LOG_LEVEL=INFO python
nn_params=--use_ends
############################################
${out_dir}:
	mkdir -p $@	
############################################
${out_dir}/%: ../test_data/data/% | ${out_dir}
	cp $^ $@
${out_dir}/train.txt: ../test_data/data/train.txt | ${out_dir}
	cat $^ | head -n $(take_n) > $@
${out_dir}/%_f.txt: ${out_dir}/%.txt
	$(python_cmd) ../crf/local/fix_data.py --input $^ > $@_
	mv $@_ $@
############################################
train: ${out_dir}/models/morph
${model}: ${out_dir}/train_f.txt ${out_dir}/words ${out_dir}/tags
	$(python_cmd) local/train.py --input ${out_dir}/train_f.txt --in_v ${out_dir}/words --in_t ${out_dir}/tags \
		$(nn_params) --out $@
${out_dir}/words ${out_dir}/tags: ${out_dir}/train_f.txt
	$(python_cmd) local/prepare_vocab.py --input $^ --out_v ${out_dir}/words --out_t ${out_dir}/tags
############################################
start/tf-serving:
	docker run -it -p 8501:8501 -v $(CURDIR)/data/models/morph:/models/morph -e MODEL_NAME=morph tensorflow/serving:2.8.3
test/serving-api:
	curl -X POST http://localhost:8501/v1/models/morph:predict -d '{"inputs": [[485,4]]}'	
test/test: ${out_dir}/test_pred.txt
${out_dir}/%_pred.txt: ${model} ${out_dir}/%_f.txt
	$(python_cmd) local/test.py --input ${out_dir}/$*_f.txt --model ${model} \
		--in_v ${out_dir}/words --in_t ${out_dir}/tags $(nn_params) --out $@_
		mv $@_ $@
${out_dir}/%_pred_restore.txt: ${out_dir}/%_pred.txt ${out_dir}/%_lemma_tags.txt ${out_dir}/tags.txt
	$(python_cmd) ../crf/local/restore.py --lemmas ${out_dir}/$*_lemma_tags.txt --pred ${out_dir}/$*_pred.txt \
	--tags ${out_dir}/tags.txt > $@_
	mv $@_ $@
${out_dir}/%_final_calc.txt: ${out_dir}/%_pred_restore.txt ${out_dir}/%_plain.txt
	$(python_cmd) $(python_dir)/compare.py --f1 ${out_dir}/$*_plain.txt --f2 ${out_dir}/$*_pred_restore.txt > $@_
	mv $@_ $@
calc/final: ${out_dir}/train_final_calc.txt ${out_dir}/test_final_calc.txt
############################################
build: test/train test/test
.PHONY: build
############################################
clean:
	rm -rf $(out_dir)
.PHONY: clean
.EXPORT_ALL_VARIABLES:
