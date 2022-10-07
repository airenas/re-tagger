############################################
-include Makefile.options
take_n?=100000000000000
############################################
data_dir?=../../data
out_dir?=data
python_dir=${CURDIR}/../../src/utils
python_cmd=PYTHONPATH=../../ LOG_LEVEL=INFO python
############################################
${out_dir}:
	mkdir -p $@	
############################################
${out_dir}/%: ../test_data/data/% | ${out_dir}
	cp $^ $@
${out_dir}/train.txt: ../test_data/data/train.txt | ${out_dir}
	cat $^ | head -n $(take_n) > $@
${out_dir}/train_f.txt: ${out_dir}/train.txt
	$(python_cmd) ../crf/local/fix_data.py --input $^ > $@_
	mv $@_ $@
${out_dir}/test_f.txt: ${out_dir}/test.txt
	$(python_cmd) ../crf/local/fix_data.py --input $^ > $@_
	mv $@_ $@
############################################
train: ${out_dir}/models/morph
${out_dir}/models/morph: ${out_dir}/train_f.txt ${out_dir}/words ${out_dir}/tags
	$(python_cmd) local/train.py --input ${out_dir}/train_f.txt --in_v ${out_dir}/words --in_t ${out_dir}/tags --out $@
${out_dir}/words ${out_dir}/tags: ${out_dir}/train_f.txt
	$(python_cmd) local/prepare_vocab.py --input $^ --out_v ${out_dir}/words --out_t ${out_dir}/tags
############################################
start/tf-serving:
	docker run -it -p 8501:8501 -v $(CURDIR)/data/models/morph:/models/morph -e MODEL_NAME=morph tensorflow/serving:2.8.3
test/serving-api:
	curl -X POST http://localhost:8501/v1/models/morph:predict -d '{"inputs": [[485,4]]}'	
# test/train: ${out_dir}/model.crf ${out_dir}/train_fl.txt
# 	$(python_cmd) local/test.py --input ${out_dir}/train_fl.txt --model ${out_dir}/model.crf \
# 	--f_before $(f_before) --f_after $(f_after) --f_func $(f_func)
# test/test: ${out_dir}/test_pred.txt
# ${out_dir}/test_pred.txt: ${out_dir}/model.crf ${out_dir}/test_fl.txt
# 	$(python_cmd) local/test.py --input ${out_dir}/test_fl.txt --model ${out_dir}/model.crf --out $@ \
# 	--f_before $(f_before) --f_after $(f_after) --f_func $(f_func)
# show/labels: ${out_dir}/train_fl.txt
# 	$(python_cmd) local/show_labels.py --input $^
# ${out_dir}/test_pred_restore.txt: ${out_dir}/test_pred.txt ${out_dir}/test_lemma_tags.txt
# 	$(python_cmd) local/restore.py --lemmas ${out_dir}/test_lemma_tags.txt --pred ${out_dir}/test_pred.txt > $@_
# 	mv $@_ $@
# ${out_dir}/test_final_calc.txt: ${out_dir}/test_pred_restore.txt ${out_dir}/test_plain.txt
# 	$(python_cmd) $(python_dir)/compare.py --f1 ${out_dir}/test_plain.txt --f2 ${out_dir}/test_pred_restore.txt > $@_
# 	mv $@_ $@
# calc/final: ${out_dir}/test_final_calc.txt
############################################
build: test/train test/test
.PHONY: build
############################################
clean:
	rm -rf $(out_dir)
.PHONY: clean
.EXPORT_ALL_VARIABLES:
