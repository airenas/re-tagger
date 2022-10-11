############################################
-include Makefile.options
take_n?=100000000000000
############################################
data_dir?=../../data
zip=MATAS-20-06-05-v1.0.zip
out_dir?=data
python_dir=${CURDIR}/../../src/utils
python_cmd=PYTHONPATH=../../ LOG_LEVEL=INFO python
f_before?=2
f_after?=2
f_func?=feat_word_v1
############################################
${out_dir}:
	mkdir -p $@	
############################################
${out_dir}/%: ../test_data/data/% | ${out_dir}
	cp $^ $@
${out_dir}/train.txt: ../test_data/data/train.txt | ${out_dir}
	cat $^ | head -n $(take_n) > $@
${out_dir}/train_f.txt: ${out_dir}/train.txt
	$(python_cmd) local/fix_data.py --input $^ > $@_
	mv $@_ $@
${out_dir}/test_f.txt: ${out_dir}/test.txt
	$(python_cmd) local/fix_data.py --input $^ > $@_
	mv $@_ $@
${out_dir}/train_fl.txt: ${out_dir}/train_f.txt ${out_dir}/train_lemma.txt
	$(python_cmd) local/join_lemma.py --input ${out_dir}/train_f.txt --lemmas ${out_dir}/train_lemma.txt > $@_
	mv $@_ $@
${out_dir}/test_fl.txt: ${out_dir}/test_f.txt ${out_dir}/test_lemma.txt
	$(python_cmd) local/join_lemma.py --input ${out_dir}/test_f.txt --lemmas ${out_dir}/test_lemma.txt > $@_
	mv $@_ $@
############################################
train: ${out_dir}/model.crf
${out_dir}/model.crf: ${out_dir}/train_fl.txt
	$(python_cmd) local/train.py --input ${out_dir}/train_fl.txt --out ${out_dir}/model.crf --f_before $(f_before) \
	--f_after $(f_after) --f_func $(f_func)
test/train: ${out_dir}/train_pred.txt
${out_dir}/train_pred.txt: ${out_dir}/model.crf ${out_dir}/train_fl.txt
	$(python_cmd) local/test.py --input ${out_dir}/train_fl.txt --model ${out_dir}/model.crf --out $@ \
	--f_before $(f_before) --f_after $(f_after) --f_func $(f_func)
test/test: ${out_dir}/test_pred.txt
${out_dir}/test_pred.txt: ${out_dir}/model.crf ${out_dir}/test_fl.txt
	$(python_cmd) local/test.py --input ${out_dir}/test_fl.txt --model ${out_dir}/model.crf --out $@ \
	--f_before $(f_before) --f_after $(f_after) --f_func $(f_func)
show/labels: ${out_dir}/train_fl.txt
	$(python_cmd) local/show_labels.py --input $^
${out_dir}/test_pred_restore.txt: ${out_dir}/test_pred.txt ${out_dir}/test_lemma_tags.txt ${out_dir}/tags.txt
	$(python_cmd) local/restore.py --lemmas ${out_dir}/test_lemma_tags.txt --pred ${out_dir}/test_pred.txt \
		--tags ${out_dir}/tags.txt > $@_
	mv $@_ $@
${out_dir}/train_pred_restore.txt: ${out_dir}/train_pred.txt ${out_dir}/train_lemma_tags.txt ${out_dir}/tags.txt
	$(python_cmd) local/restore.py --lemmas ${out_dir}/train_lemma_tags.txt --pred ${out_dir}/train_pred.txt \
		--tags ${out_dir}/tags.txt > $@_
	mv $@_ $@
${out_dir}/test_final_calc.txt: ${out_dir}/test_pred_restore.txt ${out_dir}/test_plain.txt
	$(python_cmd) $(python_dir)/compare.py --f1 ${out_dir}/test_plain.txt --f2 ${out_dir}/test_pred_restore.txt > $@_
	mv $@_ $@
${out_dir}/train_final_calc.txt: ${out_dir}/train_pred_restore.txt ${out_dir}/train_plain.txt
	$(python_cmd) $(python_dir)/compare.py --f1 ${out_dir}/train_plain.txt --f2 ${out_dir}/train_pred_restore.txt > $@_
	mv $@_ $@
calc/final: clean/test ${out_dir}/train_final_calc.txt ${out_dir}/test_final_calc.txt
############################################
build: test/train test/test
.PHONY: build
############################################
clean:
	rm -rf $(out_dir)
.PHONY: clean
clean/test:
	rm -f ${out_dir}/test_final_calc.txt
.PHONY: clean/test
.EXPORT_ALL_VARIABLES:
