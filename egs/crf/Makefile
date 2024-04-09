############################################
cfg?=Makefile.options
-include $(cfg)
take_n?=100000000000000
############################################
data_dir?=../test_data/data
out_dir?=data
python_dir=${CURDIR}/../../src/utils
python_cmd=PYTHONPATH=../../ LOG_LEVEL=INFO python
f_before?=2
f_after?=2
f_func?=feat_word_v1
f_skip_punct=
no_punct?=0
ifeq ($(no_punct),1)
    no_punct_param=--no_punct
	cmp_param=--ip1
	cmp_test_param=--ip1 --ip2
endif
############################################
${out_dir}:
	mkdir -p $@	
############################################
${out_dir}/%: $(data_dir)/% | ${out_dir}
	cp $^ $@
${out_dir}/train.txt: $(data_dir)/train.txt | ${out_dir}
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
${out_dir}/%_wanted.txt: ${out_dir}/%_f.txt
	$(python_cmd) ../test_data/local/to_plain.py --input $^ > $@_
	mv $@_ $@	
############################################
train: ${out_dir}/model.crf
${out_dir}/model.crf: ${out_dir}/train_fl.txt
	$(python_cmd) local/train.py --input ${out_dir}/train_fl.txt --out ${out_dir}/model.crf --f_before $(f_before) \
	--f_after $(f_after) --f_func $(f_func) $(f_skip_punct) $(no_punct_param)
test/train: ${out_dir}/train_pred.txt
${out_dir}/train_pred.txt: ${out_dir}/model.crf ${out_dir}/train_fl.txt
	$(python_cmd) local/test.py --input ${out_dir}/train_fl.txt --model ${out_dir}/model.crf --out $@ \
	--f_before $(f_before) --f_after $(f_after) --f_func $(f_func) $(f_skip_punct) $(no_punct_param)
test/test: ${out_dir}/test_pred.txt
${out_dir}/test_pred.txt: ${out_dir}/model.crf ${out_dir}/test_fl.txt
	$(python_cmd) local/test.py --input ${out_dir}/test_fl.txt --model ${out_dir}/model.crf --out $@ \
	--f_before $(f_before) --f_after $(f_after) --f_func $(f_func) $(f_skip_punct) $(no_punct_param)
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
	$(python_cmd) $(python_dir)/compare.py --f1 ${out_dir}/test_plain.txt --f2 ${out_dir}/test_pred_restore.txt \
	    $(cmp_test_param)> $@_
	mv $@_ $@
${out_dir}/train_final_calc.txt: ${out_dir}/train_pred_restore.txt ${out_dir}/train_plain.txt
	$(python_cmd) $(python_dir)/compare.py --f1 ${out_dir}/train_plain.txt --f2 ${out_dir}/train_pred_restore.txt > $@_
	mv $@_ $@
${out_dir}/%_pred_calc.txt: ${out_dir}/%_pred.txt ${out_dir}/%_wanted.txt
	$(python_cmd) $(python_dir)/compare.py --f1 ${out_dir}/$*_wanted.txt --f2 ${out_dir}/$*_pred.txt $(cmp_param) > $@_
	mv $@_ $@	
calc/final: clean/test ${out_dir}/train_final_calc.txt ${out_dir}/test_final_calc.txt
calc/pred: clean/pred ${out_dir}/train_pred_calc.txt ${out_dir}/test_pred_calc.txt
show/err/test: ${out_dir}/test_final_calc.txt
	cat $^ | grep "<--diff-->" | cut -f2,3 | sort | uniq -c | sort -k1 -nr	
############################################
build: test/train test/test
.PHONY: build
############################################
clean:
	rm -rf $(out_dir)
.PHONY: clean
clean/test:
	rm -f ${out_dir}/test_final_calc.txt ${out_dir}/train_final_calc.txt
clean/pred:
	rm -f ${out_dir}/train_pred_calc.txt ${out_dir}/test_pred_calc.txt	
.PHONY: clean/test
.EXPORT_ALL_VARIABLES:
