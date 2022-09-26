############################################
-include Makefile.options
take_n?=100000000000000
############################################
data_dir?=../../data
zip=MATAS-20-06-05-v1.0.zip
out_dir?=data
python_dir=${CURDIR}/../../src/utils
python_cmd=PYTHONPATH=../../ LOG_LEVEL=INFO python
############################################
${out_dir}:
	mkdir -p $@	
############################################
${out_dir}/test.txt: ../test_data/data/test.txt | ${out_dir}
	cp $^ $@
${out_dir}/train.txt: ../test_data/data/train.txt | ${out_dir}
	cat $^ | head -n $(take_n) > $@
${out_dir}/train_f.txt: ${out_dir}/train.txt
	$(python_cmd) local/fix_data.py --input $^ > $@_
	mv $@_ $@
${out_dir}/test_f.txt: ${out_dir}/test.txt
	$(python_cmd) local/fix_data.py --input $^ > $@_
	mv $@_ $@
############################################
train: ${out_dir}/model.crf
${out_dir}/model.crf: ${out_dir}/train_f.txt
	$(python_cmd) local/train.py --input ${out_dir}/train_f.txt --out ${out_dir}/model.crf
test/train: ${out_dir}/model.crf ${out_dir}/train_f.txt
	$(python_cmd) local/test.py --input ${out_dir}/train_f.txt --model ${out_dir}/model.crf
test/test: ${out_dir}/model.crf ${out_dir}/test_f.txt
	$(python_cmd) local/test.py --input ${out_dir}/test_f.txt --model ${out_dir}/model.crf
show/labels: ${out_dir}/train_f.txt
	$(python_cmd) local/show_labels.py --input $^
############################################
build: test/train test/test
.PHONY: build
############################################
clean:
	rm -rf $(out_dir)
.PHONY: clean
.EXPORT_ALL_VARIABLES:
