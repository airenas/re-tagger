############################################
-include Makefile.options
############################################
data_dir?=../../data
txt=BNDR_TARIMO.03.txt
out_dir?=data
python_dir=${CURDIR}/../../src/python
limit=10000000000
############################################
install/req:
	# conda create --name pos python=3.10
	pip install -r ../../requirements.txt
############################################
${out_dir}: 
	mkdir -p $@
############################################
${out_dir}/sentences.txt: $(data_dir)/${txt} | ${out_dir}
	pv $^ | head -n $(limit) | python $(python_dir)/drop_parenthesis.py > $@_
	mv $@_ $@
${out_dir}/word2vec.model: ${out_dir}/sentences.txt
	python $(python_dir)/word2vec.py --input $^ --output $@
${out_dir}/fasttext.model: ${out_dir}/sentences.txt
	python $(python_dir)/word2vec.py --input $^ --output $@ --fast	
############################################
build: ${out_dir}/word2vec.model ${out_dir}/fasttext.model
.PHONY: build
############################################
clean:
	rm -rf $(out_dir)
.PHONY: clean	
.EXPORT_ALL_VARIABLES:	
