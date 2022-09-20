############################################
-include Makefile.options
############################################
data_dir?=../../data
zip=MATAS-20-06-05-v1.0.zip
out_dir?=data
python_dir=${CURDIR}/../../src/python
############################################
install/req:
	# conda create --name pos python=3.10
	pip install -r ../../requirements.txt
############################################
${out_dir}/extracted: 
	mkdir -p $@
############################################
${out_dir}/extracted/.done: $(data_dir)/${zip} | ${out_dir}/extracted
	unzip $(data_dir)/${zip} -d ${out_dir}/extracted
	touch $@
${out_dir}/all.txt: ${out_dir}/extracted/.done
	pv ${out_dir}/extracted/MATAS-v1.0/TAB-WPL/*.txt > $@
${out_dir}/sentences.txt: ${out_dir}/all.txt
	cat $^ | python $(python_dir)/tabwpl_to_sentences.py > $@_
	mv $@_ $@
${out_dir}/word2vec.model: ${out_dir}/sentences.txt
	cat $^ | python $(python_dir)/word2vec.py --output $@
${out_dir}/fasttext.model: ${out_dir}/sentences.txt
	cat $^ | python $(python_dir)/word2vec.py --output $@ --fast	
############################################
build: ${out_dir}/word2vec.model ${out_dir}/fasttext.model
.PHONY: build
############################################
clean:
	rm -rf $(out_dir)
.PHONY: clean	
.EXPORT_ALL_VARIABLES:	
