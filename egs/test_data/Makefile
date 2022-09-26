############################################
-include Makefile.options
############################################
data_dir?=../../data
zip=MATAS-20-06-05-v1.0.zip
out_dir?=data
python_dir=${CURDIR}/../../src/utils
python_cmd=PYTHONPATH=../../ LOG_LEVEL=INFO python
############################################
install/req:
	# conda create --name pos python=3.10
	pip install -r ../../requirements.txt
############################################
${out_dir}/extracted: 
	mkdir -p $@
${out_dir}: 
	mkdir -p $@	
############################################
${out_dir}/extracted/.done: $(data_dir)/${zip} | ${out_dir}/extracted
	unzip $(data_dir)/${zip} -d ${out_dir}/extracted
	touch $@
${out_dir}/all.txt: ${out_dir}/extracted/.done
	pv ${out_dir}/extracted/MATAS-v1.0/CONLLU/*.conllu > $@
${out_dir}/test_dad.txt: ${data_dir}/ma_testai.csv | ${out_dir}/extracted
	python local/csv_to_sentences.py --input $^ > $@_
	mv $@_ $@
${out_dir}/test.txt: ${out_dir}/test_dad.txt ${out_dir}/all.txt
	$(python_cmd) local/split.py --input ${out_dir}/all.txt --test_file ${out_dir}/test_dad.txt > $@_
	mv $@_ $@
${out_dir}/train.txt: ${out_dir}/test_dad.txt ${out_dir}/all.txt
	$(python_cmd) local/split.py --input ${out_dir}/all.txt --test_file ${out_dir}/test_dad.txt --return_train > $@_
	mv $@_ $@
############################################
${out_dir}/test_lemma.txt: ${out_dir}/test.txt
	$(python_cmd) local/lemmatize.py --input $^ > $@_
	mv $@_ $@
${out_dir}/train_lemma.txt: ${out_dir}/train.txt
	$(python_cmd) local/lemmatize.py --input $^ > $@_
	mv $@_ $@
${out_dir}/test_morph.txt: ${out_dir}/test.txt
	$(python_cmd) local/morphize.py --input $^ > $@_
	mv $@_ $@	
############################################
count/possibilities: ${out_dir}/test_lemma.txt
	$(python_cmd) local/count_posibilities.py --input $^
count/morph: ${out_dir}/test_morph.txt
	$(python_cmd) local/count_morph.py --show --input $^
############################################
start/lema:
	docker run -it -e SECRET=$(LEMA_SECRET) -e LEMA_TYPE=hunspell \
		-e LEMA_HUNSPELL_SKIP_OFFENSIVE=false \
		-p 8001:8000 airenas/lema:1.4.1212		
.PHONY: start/lema
start/morph:
	docker run -it \
		-p 8002:8090 semantikadocker.vdu.lt/v2/morph:0.2
.PHONY: start/morph
${out_dir}/unneeded_params.txt: ${out_dir}/train_lemma.txt
	$(python_cmd) local/find_unneeded.py --input $^
############################################
build: ${out_dir}/test_lemma.txt ${out_dir}/train_lemma.txt ${out_dir}/test_morph.txt \
	${out_dir}/unneeded_params.txt
.PHONY: build
############################################
clean:
	rm -rf $(out_dir)
.PHONY: clean	
.EXPORT_ALL_VARIABLES:	
