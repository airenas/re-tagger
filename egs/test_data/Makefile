############################################
cfg?=Makefile.options
-include $(cfg)
############################################
data_dir?=../../data
zip?=MATAS-20-06-05-v1.0.zip
zip_dir?=MATAS-v1.0/CONLLU
out_dir?=data
python_dir=${CURDIR}/../../src/utils
python_cmd=PYTHONPATH=../../ LOG_LEVEL=INFO python
lemma_docker?=airenas/lema:1.4.1212
############################################
install/req:
	# conda create --name pos python=3.11
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
${out_dir}/all_1.txt: ${out_dir}/extracted/.done
	pv ${out_dir}/extracted/$(zip_dir)/*.conllu > $@
${out_dir}/all.txt: ${out_dir}/all_1.txt
	cat $^ | sed 's/ _/\t_/g' | \
		sed 's/Sing\t_\tMultext/Sing\t_\t_\t_\tMultext/g' > $@_
	mv $@_ $@	
${out_dir}/test_dad.txt: ${data_dir}/ma_testai.csv | ${out_dir}/extracted
	python local/csv_to_sentences.py --input $^ > $@_
	mv $@_ $@
${out_dir}/test_v0.txt: ${out_dir}/test_dad.txt ${out_dir}/all.txt
	$(python_cmd) local/split.py --input ${out_dir}/all.txt --test_file ${out_dir}/test_dad.txt > $@_
	mv $@_ $@
${out_dir}/train_v0.txt: ${out_dir}/test_dad.txt ${out_dir}/all.txt
	$(python_cmd) local/split.py --input ${out_dir}/all.txt --test_file ${out_dir}/test_dad.txt --return_train > $@_
	mv $@_ $@
############################################
${out_dir}/test_lemma.txt: ${out_dir}/test.txt ${out_dir}/clitics_lemma.txt
	$(python_cmd) local/lemmatize.py --input ${out_dir}/test.txt --clitics ${out_dir}/clitics_lemma.txt > $@_
	mv $@_ $@
${out_dir}/clitics_lemma.txt: ${out_dir}/train.txt
	$(python_cmd) local/find_clitics.py --input $^ > $@_
	mv $@_ $@	
${out_dir}/train_lemma.txt: ${out_dir}/train.txt ${out_dir}/clitics_lemma.txt
	$(python_cmd) local/lemmatize.py --input ${out_dir}/train.txt --clitics ${out_dir}/clitics_lemma.txt > $@_
	mv $@_ $@
${out_dir}/test_morph.txt: ${out_dir}/test.txt
	$(python_cmd) local/morphize.py --input $^ > $@_
	mv $@_ $@
${out_dir}/test_bilstm.txt: ${out_dir}/test.txt
	$(python_cmd) local/bilstm_run.py --input $^ > $@_
	mv $@_ $@
############################################
count/possibilities: ${out_dir}/test_lemma.txt
	$(python_cmd) local/count_posibilities.py --input $^
count/morph-vs-matas: ${out_dir}/test_morph.txt ${out_dir}/test_plain.txt
	$(python_cmd) $(python_dir)/compare.py --f1 ${out_dir}/test_plain.txt --f2 ${out_dir}/test_morph.txt --p2 2 > ${out_dir}/morf.cmp
count/bilstm-vs-matas: ${out_dir}/test_bilstm.txt ${out_dir}/test_plain.txt
	$(python_cmd) $(python_dir)/compare.py --f1 ${out_dir}/test_plain.txt --f2 ${out_dir}/test_bilstm.txt --p2 2 > ${out_dir}/bilstm.cmp
count/oracle: ${out_dir}/test_lemma_tags.txt ${out_dir}/test_plain.txt
	$(python_cmd) local/count_oracle.py --f1 ${out_dir}/test_plain.txt --f2 ${out_dir}/test_lemma_tags.txt > ${out_dir}/test.oracle
count/oracle-train: ${out_dir}/train_lemma_tags.txt ${out_dir}/train_plain.txt
	$(python_cmd) local/count_oracle.py --f1 ${out_dir}/train_plain.txt --f2 ${out_dir}/train_lemma_tags.txt > ${out_dir}/train.oracle
count/nom: ${out_dir}/train.txt
	$(python_cmd) local/count_nom.py --input $^
find/punctuations: ${out_dir}/all.txt
	$(python_cmd) local/find_punctuations.py --input $^ > ${out_dir}/punct.txt
	cat ${out_dir}/punct.txt
count/tags: ${out_dir}/tags.txt
${out_dir}/tags.txt: ${out_dir}/train_plain.txt
	$(python_cmd) local/count_tags.py --input $^ > $@_
	mv $@_ $@
count/full_tags: ${out_dir}/full_tags.txt
${out_dir}/full_tags.txt: ${out_dir}/train.txt
	$(python_cmd) local/count_full_tags.py --input $^ > $@_
	mv $@_ $@
show/err/morph: ${out_dir}/morf.cmp
	cat $^ | grep "<--diff-->" | cut -f2,3 | sort | uniq -c | sort -k1 -nr
show/err/bilstm: ${out_dir}/bilstm.cmp
	cat $^ | grep "<--diff-->" | cut -f2,3 | sort | uniq -c | sort -k1 -nr
############################################
start/lemma:
	docker run -it -e SECRET=$(LEMA_SECRET) -e LEMA_TYPE=hunspell \
		-e LEMA_HUNSPELL_SKIP_OFFENSIVE=false \
		-p 8001:8000 $(lemma_docker)
.PHONY: start/lemma
start/morph:
	docker run -it \
		-p 8002:8090 semantikadocker.vdu.lt/v2/morph:0.2
.PHONY: start/morph
${out_dir}/unneeded_params.txt: ${out_dir}/train_lemma.txt
	$(python_cmd) local/find_unneeded.py --input $^
############################################
${out_dir}/test_lemma_tags.txt: ${out_dir}/test_lemma.txt
	$(python_cmd) local/to_tags.py --input $^ > $@_
	mv $@_ $@
${out_dir}/train_lemma_tags.txt: ${out_dir}/train_lemma.txt
	$(python_cmd) local/to_tags.py --input $^ > $@_
	mv $@_ $@
############################################
${out_dir}/%.txt: ${out_dir}/%_v0.txt
	$(python_cmd) local/fix_data.py --input $^ > $@_
	mv $@_ $@
${out_dir}/%_plain.txt: ${out_dir}/%.txt
	$(python_cmd) local/to_plain.py --input $^ > $@_
	mv $@_ $@
############################################
build: ${out_dir}/test_lemma_tags.txt ${out_dir}/train_lemma.txt ${out_dir}/test_morph.txt \
	${out_dir}/test_plain.txt ${out_dir}/train_lemma_tags.txt ${out_dir}/train_plain.txt \
	${out_dir}/tags.txt
.PHONY: build
unneeded: ${out_dir}/unneeded_params.txt 
############################################
clean:
	rm -rf $(out_dir)
.PHONY: clean	
.EXPORT_ALL_VARIABLES:	
