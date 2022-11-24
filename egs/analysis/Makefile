log?=INFO
python_cmd=PYTHONPATH=../../ $(cuda) LOG_LEVEL=$(log) python
spec-err?="Ncfpnn-\tNcfsgn-"

data:
	mkdir -p $@

data/%.txt: ../test_data/data/%.txt | data
	cp $^ $@	

data/%_final_calc_ws.txt: data/%_final_calc.txt data/%.txt
	$(python_cmd) local/join.py --input data/$*_final_calc.txt --in_s data/$*.txt > $@_
	mv $@_ $@

data/%_final_calc.txt: | data
	curl https://sinteze.intelektika.lt/list.data/matas-test/$*_final_calc.txt -o $@

errors/types: data/train_final_calc.txt
	cat data/train_final_calc.txt | grep "<--diff-->" | cut -f2,3 | sort | uniq -c | sort -k1 -nr | head -n 20

errors/spec/%: data/%_final_calc_ws.txt
	cat $^ | grep -P  -C20 $(spec-err) --color=always | awk -v OFS='\t' '{$$1 = "\033[30m" $$1 "\033[0m";} 1' \
	    | aha --black --title '$(spec-err)' > $*_out.html

build: errors/spec/test errors/spec/train data/test_final_calc_ws.txt data/train_final_calc_ws.txt

clean:
	rm -rf data	
