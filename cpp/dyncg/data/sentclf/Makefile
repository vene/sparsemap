CORENLP_PATH ?= /home/vlad/code/stanford-corenlp-full-2017-06-09/

download-all: download-sst download-subj download-trec

download/trainDevTestTrees_PTB.zip:
	mkdir -p download
	wget -P download https://nlp.stanford.edu/sentiment/trainDevTestTrees_PTB.zip

download-subj:
	mkdir -p download
	wget -P download http://www.cs.cornell.edu/people/pabo/movie-review-data/rotten_imdb.tar.gz 

download-trec:
	mkdir -p download
	wget -P download http://cogcomp.org/Data/QA/QC/train_5500.label
	wget -P download http://cogcomp.org/Data/QA/QC/TREC_10.label

sst.txt: download/trainDevTestTrees_PTB.zip
	unzip download/trainDevTestTrees_PTB.zip
	python sst_labels.py trees/train.txt trees/dev.txt trees/test.txt

sst.txt.conllu: sst.txt
	java -cp "$(CORENLP_PATH)/*" -Xmx2g edu.stanford.nlp.pipeline.StanfordCoreNLP -props props -file sst.txt 

trec.txt:
	cat download/train_5500.label download/TREC_10.label | cut --complement -f 1 -d " " > trec.txt

trec.txt.conllu: trec.txt
	java -cp "$(CORENLP_PATH)/*" -Xmx2g edu.stanford.nlp.pipeline.StanfordCoreNLP -props props -file trec.txt 

subj.txt: 
	tar xf download/rotten_imdb.tar.gz
	iconv -f ISO-8859-1 -t UTF-8 plot.tok.gt9.5000 > subj_plot.txt
	iconv -f ISO-8859-1 -t UTF-8 quote.tok.gt9.5000 > subj_quote.txt
	cat subj_plot.txt subj_quote.txt > subj.txt
	python clean_subj.py 

subj.txt.conllu: subj.txt
	java -cp "$(CORENLP_PATH)/*" -Xmx4g edu.stanford.nlp.pipeline.StanfordCoreNLP -props props -file subj.txt 

%.indices: %.conllu
	python conllu_to_ix.py $< > $@

sst.target: download/stanfordSentimentTreebank.zip
	unzip download/stanfordSentimentTreebank.zip stanfordSentimentTreebank/dictionary.txt
	unzip download/stanfordSentimentTreebank.zip stanfordSentimentTreebank/sentiment_labels.txt
	python sst_labels.py > sst.target

sst.target: sst.txt

sst.split: sst.txt

trec.target: 
	cat download/train_5500.label download/TREC_10.label | cut -f 1 -d " " > trec.target

trec.split: 
	python split.py download/train_5500.label --valid 0.1 > trec.split
	while read p; do echo "test" >> trec.split; done <download/TREC_10.label 

subj.target:
	while read p; do echo "plot" >> subj.target; done <subj_plot.txt 
	while read p; do echo "quote" >> subj.target; done <subj_quote.txt 

subj.split:
	python split.py subj.txt --valid 0.1 --test 0.2 > subj.split
