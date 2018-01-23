#required OPENMP 4.0 (since gcc 4.9)
SRC = ./cpp
HPP = ./cpp/Data-Analysis/lib/
STD = -std=c++11
OMP = -fopenmp
fmath = -ffast-math

paper = /tex/main.tex

ifeq ($(shell uname -o), Cygwin)
    gnu = -std=gnu++11
endif

ifeq ($(OS), Windows_NT)
	remove = del /s
	empty = nul
else
	remove = rm
	empty = /dev/null
endif

all: fbp_ga \
	$(paper) \
	$(wildcard img/**/*)

	latexmk -synctex=1 -bibtex -interaction=nonstopmode -file-line-error -pdf $(basename $<)
	$(MAKE) clean

fbp_ga: $(SRC)/main.cpp \

	$(CXX) $(STD) $(gnu) $(OMP) -O3 -I $(HPP) -o fbp_ga $(SRC)/main.cpp

pipeline: Snakefile
	snakemake

manual: $(paper) \
		$(wildcard img/**/*) \

	pdflatex $< 
	-bibtex $(basename $<)
	pdflatex $< 
	pdflatex $< 
	$(MAKE) cleanall

.PHONY: clean
clean: $(paper)
	@$(remove) $(basename $<).blg 2> $(empty)
	@$(remove) $(basename $<).log 2> $(empty)
	@$(remove) $(basename $<).out 2> $(empty)
	@$(remove) $(basename $<).fls 2> $(empty)
	@$(remove) $(basename $<).synctex.gz 2> $(empty)

.PHONY: cleanall
cleanall: $(paper) clean
	@$(remove) $(basename $<).aux 2> $(empty)
	@$(remove) $(basename $<).bbl 2> $(empty)
	@$(remove) $(basename $<).fdb_latexmk 2> $(empty)