#required OPENMP 4.0 (since gcc 4.9)
SRC = ./cpp
HPP = ./cpp/Data-Analysis/lib/
STD = -std=c++11
OMP = -fopenmp
fmath = -ffast-math

ifeq ($(OS), Windows_NT)
	inst = 'powershell "./install.ps1"'
	remove = del /s 
	empty =  nul
	sep = \\
else
	inst = ./install.sh
	remove = rm
	empty = /dev/null
	sep = /
endif

paper_file = traffic_report.tex
paper_out = traffic
tex_dir = tex

ifeq ($(OS), Cygwin)
    gnu = -std=gnu++14
endif

install: $(inst)
	$(inst)

pipeline: $(snake)
	snakemake

graph_pipe: $(snake)
	snakemake --dag | dot -Tpdf > protein_pipe.pdf

paper: $(tex_dir)/$(paper_file) \
	   $(wildcard $(tex_dir)/img/**/*)
	cd $(tex_dir) && latexmk -synctex=1 -bibtex -interaction=nonstopmode -file-line-error -pdf $(basename $(paper_file)) -jobname=$(paper_out) && cd ..
	$(MAKE) clean

fbp_ga: $(SRC)/fbp_ga.cpp
	$(CXX) $(STD) $(gnu) $(OMP) -O3 -I $(HPP) -o fbp_ga $(SRC)/fbp_ga.cpp

fbp_val: $(SRC)/fbp_val.cpp
	$(CXX) $(STD) $(gnu) $(OMP) -O3 -I $(HPP) -o fbp_ga $(SRC)/fbp_val.cpp

.PHONY: clean
clean: $(paper_out)
	$(remove) $(tex_dir)$(sep)$(paper_out).blg 2> $(empty)
	$(remove) $(tex_dir)$(sep)$(paper_out).log 2> $(empty)
	$(remove) $(tex_dir)$(sep)$(paper_out).out 2> $(empty)
	$(remove) $(tex_dir)$(sep)$(paper_out).fls 2> $(empty)
	$(remove) $(tex_dir)$(sep)$(paper_out).synctex.gz 2> $(empty)

.PHONY: cleanall
cleanall: $(paper_out) clean
	@$(remove) $(tex_dir)$(sep)$(paper_out).aux 2> $(empty)
	@$(remove) $(tex_dir)$(sep)$(paper_out).bbl 2> $(empty)
	@$(remove) $(tex_dir)$(sep)$(paper_out).fdb_latexmk 2> $(empty)