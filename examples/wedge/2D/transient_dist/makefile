.PHONY: grid
grid:
	python grid.py
	ibis partition -f grid.su2 -n 2 -o partitioned_grid

.PHONY: job
job: grid
	ibis prep

.PHONY: prep
prep: job

.PHONY: run
run:
	mpirun -np 2 ibis run-dist

.PHONY: post
post:
	ibis post plot

.PHONY: clean
clean:
	ibis clean
	rm -f grid.su2
	rm -rf partitioned_grid
