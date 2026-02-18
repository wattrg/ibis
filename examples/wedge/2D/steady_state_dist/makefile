NUM_PROCS = 2

.PHONY: grid
grid:
	python grid.py
	ibis partition -f grid.su2 -o partitioned_grid -n ${NUM_PROCS}

.PHONY: job
job: grid
	ibis prep

.PHONY: prep
prep: job

.PHONY: run
run:
	mpirun -np ${NUM_PROCS} ibis run-dist

.PHONY: post
post:
	ibis post plot

.PHONY: clean
clean:
	ibis clean
	rm -f grid.su2
	rm -rf partitioned_grid
