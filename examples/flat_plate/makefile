.PHONY: grid
grid:
	python grid.py

.PHONY: job
job: grid
	ibis prep

.PHONY: prep
prep: job

.PHONY: run
run:
	ibis run

.PHONY: post
post:
	ibis post plot_vtk

.PHONY: clean
clean:
	ibis clean
	rm -f grid.su2
