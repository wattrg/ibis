grid.su2: grid.py
	python grid.py

config/config_status: grid.su2 job.py
	ibis prep

.PHONY: prep
prep: config/config_status

.PHONY: run
run:
	ibis run

.PHONY: post
post:
	ibis post plot

.PHONY: clean
clean:
	- ibis clean
	- rm grid.su2
