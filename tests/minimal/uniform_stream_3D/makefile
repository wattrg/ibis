
.PHONY: prep
prep:
	ibis prep

.PHONY: run
run:
	ibis run

.PHONY: post
post:
	ibis post plot

.PHONY: all
all: clean prep run post

.PHONY: clean
clean:
	ibis clean
	# rm -f grid.su2
