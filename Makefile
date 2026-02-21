.PHONY: clean e2e validate-report test

clean:
	rm -rf artifacts reports

e2e: clean
	rlhf-lab run --backend fallback --preset offline_hh_small --seed 0
	rlhf-lab report
	rlhf-lab validate --report reports

validate-report:
	rlhf-lab validate --report reports

test:
	pytest -q