.PHONY: clean guard-gen e2e validate-report test check

clean:
	rm -rf artifacts reports outputs report.md

guard-gen:
	python3 scripts/guard_no_tracked_generated.py

e2e: clean
	rlhf-lab run --backend fallback --preset offline_hh_small --seed 0
	rlhf-lab report
	rlhf-lab validate --report reports

validate-report:
	rlhf-lab validate --report reports

test:
	pytest -q

check: guard-gen e2e test
	git diff --exit-code
