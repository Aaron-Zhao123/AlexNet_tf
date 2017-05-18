clean-pyc:
	find . -name '*.pyc' -exec rm -rf {} +
	find . -name '*.pyo' -exec rm -rf {} +
	find . -name '*.~' -exec rm -rf {} +
run:
	python run.py
git-add:
	git add -A
	git commit -m"auto git add all"
	git push
git-commit:
	git commit -am"auto git commit"
	git push
