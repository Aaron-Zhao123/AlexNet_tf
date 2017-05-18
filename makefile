clean:
	find . -name '*.pyc' -exec rm -rf {} +
	find . -name '*.pyo' -exec rm -rf {} +
	find . -name '*.~' -exec rm -rf {} +
macrun:
	python run.py
linuxrun:
	CUDA_VISIBLE_DEVICES=$(GPU) python run.py
git-add:
	git add -A
	git commit -m"auto git add all"
	git push
git-commit:
	git commit -am"auto git commit"
	git push
