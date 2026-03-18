MODEL_SRC := ../core/src/main/resources/com/jlpka/cjclassifier/cjlogprobs.gz
MODEL_DST := src/cjclassifier/cjlogprobs.gz

.PHONY: model build install install-dev test clean

model: $(MODEL_DST)

$(MODEL_DST): $(MODEL_SRC)
	cp $(MODEL_SRC) $(MODEL_DST)

build: model
	python3 -m build

install: model
	pip install .

install-dev: model
	pip install -e .

test: model
	python3 -m pytest tests/ -v

clean:
	rm -f $(MODEL_DST)
	rm -rf build dist *.egg-info src/*.egg-info
