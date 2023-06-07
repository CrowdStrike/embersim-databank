SOURCES := `find . -type f -name "*.py" ! -path "./venv/*" ! -path "*/.*"`

typecheck:
	@mypy $(SOURCES)

format:
	@autoflake --in-place --remove-all-unused-imports $(SOURCES) \
		&& isort $(SOURCES) \
		&& black -l 100 $(SOURCES)

TO_CLEAN := \
	__pycache__ \
	.ipynb_checkpoints \
	.mypy_cache
clean:
	@for f in $(TO_CLEAN); do \
		echo "> Removing $$f"; \
		find . -name $$f -print0 | xargs -0 rm -rf || true; \
	done
