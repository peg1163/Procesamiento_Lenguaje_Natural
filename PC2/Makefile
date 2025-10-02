# Makefile para generación reproducible del corpus
PYTHON  ?= python3
SEED    ?= 42
N       ?= 5000
DATA_DIR := data
RAW      := $(DATA_DIR)/nlp_prueba_cc0c2.csv
HASHFILE := $(DATA_DIR)/nlp_prueba_cc0c2.sha256

ifeq ($(shell command -v sha256sum >/dev/null 2>&1; echo $$?),0)
  SHA256 = sha256sum
else
  SHA256 = shasum -a 256
endif

.PHONY: data hash verify-repro clean-data help

$(DATA_DIR):
	@mkdir -p $(DATA_DIR)

data: $(DATA_DIR)
	$(PYTHON) scripts/make_dataset.py --seed $(SEED) --n-samples $(N) --out $(RAW) --balance
	@$(SHA256) $(RAW) | tee $(HASHFILE)

hash:
	@$(SHA256) $(RAW)

verify-repro:
	@test -f $(RAW) || (echo "No se encontró $(RAW)"; exit 1)
	@test -f $(HASHFILE) || (echo "No se encontró $(HASHFILE)"; exit 1)
	@$(SHA256) --check $(HASHFILE) || (echo "El hash NO coincide"; exit 1)
	@echo "Reproducible: OK"

clean-data:
	@rm -f $(RAW) $(HASHFILE)

help:
	@echo "Targets: data, hash, verify-repro, clean-data"