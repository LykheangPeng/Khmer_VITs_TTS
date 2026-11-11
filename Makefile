# Reproduces the reference run. Override variables, e.g.:  make train HOURS=20 BATCH=48
RAW   ?= data/raw
PREP  ?= data/prep
RUN   ?= runs/khmer_multispeaker
HOURS ?= 11
BATCH ?= 32

.PHONY: install test test-fast download prepare train tensorboard samples

install:
	pip install -e ".[download,logging,dev]"

test:
	pytest -q

test-fast:
	pytest -q -m "not slow"

download:            # needs: export MDC_API_KEY=...
	khmer-vits download --out $(RAW)

prepare:
	khmer-vits prepare --src $(RAW) --out $(PREP)

train:
	khmer-vits train --data $(PREP) --run_dir $(RUN) --hours $(HOURS) --batch_size $(BATCH)

tensorboard:
	tensorboard --logdir $(RUN)/logs

samples:
	khmer-vits synth --ckpt $(RUN)/G_latest.pt --speaker all --out samples/sample.wav \
		--text "ប្រទេសកម្ពុជាមានប្រជាជនជាង ១៦ លាននាក់"
