PYTHON ?= python3
PYCACHE_DIR ?= .pycache_compile

.PHONY: help install test compile quality eda preprocess unzip-oulad train train-dropout train-oulad shap chatbot-prep chatbot clean run

help:
	@printf "Kullanilabilir hedefler:\n"
	@printf "  make install        - Bagimliliklari kur\n"
	@printf "  make eda            - Tum EDA scriptlerini calistir\n"
	@printf "  make preprocess     - Dropout ve OULAD on isleme scriptlerini calistir\n"
	@printf "  make train          - Guncel Dropout ve OULAD modellerini egit\n"
	@printf "  make chatbot-prep   - Chatbot config ve referans istatistiklerini uret\n"
	@printf "  make test           - Tum unittest testlerini calistir\n"
	@printf "  make compile        - Ana Python dosyalari icin syntax derleme kontrolu yap\n"
	@printf "  make quality        - compile + test\n"
	@printf "  make run TASK=...   - Python task runner ile grup komut calistir\n"
	@printf "  make clean          - Gecici Python cache dizinlerini temizle\n"

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

test:
	MPLCONFIGDIR=.matplotlib $(PYTHON) -m unittest discover -s tests -v

compile:
	PYTHONPYCACHEPREFIX=$(PYCACHE_DIR) $(PYTHON) -m py_compile \
		preprocessing/common.py \
		preprocessing/preprocess_dropout.py \
		preprocessing/prepare_oulad.py \
		chatbot/core.py \
		chatbot/app.py \
		modeling/common.py \
		modeling/model_dropout_localized.py \
		modeling/model_oulad_v2.py \
		modeling/shap_dropout_localized.py \
		modeling/shap_oulad.py \
		run_pipeline.py \
		tests/test_modeling_common.py \
		tests/test_preprocessing_common.py \
		tests/test_chatbot_core.py \
		tests/test_smoke.py

quality: compile test

eda:
	$(PYTHON) eda/eda_dropout.py
	$(PYTHON) eda/eda_oulad.py
	$(PYTHON) eda/eda_habits.py

unzip-oulad:
	unzip -o datasets/oulad/oulad.zip -d datasets/oulad

preprocess:
	$(PYTHON) -m preprocessing.preprocess_dropout
	$(PYTHON) -m preprocessing.prepare_oulad

train: train-dropout train-oulad

train-dropout:
	$(PYTHON) -m modeling.model_dropout_localized

train-oulad:
	$(PYTHON) -m modeling.model_oulad_v2

shap:
	$(PYTHON) -m modeling.shap_dropout_localized
	$(PYTHON) -m modeling.shap_oulad

chatbot-prep:
	$(PYTHON) -m chatbot.prepare_chatbot

chatbot:
	streamlit run chatbot/app.py

clean:
	rm -rf $(PYCACHE_DIR) __pycache__ .pytest_cache

run:
	MPLCONFIGDIR=.matplotlib $(PYTHON) run_pipeline.py $(TASK)
