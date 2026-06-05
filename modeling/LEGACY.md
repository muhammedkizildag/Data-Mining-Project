# Legacy Modelleme Dosyalari

Bu klasordeki bazi script'ler tarihsel karsilastirma icin tutulur ve aktif final
modelleme akisinin parcasi degildir.

## Aktif script'ler

- `model_dropout_localized.py`
- `model_oulad_v2.py`
- `shap_dropout_localized.py`
- `shap_oulad.py`

## Legacy script'ler

- `legacy/model_dropout.py`
- `legacy/model_dropout_v2.py`
- `legacy/model_habits.py`
- `legacy/model_habits_v2.py`
- `legacy/model_oulad.py`
- `legacy/ablation_study.py`
- `legacy/ablation_study_oulad.py`

## Not

Legacy script'lerin bazilarinda:

- split oncesi scaler/feature engineering,
- eski preprocessing beklentileri,
- tarihsel metrik ve deney kurgulari

bulunabilir. Bu nedenle bu dosyalar final rapor, guncel benchmark veya chatbot
icin referans alinmamalidir.
