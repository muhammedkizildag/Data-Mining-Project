from pathlib import Path

import json

import joblib
import pandas as pd
import streamlit as st
from groq import Groq

from chatbot.core import (
    SEM_FALLBACK_MAP,
    auto_convert_turkish_scale,
    build_categorical_value_map,
    build_system_prompt,
    build_tr_to_en_map,
    check_analysis_ready,
    clean_non_turkish,
    extract_data_from_response,
)


st.set_page_config(
    page_title="Akademik Danisman",
    page_icon="🎓",
    layout="centered",
    menu_items={},
)

st.markdown(
    """
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display: none;}
    [data-testid="stToolbar"] {display: none;}
</style>
""",
    unsafe_allow_html=True,
)

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
MODEL_PATH = PROJECT_DIR / "models" / "best_model_dropout_localized.pkl"
FEATURE_CONFIG_PATH = BASE_DIR / "feature_config.json"
REFERENCE_STATS_PATH = BASE_DIR / "reference_stats.json"
API_KEY_PATH = BASE_DIR / "api_key.txt"

TARGET_NAMES = {
    0: "Terk Riski (Dropout)",
    1: "Devam Ediyor (Enrolled)",
    2: "Mezuniyet (Graduate)",
}
LLM_MODEL = "llama-3.3-70b-versatile"
LLM_USER_ERROR = (
    "Sohbet servisine su an ulasilamiyor. Lutfen biraz sonra tekrar deneyin."
)


def get_api_key():
    key = st.secrets.get("GROQ_API_KEY") if hasattr(st, "secrets") else None
    if key:
        return key

    env_key = st.session_state.get("_env_api_key")
    if env_key:
        return env_key

    import os

    env_key = os.environ.get("GROQ_API_KEY")
    if env_key:
        st.session_state["_env_api_key"] = env_key
        return env_key

    if API_KEY_PATH.exists():
        return API_KEY_PATH.read_text(encoding="utf-8").strip()

    return None


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_configs():
    with FEATURE_CONFIG_PATH.open("r", encoding="utf-8") as f:
        features = json.load(f)
    with REFERENCE_STATS_PATH.open("r", encoding="utf-8") as f:
        ref_stats = json.load(f)
    return features, ref_stats


@st.cache_data
def get_system_prompt(feature_config_payload, feature_order):
    return build_system_prompt(feature_config_payload, feature_order)


pipeline = load_model()
feature_config, reference_stats = load_configs()
FEATURE_ORDER = list(feature_config.keys())
TR_TO_EN_MAP = build_tr_to_en_map(feature_config)
CATEGORICAL_VALUE_MAP = build_categorical_value_map(feature_config)
SYSTEM_PROMPT = get_system_prompt(feature_config, FEATURE_ORDER)
API_KEY = get_api_key()
groq_client = Groq(api_key=API_KEY) if API_KEY else None


def chat_with_llm(messages):
    if groq_client is None:
        raise RuntimeError("Groq istemcisi baslatilamadi.")

    try:
        response = groq_client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            temperature=0.7,
            max_tokens=1024,
        )
    except Exception as exc:
        raise RuntimeError(LLM_USER_ERROR) from exc

    content = response.choices[0].message.content
    if not content:
        raise RuntimeError("Sohbet servisi bos bir yanit dondurdu.")
    return content


def predict_student(collected_data):
    input_values = []
    fallback_used = []

    for feat in FEATURE_ORDER:
        if feat in collected_data:
            raw_val = float(collected_data[feat])
        elif feat in SEM_FALLBACK_MAP and SEM_FALLBACK_MAP[feat] in collected_data:
            raw_val = float(collected_data[SEM_FALLBACK_MAP[feat]])
            fallback_used.append(feat)
        else:
            raw_val = float(feature_config[feat]["default"])
            raw_val = auto_convert_turkish_scale(feat, raw_val)
        input_values.append(raw_val)

    X = pd.DataFrame([input_values], columns=FEATURE_ORDER)
    prediction = pipeline.predict(X)[0]
    probabilities = pipeline.predict_proba(X)[0]
    return prediction, probabilities, fallback_used


def get_display_name(feature_name):
    return feature_config.get(feature_name, {}).get("tr", feature_name)


def format_feature_value(feature_name, value):
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.2f}"


def get_missing_priority_features(collected_data):
    missing = []
    for feat in FEATURE_ORDER:
        cfg = feature_config.get(feat, {})
        if cfg.get("priority") in ("essential", "high") and feat not in collected_data:
            missing.append(feat)
    return missing


def get_feature_comparison(collected_data):
    comparisons = []
    grad_stats = reference_stats["Graduate"]
    drop_stats = reference_stats["Dropout"]

    important_features = [
        "Curricular units 1st sem (approved)",
        "Curricular units 1st sem (grade)",
        "Curricular units 2nd sem (approved)",
        "Curricular units 2nd sem (grade)",
        "Admission grade",
        "Age at enrollment",
    ]

    for feat in important_features:
        if feat in collected_data:
            tr_name = feature_config[feat]["tr"]
            comparisons.append(
                {
                    "feature": tr_name,
                    "value": collected_data[feat],
                    "graduate_avg": grad_stats[feat]["mean"],
                    "dropout_avg": drop_stats[feat]["mean"],
                }
            )

    return comparisons


def build_prediction_response():
    prediction, probabilities, fallback_used = predict_student(
        st.session_state.collected_data
    )
    comparisons = get_feature_comparison(st.session_state.collected_data)
    pred_name = TARGET_NAMES[prediction]

    result_text = "\n\n---\n### Analiz Sonucu\n\n"
    if fallback_used:
        result_text += (
            "_Not: 2. donem bilgilerinin bir kismi eksik oldugu icin "
            "bu alanlarda 1. donem performansinin benzer devam ettigi varsayildi._\n\n"
        )

    result_text += "**Olasilik Dagilimi:**\n"
    for i in TARGET_NAMES:
        prob = probabilities[i] * 100
        bar = "#" * int(prob / 2)
        result_text += f"- {TARGET_NAMES[i]}: %{prob:.1f} {bar}\n"

    if comparisons:
        result_text += "\n**Senin Durumun vs Basarili Ogrenciler:**\n"
        for comp in comparisons:
            direction = "Ustunde" if comp["value"] >= comp["graduate_avg"] else "Altinda"
            result_text += (
                f"- {comp['feature']}: Sen {comp['value']}, Mezun ort {comp['graduate_avg']}, "
                f"Terk ort {comp['dropout_avg']} ({direction})\n"
            )

    analysis_messages = [
        {
            "role": "system",
            "content": (
                "Asagidaki analiz sonuclarini ogrenciye yapici bir dille aktar. "
                "Sadece Turkce konus. Kesin yargi kurma. Somut oneriler ver. "
                "4-5 cumleyi gecme.\n\n"
                f"TAHMIN SONUCU: {pred_name}\n"
                f"OLASILIKLAR: Terk %{probabilities[0]*100:.1f}, "
                f"Devam %{probabilities[1]*100:.1f}, Mezuniyet %{probabilities[2]*100:.1f}\n\n"
                f"KARSILASTIRMALAR:\n{json.dumps(comparisons, ensure_ascii=False, indent=2)}\n\n"
                "TOPLANAN VERILER:\n"
                f"{json.dumps(st.session_state.collected_data, ensure_ascii=False, indent=2)}"
            ),
        },
        {"role": "user", "content": "Bu sonuclari bana anlat ve oneriler ver."},
    ]

    analysis_text = chat_with_llm(analysis_messages)
    _, clean_analysis = extract_data_from_response(
        analysis_text,
        FEATURE_ORDER,
        feature_config,
        TR_TO_EN_MAP,
        CATEGORICAL_VALUE_MAP,
    )
    clean_analysis = clean_non_turkish(clean_analysis)
    return clean_analysis + result_text


if "messages" not in st.session_state:
    st.session_state.messages = []
if "collected_data" not in st.session_state:
    st.session_state.collected_data = {}
if "prediction_done" not in st.session_state:
    st.session_state.prediction_done = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "pending_prediction" not in st.session_state:
    st.session_state.pending_prediction = False


with st.sidebar:
    st.title("🎓 Akademik Danisman")
    st.markdown(
        "Seninle sohbet ederek akademik durumunu analiz eder ve kisisel oneriler sunar."
    )
    st.divider()

    if st.button("Yeni Sohbet Baslat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.collected_data = {}
        st.session_state.prediction_done = False
        st.session_state.chat_history = []
        st.session_state.pending_prediction = False
        st.rerun()


st.title("🎓 Akademik Danisman Asistani")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


if st.session_state.pending_prediction and not st.session_state.prediction_done:
    collected_items = list(st.session_state.collected_data.items())
    missing_features = get_missing_priority_features(st.session_state.collected_data)

    st.divider()
    st.subheader("Tahmin Oncesi Kontrol")
    st.caption("Model calistirilmadan once toplanan bilgiler asagidadir.")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Toplanan Bilgiler**")
        if collected_items:
            for feat, value in collected_items:
                st.markdown(
                    f"- **{get_display_name(feat)}:** {format_feature_value(feat, value)}"
                )
        else:
            st.markdown("- Henuz veri toplanmadi.")

    with col2:
        st.markdown("**Eksik Oncelikli Bilgiler**")
        if missing_features:
            for feat in missing_features:
                st.markdown(f"- {get_display_name(feat)}")
        else:
            st.markdown("- Oncelikli alanlar tamamlandi.")

    action_col1, action_col2 = st.columns(2)
    with action_col1:
        if st.button("Tahmini Olustur", type="primary", use_container_width=True):
            with st.spinner("Analiz hazirlaniyor..."):
                try:
                    final_response = build_prediction_response()
                except RuntimeError as exc:
                    final_response = str(exc)

            st.session_state.prediction_done = True
            st.session_state.pending_prediction = False
            st.session_state.messages.append(
                {"role": "assistant", "content": final_response}
            )
            st.session_state.chat_history.append(
                {"role": "assistant", "content": final_response}
            )
            st.rerun()

    with action_col2:
        if st.button("Bilgileri Duzelt", use_container_width=True):
            follow_up = (
                "Tamam, tahmini bekletiyorum. Duzeltmek veya eklemek istedigin bilgileri yazabilirsin."
            )
            st.session_state.pending_prediction = False
            st.session_state.messages.append({"role": "assistant", "content": follow_up})
            st.session_state.chat_history.append(
                {"role": "assistant", "content": follow_up}
            )
            st.rerun()


if not st.session_state.messages:
    if not API_KEY:
        error_msg = "Sistem yapilandirmasi eksik. Lutfen yoneticiyle iletisime gecin."
        st.session_state.messages.append({"role": "assistant", "content": error_msg})
        with st.chat_message("assistant"):
            st.markdown(error_msg)
    else:
        greeting = """Merhaba! Ben akademik danisman asistaniyim.

Seninle sohbet ederek akademik durumunu anlamak ve sana ozel oneriler sunmak istiyorum.

Nasılsın, bu donem dersler nasil gidiyor?"""

        st.session_state.messages.append({"role": "assistant", "content": greeting})
        st.session_state.chat_history.append({"role": "assistant", "content": greeting})
        with st.chat_message("assistant"):
            st.markdown(greeting)


if prompt := st.chat_input("Mesajini yaz..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if not API_KEY:
        error_msg = "Sistem su an kullanilamiyor. Lutfen daha sonra tekrar deneyin."
        st.session_state.messages.append({"role": "assistant", "content": error_msg})
        with st.chat_message("assistant"):
            st.markdown(error_msg)
    else:
        with st.chat_message("assistant"):
            with st.spinner("Dusunuyorum..."):
                try:
                    collected_summary = json.dumps(
                        st.session_state.collected_data, ensure_ascii=False
                    )
                    llm_messages = [
                        {
                            "role": "system",
                            "content": (
                                SYSTEM_PROMPT
                                + f"\n\nSu ana kadar toplanan veriler: {collected_summary}"
                            ),
                        }
                    ]
                    llm_messages.extend(st.session_state.chat_history)
                    llm_messages.append({"role": "user", "content": prompt})

                    response_text = chat_with_llm(llm_messages)
                    extracted_data, clean_response = extract_data_from_response(
                        response_text,
                        FEATURE_ORDER,
                        feature_config,
                        TR_TO_EN_MAP,
                        CATEGORICAL_VALUE_MAP,
                    )

                    if extracted_data:
                        st.session_state.collected_data.update(extracted_data)

                    clean_response = clean_response.replace("ANALIZ_HAZIR", "").strip()
                    clean_response = clean_non_turkish(clean_response)

                    if check_analysis_ready(response_text) and not st.session_state.prediction_done:
                        review_message = clean_response
                        if review_message:
                            review_message += (
                                "\n\nAsagida topladigim bilgileri kontrol et. "
                                "Uygunsa tahmini olusturabilirim."
                            )
                        else:
                            review_message = (
                                "Tahmin icin yeterli bilgiyi topladim. "
                                "Asagida topladigim verileri kontrol et; uygunsa tahmini olusturabilirim."
                            )
                        st.session_state.pending_prediction = True
                    else:
                        review_message = clean_response

                    st.markdown(review_message)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": review_message}
                    )
                    st.session_state.chat_history.append(
                        {"role": "user", "content": prompt}
                    )
                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": review_message}
                    )
                except RuntimeError as exc:
                    error_msg = str(exc)
                    st.markdown(error_msg)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": error_msg}
                    )
