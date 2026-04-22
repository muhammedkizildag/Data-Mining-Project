import streamlit as st
import joblib
import json
import numpy as np
import os

from groq import Groq

# ============================================================
# SAYFA AYARLARI
# ============================================================
st.set_page_config(
    page_title="Akademik Danışman",
    page_icon="🎓",
    layout="centered",
    menu_items={}
)

st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display: none;}
    [data-testid="stToolbar"] {display: none;}
</style>
""", unsafe_allow_html=True)

# ============================================================
# API KEY
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)

def get_api_key():
    key = os.environ.get("GROQ_API_KEY")
    if key:
        return key
    config_path = os.path.join(BASE_DIR, "api_key.txt")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return f.read().strip()
    return None

API_KEY = get_api_key()

# ============================================================
# VERİ VE MODEL YÜKLEME
# ============================================================
@st.cache_resource
def load_model():
    return joblib.load(os.path.join(PROJECT_DIR, "models/best_model_dropout_localized.pkl"))

@st.cache_data
def load_configs():
    with open(os.path.join(BASE_DIR, "scaler_params.json"), "r", encoding="utf-8") as f:
        scaler = json.load(f)
    with open(os.path.join(BASE_DIR, "feature_config.json"), "r", encoding="utf-8") as f:
        features = json.load(f)
    with open(os.path.join(BASE_DIR, "reference_stats.json"), "r", encoding="utf-8") as f:
        ref_stats = json.load(f)
    return scaler, features, ref_stats

model = load_model()
scaler_params, feature_config, reference_stats = load_configs()

FEATURE_ORDER = list(feature_config.keys())

TARGET_NAMES = {0: "Terk Riski (Dropout)", 1: "Devam Ediyor (Enrolled)", 2: "Mezuniyet (Graduate)"}

# ============================================================
# GROQ CLIENT
# ============================================================
if API_KEY:
    groq_client = Groq(api_key=API_KEY)
else:
    groq_client = None

LLM_MODEL = "llama-3.3-70b-versatile"

# ============================================================
# YARDIMCI FONKSİYONLAR
# ============================================================
def normalize_value(feature_name, raw_value):
    raw_value = float(raw_value)
    params = scaler_params[feature_name]
    mn, mx = params['min'], params['max']
    if mx == mn:
        return 0.0
    return (raw_value - mn) / (mx - mn)

def predict_student(collected_data):
    input_values = []
    for feat in FEATURE_ORDER:
        if feat in collected_data:
            raw_val = collected_data[feat]
        else:
            raw_val = feature_config[feat]['default']
        normalized = normalize_value(feat, raw_val)
        normalized = max(0.0, min(1.0, normalized))
        input_values.append(normalized)

    X = np.array([input_values])
    prediction = model.predict(X)[0]
    probabilities = model.predict_proba(X)[0]
    return prediction, probabilities

def get_feature_comparison(collected_data):
    comparisons = []
    grad_stats = reference_stats['Graduate']
    drop_stats = reference_stats['Dropout']

    important_features = [
        'Curricular units 1st sem (approved)', 'Curricular units 1st sem (grade)',
        'Curricular units 2nd sem (approved)', 'Curricular units 2nd sem (grade)',
        'Admission grade', 'Age at enrollment'
    ]

    for feat in important_features:
        if feat in collected_data:
            val = collected_data[feat]
            grad_mean = grad_stats[feat]['mean']
            drop_mean = drop_stats[feat]['mean']
            tr_name = feature_config[feat]['tr']
            comparisons.append({
                'feature': tr_name,
                'value': val,
                'graduate_avg': grad_mean,
                'dropout_avg': drop_mean
            })
    return comparisons

def build_system_prompt():
    feature_info = []
    for feat, cfg in feature_config.items():
        if cfg['priority'] in ('essential', 'high'):
            if cfg['type'] == 'categorical':
                opts = ", ".join(cfg['options'].keys())
                feature_info.append(f"- {cfg['tr']}: [{opts}]")
            else:
                feature_info.append(f"- {cfg['tr']}: ({cfg['range'][0]}-{cfg['range'][1]})")

    features_text = "\n".join(feature_info)

    return f"""Sen bir Türk üniversitesinde çalışan akademik danışman asistanısın.

DİL KURALLARI:
- SADECE Türkçe konuş. Kesinlikle İngilizce kelime veya ifade kullanma.
- "sometimes", "information", "majority" gibi İngilizce kelimeler YASAK.
- Doğal, samimi ve sıcak bir Türkçe kullan.

GÖREVİN:
Öğrenciyle doğal bir sohbet kurarak akademik durumunu anlamak ve ona yardımcı olmak.

TOPLANMASI GEREKEN BİLGİLER (öncelik sırasıyla):
{features_text}

NOT SİSTEMİ DÖNÜŞÜMÜ:
- Türkiye'de üniversiteler genelde 4'lük not sistemi kullanır (AA=4.0, BA=3.5, BB=3.0, ..., FF=0).
- Bu sistemdeki veriler 20'lik ölçektedir. Dönüşüm: (4lük_not / 4.0) × 20
- Örnek: 4 üzerinden 2.4 → (2.4/4.0)×20 = 12.0
- Öğrenci 100'lük sistem kullanıyorsa: (100lük_not / 100) × 20
- Öğrenci hangi sistemi kullandığını belirtmezse, Türkiye'de yaygın olan 4'lük sistem varsay.
- Dönüşümü sen yap, öğrenciye 20'lik sistemi sorma.

DERS SAYISI HESAPLAMA:
- Öğrenci "alttan 3 dersim var" derse, bu geçemediği ders sayısıdır.
- Alınan ders = geçilen ders + kalan ders olarak hesaplayabilirsin.
- Öğrenci sadece toplam ders ve kalan ders veriyorsa, geçilen dersi kendin hesapla.

SOHBET KURALLARI:
1. Soruları doğal sohbet akışında sor. Anket gibi sıralama yapma, madde madde listeleme.
2. Öğrenci bir bilgi verdiğinde kısaca anladığını göster, empati kur.
3. Birden fazla bilgiyi tek mesajda toplamaya çalış ama zorla değil.
4. Öğrenci bir bilgiyi bilmiyorsa veya dolaylı cevap veriyorsa, cevabından çıkarım yap. Aynı soruyu tekrar sorma.
5. Öğrenci dolaylı cevap verirse anlamaya çalış. Örnek: "Devlet okulunda okuyorum" → burs sorusuna doğrudan cevap değil, burs alıp almadığını tekrar sor ama farklı bir şekilde.
6. Yeterli bilgi topladığında (en az her iki dönem için: alınan ders, geçilen ders ve not ortalaması) "ANALIZ_HAZIR" yaz.
7. ASLA "kalırsın", "başarısız olursun", "bırakma rikin var" gibi olumsuz kesin yargılar kullanma.
8. Her zaman yapıcı, destekleyici ve motive edici ol.
9. Kısa ve öz cevaplar ver. 2-3 cümleyi geçme.
10. Öğrenci sınıf bilgisi verirse (örn. "3. sınıfım") bunu not et ama kayıt yaşını sormaya devam et.

ÖNEMLİ: Topladığın her bilgiyi şu formatta mesajının SONUNA ekle (öğrenci görmez, sistem okur):
[DATA: {{"özellik_adı": sayısal_değer, ...}}]

Değerler HER ZAMAN sayı olmalı, string olmamalı. Doğru: 12.0  Yanlış: "12.0"
Not dönüşümünü yaptıktan sonraki değeri yaz.

Örnek: Öğrenci "2. dönem 5 ders aldım 3ünü geçtim notum 4 üzerinden 2.4" derse:
[DATA: {{"Curricular units 2nd sem (enrolled)": 5, "Curricular units 2nd sem (approved)": 3, "Curricular units 2nd sem (grade)": 12.0}}]

Özellik adlarını TAM OLARAK şu listeden kullan:
{json.dumps(FEATURE_ORDER, ensure_ascii=False)}
"""

def extract_data_from_response(response_text):
    import re
    data_match = re.search(r'\[DATA:\s*(\{.*?\})\]', response_text, re.DOTALL)
    if data_match:
        try:
            data = json.loads(data_match.group(1))
            clean_text = re.sub(r'\[DATA:.*?\]', '', response_text, flags=re.DOTALL).strip()
            return data, clean_text
        except json.JSONDecodeError:
            pass
    return None, response_text

def check_analysis_ready(response_text):
    return "ANALIZ_HAZIR" in response_text

def chat_with_llm(messages):
    response = groq_client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        temperature=0.7,
        max_tokens=1024
    )
    return response.choices[0].message.content

# ============================================================
# SESSION STATE
# ============================================================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "collected_data" not in st.session_state:
    st.session_state.collected_data = {}
if "prediction_done" not in st.session_state:
    st.session_state.prediction_done = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.title("🎓 Akademik Danışman")
    st.markdown("Seninle sohbet ederek akademik durumunu analiz eder ve kişisel öneriler sunar.")

    st.divider()

    if st.button("🔄 Yeni Sohbet Başlat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.collected_data = {}
        st.session_state.prediction_done = False
        st.session_state.chat_history = []
        st.rerun()

# ============================================================
# ANA EKRAN
# ============================================================
st.title("🎓 Akademik Danışman Asistanı")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ============================================================
# İLK MESAJ
# ============================================================
if not st.session_state.messages:
    if not API_KEY:
        error_msg = "⚠️ Sistem yapılandırması eksik. Lütfen yöneticiyle iletişime geçin."
        st.session_state.messages.append({"role": "assistant", "content": error_msg})
        with st.chat_message("assistant"):
            st.markdown(error_msg)
    else:
        greeting = """Merhaba! Ben akademik danışman asistanınım.

Seninle biraz sohbet ederek akademik durumunu anlamak ve sana özel öneriler sunmak istiyorum. Cevapların tamamen gizli kalır ve sadece sana yardımcı olmak için kullanılır.

Nasılsın, bu dönem dersler nasıl gidiyor?"""

        st.session_state.messages.append({"role": "assistant", "content": greeting})
        st.session_state.chat_history.append({"role": "assistant", "content": greeting})
        with st.chat_message("assistant"):
            st.markdown(greeting)

# ============================================================
# CHAT INPUT
# ============================================================
if prompt := st.chat_input("Mesajını yaz..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if not API_KEY:
        error_msg = "⚠️ Sistem şu an kullanılamıyor. Lütfen daha sonra tekrar deneyin."
        st.session_state.messages.append({"role": "assistant", "content": error_msg})
        with st.chat_message("assistant"):
            st.markdown(error_msg)
    else:
        with st.chat_message("assistant"):
            with st.spinner("Düşünüyorum..."):
                try:
                    system_prompt = build_system_prompt()
                    collected_summary = json.dumps(st.session_state.collected_data, ensure_ascii=False)

                    llm_messages = [
                        {"role": "system", "content": system_prompt + f"\n\nŞu ana kadar toplanan veriler: {collected_summary}"}
                    ]
                    llm_messages.extend(st.session_state.chat_history)
                    llm_messages.append({"role": "user", "content": prompt})

                    response_text = chat_with_llm(llm_messages)

                    extracted_data, clean_response = extract_data_from_response(response_text)

                    if extracted_data:
                        st.session_state.collected_data.update(extracted_data)

                    if check_analysis_ready(response_text) and not st.session_state.prediction_done:
                        st.session_state.prediction_done = True
                        prediction, probabilities = predict_student(st.session_state.collected_data)

                        comparisons = get_feature_comparison(st.session_state.collected_data)

                        pred_name = TARGET_NAMES[prediction]

                        result_text = f"\n\n---\n### 📊 Analiz Sonucu\n\n"

                        result_text += "**Olasılık Dağılımı:**\n"
                        for i in TARGET_NAMES:
                            prob = probabilities[i] * 100
                            bar = "█" * int(prob / 2)
                            result_text += f"- {'🔴' if i==0 else '🟡' if i==1 else '🟢'} {TARGET_NAMES[i]}: %{prob:.1f} {bar}\n"

                        if comparisons:
                            result_text += "\n**Senin Durumun vs Başarılı Öğrenciler:**\n"
                            for comp in comparisons:
                                direction = "✅" if comp['value'] >= comp['graduate_avg'] else "⚠️"
                                result_text += f"- {direction} {comp['feature']}: Sen: {comp['value']}, Mezun ort: {comp['graduate_avg']}, Terk ort: {comp['dropout_avg']}\n"

                        analysis_messages = [
                            {"role": "system", "content": f"""Aşağıdaki analiz sonuçlarını öğrenciye YAPICI ve DESTEKLEYİCİ bir dille aktar. Türkçe konuş.
Kesinlikle "kalırsın" veya "başarısız olursun" gibi ifadeler kullanma.
Somut ve uygulanabilir öneriler ver.
What-if analizi yap: "Eğer şunu yaparsan, durumun iyileşir" gibi.
Kısa ve öz tut.

TAHMİN SONUCU: {pred_name}
OLASILIKLAR: Terk: %{probabilities[0]*100:.1f}, Devam: %{probabilities[1]*100:.1f}, Mezuniyet: %{probabilities[2]*100:.1f}

KARŞILAŞTIRMALAR:
{json.dumps(comparisons, ensure_ascii=False, indent=2)}

TOPLANAN VERİLER:
{json.dumps(st.session_state.collected_data, ensure_ascii=False, indent=2)}"""},
                            {"role": "user", "content": "Bu sonuçları bana anlat ve öneriler ver."}
                        ]

                        analysis_text = chat_with_llm(analysis_messages)
                        _, clean_analysis = extract_data_from_response(analysis_text)

                        final_response = clean_analysis + result_text
                        st.markdown(final_response)
                        st.session_state.messages.append({"role": "assistant", "content": final_response})
                        st.session_state.chat_history.append({"role": "user", "content": prompt})
                        st.session_state.chat_history.append({"role": "assistant", "content": final_response})

                    else:
                        clean_response = clean_response.replace("ANALIZ_HAZIR", "").strip()
                        st.markdown(clean_response)
                        st.session_state.messages.append({"role": "assistant", "content": clean_response})
                        st.session_state.chat_history.append({"role": "user", "content": prompt})
                        st.session_state.chat_history.append({"role": "assistant", "content": clean_response})

                except Exception as e:
                    error_msg = f"⚠️ Bir hata oluştu: {str(e)}"
                    st.markdown(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
