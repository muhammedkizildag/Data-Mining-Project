import streamlit as st
import joblib
import json
import pandas as pd
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
    with open(os.path.join(BASE_DIR, "feature_config.json"), "r", encoding="utf-8") as f:
        features = json.load(f)
    with open(os.path.join(BASE_DIR, "reference_stats.json"), "r", encoding="utf-8") as f:
        ref_stats = json.load(f)
    return features, ref_stats

pipeline = load_model()
feature_config, reference_stats = load_configs()

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
def predict_student(collected_data):
    input_values = []
    for feat in FEATURE_ORDER:
        if feat in collected_data:
            raw_val = float(collected_data[feat])
        else:
            raw_val = float(feature_config[feat]['default'])
        input_values.append(raw_val)

    X = pd.DataFrame([input_values], columns=FEATURE_ORDER)
    prediction = pipeline.predict(X)[0]
    probabilities = pipeline.predict_proba(X)[0]
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

KESİN DİL KURALLARI:
- SADECE Türkçe konuş. Tek bir İngilizce, Çince veya başka dilde kelime bile kullanma.
- Yasak örnekler: "sometimes", "information", "improve", "majority" vb.
- Sadece Türkçe alfabedeki harfleri kullan (a-z, ç, ğ, ı, ö, ş, ü).
- Doğal, samimi ve sıcak bir Türkçe kullan. Üniversite öğrencisiyle konuşuyorsun, resmi olma.

GÖREVİN:
Öğrenciyle doğal bir sohbet kurarak akademik durumunu anlamak ve ona yardımcı olmak.

TOPLANMASI GEREKEN BİLGİLER (öncelik sırasıyla):
{features_text}

NOT SİSTEMİ DÖNÜŞÜMÜ (GİZLİ — öğrenciye gösterme):
- Türkiye'de üniversiteler genelde 4'lük not sistemi kullanır.
- Sistem içinde 20'lik ölçek kullanılır. Dönüşüm formülleri:
  * 4'lük sistem: (not / 4.0) × 20 → Örnek: 2.4 → 12.0
  * 100'lük sistem: (not / 100) × 20 → Örnek: 70 → 14.0
- Öğrenci hangi sistemi kullandığını belirtmezse 4'lük varsay.
- Bu dönüşümü SADECE [DATA] etiketinde yap, öğrenciye "20'lik sisteme çeviriyorum" gibi şeyler SÖYLEME.
- Öğrenciye sadece "Anladım, notun şuymuş" de, formül gösterme.

YAŞ ve KAYIT YAŞI FARKI:
- "Age at enrollment" = öğrencinin ÜNİVERSİTEYE İLK KAYIT OLDUĞU YAŞ, şu anki yaşı değil.
- Öğrenci "22 yaşındayım, 3. sınıfım" derse → kayıt yaşı yaklaşık 22-3 = 19 olabilir.
- Öğrenci "20 yaşındayım" derse ve sınıfını bilmiyorsan → "Üniversiteye kaç yaşında başladın?" diye sor.
- Öğrencinin şimdiki yaşını kayıt yaşı olarak YAZMA.

DERS SAYISI HESAPLAMA:
- Öğrenci "alttan 3 dersim var" derse: kalan ders = 3, geçilen = alınan - 3.
- "6 ders aldım alttan 2 kaldı" → alınan: 6, geçilen: 4.
- Öğrenci sadece toplam ve kalan veriyorsa geçileni kendin hesapla.
- "evaluations" (girilen sınav sayısı) bilgisini de sormaya çalış: "Bu derslerde kaç sınava girdin?" gibi doğal bir şekilde.

SOHBET KURALLARI:
1. Soruları doğal sohbet akışında sor. Anket gibi madde madde sıralama.
2. Öğrenci bilgi verdiğinde KISA bir şekilde anladığını göster (1 cümle).
3. Birden fazla bilgiyi tek mesajda toplamaya çalış ama zorla değil.
4. Öğrenci bilmiyorsa veya hatırlamıyorsa geç, başka bir konuya yönel. Aynı soruyu tekrarlama.
5. Dolaylı cevaplardan çıkarım yap ama emin olamadığın bilgiyi DATA'ya yazma.
6. Her iki dönem için alınan ders, geçilen ders ve not ortalaması toplandığında ANALIZ_HAZIR yaz.
7. ASLA "kalırsın", "başarısız olursun", "bırakma riskin var" gibi olumsuz kesin yargılar kullanma.
8. Her zaman yapıcı, destekleyici ve motive edici ol.
9. Kısa ve öz cevaplar ver. Maksimum 2-3 cümle. Uzun paragraflar yazma.
10. Öğrenci İngilizce yazarsa: "Ben Türkçe konuşuyorum, Türkçe yazabilir misin?" de.

[DATA] ETİKETİ KURALLARI:
- Sadece KESİN bilgi topladığında [DATA] ekle. Belirsiz bilgi varsa ekleme.
- Hiç bilgi toplanmadıysa [DATA] etiketi KOYMA. Boş [DATA: {{}}] veya null değerli [DATA] yasak.
- Değerler HER ZAMAN sayı olmalı (int veya float). String veya null YASAK.
- Not dönüşümünü yaptıktan sonraki 20'lik değeri yaz.
- Özellik adlarını HARF HARF aşağıdaki listeden kopyala. Yazım hatası yapma (örn. "Curicular" YANLIŞ, "Curricular" DOĞRU).

Format:
[DATA: {{"özellik_adı": sayısal_değer, ...}}]

Örnek: Öğrenci "2. dönem 5 ders aldım 3ünü geçtim notum 4 üzerinden 2.4" derse:
[DATA: {{"Curricular units 2nd sem (enrolled)": 5, "Curricular units 2nd sem (approved)": 3, "Curricular units 2nd sem (grade)": 12.0}}]

DOĞRU özellik adları listesi (bunları birebir kullan):
{json.dumps(FEATURE_ORDER, ensure_ascii=False)}
"""

TR_TO_EN_MAP = {}
for feat_name, cfg in feature_config.items():
    tr_name = cfg.get('tr', '')
    TR_TO_EN_MAP[tr_name.lower()] = feat_name
    TR_TO_EN_MAP[feat_name.lower()] = feat_name

TR_TO_EN_MAP.update({
    "cinsiyet": "Gender",
    "burs durumu": "Scholarship holder",
    "burslu": "Scholarship holder",
    "kayıt yaşı": "Age at enrollment",
    "yaş": "Age at enrollment",
    "1. dönem alınan ders": "Curricular units 1st sem (enrolled)",
    "1. dönem geçilen ders": "Curricular units 1st sem (approved)",
    "1. dönem not ortalaması": "Curricular units 1st sem (grade)",
    "1. dönem not": "Curricular units 1st sem (grade)",
    "1. dönem girilen sınav": "Curricular units 1st sem (evaluations)",
    "2. dönem alınan ders": "Curricular units 2nd sem (enrolled)",
    "2. dönem geçilen ders": "Curricular units 2nd sem (approved)",
    "2. dönem not ortalaması": "Curricular units 2nd sem (grade)",
    "2. dönem not": "Curricular units 2nd sem (grade)",
    "2. dönem girilen sınav": "Curricular units 2nd sem (evaluations)",
    "medeni durum": "Marital status",
    "başvuru türü": "Application mode",
    "tercih sırası": "Application order",
    "bölüm": "Course",
    "önceki eğitim": "Previous qualification",
    "önceki eğitim notu": "Previous qualification (grade)",
    "anne eğitim düzeyi": "Mother's qualification",
    "baba eğitim düzeyi": "Father's qualification",
    "anne mesleği": "Mother's occupation",
    "baba mesleği": "Father's occupation",
    "üniversite giriş puanı": "Admission grade",
    "giriş puanı": "Admission grade",
})

CATEGORICAL_VALUE_MAP = {}
for feat_name, cfg in feature_config.items():
    if cfg.get('type') == 'categorical' and 'options' in cfg:
        for option_name, option_val in cfg['options'].items():
            CATEGORICAL_VALUE_MAP[(feat_name, option_name.lower())] = option_val

CATEGORICAL_VALUE_MAP.update({
    ("Gender", "kadın"): 0, ("Gender", "kız"): 0, ("Gender", "female"): 0,
    ("Gender", "erkek"): 1, ("Gender", "male"): 1,
    ("Scholarship holder", "evet"): 1, ("Scholarship holder", "hayır"): 0,
    ("Scholarship holder", "var"): 1, ("Scholarship holder", "yok"): 0,
})

COMMON_TYPOS = {
    "curicular": "curricular",
    "curriclar": "curricular",
    "curiclar": "curricular",
    "scholarhip": "scholarship",
    "enroled": "enrolled",
    "enrollement": "enrollment",
}

def fix_feature_name(name):
    if name in FEATURE_ORDER:
        return name
    name_lower = name.lower().strip()
    if name_lower in TR_TO_EN_MAP:
        return TR_TO_EN_MAP[name_lower]
    for correct in FEATURE_ORDER:
        if correct.lower() == name_lower:
            return correct
    fixed_name = name_lower
    for typo, correction in COMMON_TYPOS.items():
        fixed_name = fixed_name.replace(typo, correction)
    if fixed_name != name_lower:
        for correct in FEATURE_ORDER:
            if correct.lower() == fixed_name:
                return correct
    for correct in FEATURE_ORDER:
        correct_words = set(correct.lower().split())
        name_words = set(name_lower.split())
        common = len(correct_words & name_words)
        if common >= 2 and common >= len(correct_words) - 1:
            return correct
    return None

def resolve_categorical_value(feature_name, value):
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        lookup = (feature_name, value.lower().strip())
        if lookup in CATEGORICAL_VALUE_MAP:
            return CATEGORICAL_VALUE_MAP[lookup]
    return None

GRADE_FEATURES = {
    "Curricular units 1st sem (grade)",
    "Curricular units 2nd sem (grade)",
}

def auto_convert_grade(feature_name, value):
    if feature_name in GRADE_FEATURES and value <= 4.0:
        return (value / 4.0) * 20.0
    return value

def clean_extracted_data(raw_data):
    cleaned = {}
    for key, value in raw_data.items():
        if value is None:
            continue
        if isinstance(value, str) and value.lower() in ('null', 'none', ''):
            continue
        fixed_key = fix_feature_name(key)
        if fixed_key is None:
            continue
        if isinstance(value, str):
            cat_val = resolve_categorical_value(fixed_key, value)
            if cat_val is not None:
                cleaned[fixed_key] = float(cat_val)
                continue
        try:
            num_val = float(value)
            num_val = auto_convert_grade(fixed_key, num_val)
            cleaned[fixed_key] = num_val
        except (ValueError, TypeError):
            continue
    return cleaned

def extract_data_from_response(response_text):
    import re
    data_match = re.search(r'\[DATA:\s*(\{.*?\})\]', response_text, re.DOTALL)
    if data_match:
        try:
            data = json.loads(data_match.group(1))
            cleaned = clean_extracted_data(data)
            clean_text = re.sub(r'\[DATA:.*?\]', '', response_text, flags=re.DOTALL).strip()
            return cleaned if cleaned else None, clean_text
        except json.JSONDecodeError:
            pass
    clean_text = re.sub(r'\[DATA:\s*\]', '', response_text).strip()
    return None, clean_text

def clean_non_turkish(text):
    import re
    text = re.sub(r'[一-鿿぀-ゟ゠-ヿ가-힯]+', '', text)
    text = re.sub(r'  +', ' ', text)
    return text.strip()

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
                            {"role": "system", "content": f"""Aşağıdaki analiz sonuçlarını öğrenciye YAPICI ve DESTEKLEYİCİ bir dille aktar.
SADECE Türkçe konuş, tek bir İngilizce veya yabancı dilde kelime kullanma.
Kesinlikle "kalırsın" veya "başarısız olursun" gibi ifadeler kullanma.
Somut ve uygulanabilir öneriler ver.
"Eğer şunu yaparsan, durumun iyileşir" gibi yapıcı senaryolar sun.
Kısa ve öz tut, 4-5 cümleyi geçme.

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

                        clean_analysis = clean_non_turkish(clean_analysis)
                        final_response = clean_analysis + result_text
                        st.markdown(final_response)
                        st.session_state.messages.append({"role": "assistant", "content": final_response})
                        st.session_state.chat_history.append({"role": "user", "content": prompt})
                        st.session_state.chat_history.append({"role": "assistant", "content": final_response})

                    else:
                        clean_response = clean_response.replace("ANALIZ_HAZIR", "").strip()
                        clean_response = clean_non_turkish(clean_response)
                        st.markdown(clean_response)
                        st.session_state.messages.append({"role": "assistant", "content": clean_response})
                        st.session_state.chat_history.append({"role": "user", "content": prompt})
                        st.session_state.chat_history.append({"role": "assistant", "content": clean_response})

                except Exception as e:
                    error_msg = f"⚠️ Bir hata oluştu: {str(e)}"
                    st.markdown(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
