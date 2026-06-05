import json
import re


SEM_FALLBACK_MAP = {
    "Curricular units 2nd sem (enrolled)": "Curricular units 1st sem (enrolled)",
    "Curricular units 2nd sem (evaluations)": "Curricular units 1st sem (evaluations)",
    "Curricular units 2nd sem (approved)": "Curricular units 1st sem (approved)",
    "Curricular units 2nd sem (grade)": "Curricular units 1st sem (grade)",
}

COMMON_TYPOS = {
    "curicular": "curricular",
    "curriclar": "curricular",
    "curiclar": "curricular",
    "scholarhip": "scholarship",
    "enroled": "enrolled",
    "enrollement": "enrollment",
}

GRADE_FEATURES = {
    "Curricular units 1st sem (grade)",
    "Curricular units 2nd sem (grade)",
}


def normalize_lookup_key(value):
    normalized = value.lower().strip()
    translation_table = str.maketrans(
        {
            "ç": "c",
            "ğ": "g",
            "ı": "i",
            "ö": "o",
            "ş": "s",
            "ü": "u",
        }
    )
    return normalized.translate(translation_table)


def build_tr_to_en_map(feature_config):
    mapping = {}
    for feat_name, cfg in feature_config.items():
        tr_name = cfg.get("tr", "")
        mapping[normalize_lookup_key(tr_name)] = feat_name
        mapping[normalize_lookup_key(feat_name)] = feat_name

    mapping.update(
        {
            "cinsiyet": "Gender",
            "burs durumu": "Scholarship holder",
            "burslu": "Scholarship holder",
            "kayit yasi": "Age at enrollment",
            "yas": "Age at enrollment",
            "1. donem alinan ders": "Curricular units 1st sem (enrolled)",
            "1. donem gecilen ders": "Curricular units 1st sem (approved)",
            "1. donem not ortalamasi": "Curricular units 1st sem (grade)",
            "1. donem not": "Curricular units 1st sem (grade)",
            "1. donem girilen sinav": "Curricular units 1st sem (evaluations)",
            "2. donem alinan ders": "Curricular units 2nd sem (enrolled)",
            "2. donem gecilen ders": "Curricular units 2nd sem (approved)",
            "2. donem not ortalamasi": "Curricular units 2nd sem (grade)",
            "2. donem not": "Curricular units 2nd sem (grade)",
            "2. donem girilen sinav": "Curricular units 2nd sem (evaluations)",
            "medeni durum": "Marital status",
            "basvuru turu": "Application mode",
            "tercih sirasi": "Application order",
            "bolum": "Course",
            "onceki egitim": "Previous qualification",
            "onceki egitim notu": "Previous qualification (grade)",
            "anne egitim duzeyi": "Mother's qualification",
            "baba egitim duzeyi": "Father's qualification",
            "anne meslegi": "Mother's occupation",
            "baba meslegi": "Father's occupation",
            "universite giris puani": "Admission grade",
            "giris puani": "Admission grade",
        }
    )
    return mapping


def build_categorical_value_map(feature_config):
    mapping = {}
    for feat_name, cfg in feature_config.items():
        if cfg.get("type") == "categorical" and "options" in cfg:
            for option_name, option_val in cfg["options"].items():
                mapping[(feat_name, option_name.lower())] = option_val

    mapping.update(
        {
            ("Gender", "kadın"): 0,
            ("Gender", "kız"): 0,
            ("Gender", "female"): 0,
            ("Gender", "erkek"): 1,
            ("Gender", "male"): 1,
            ("Scholarship holder", "evet"): 1,
            ("Scholarship holder", "hayır"): 0,
            ("Scholarship holder", "var"): 1,
            ("Scholarship holder", "yok"): 0,
        }
    )
    return mapping


def build_system_prompt(feature_config, feature_order):
    feature_info = []
    for feat, cfg in feature_config.items():
        if cfg["priority"] in ("essential", "high"):
            if cfg["type"] == "categorical":
                opts = ", ".join(cfg["options"].keys())
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

NOT SİSTEMİ:
- Öğrencinin söylediği notu OLDUĞU GİBİ [DATA] etiketine yaz. Dönüşüm yapma.
- Öğrenci "4 üzerinden 2.8" derse → 2.8 yaz
- Öğrenci "100 üzerinden 85" derse → 85 yaz
- Öğrenci "notum 14" derse → 14 yaz (zaten 20'lik olabilir)
- Python tarafı notu otomatik tanıyıp dönüştürür, sen sadece ham değeri aktar.
- Öğrenciye not sistemi veya formül gösterme, sadece "Anladım" de.

YKS PUANI ve LİSE ORTALAMASI:
- Öğrenci YKS puanını veya lise ortalamasını SÖYLEDİĞİ GİBİ [DATA] etiketine yaz. Dönüşüm yapma.
- Python tarafı otomatik dönüştürür, sen ham değeri aktar.
- Örnek: "YKS'den 380 aldım" → [DATA: {{"Admission grade": 380}}]
- Örnek: "Lise ortalamam 85" → [DATA: {{"Previous qualification (grade)": 85}}]

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
- Notları ve puanları öğrencinin söylediği gibi yaz, dönüşüm yapma. Python otomatik dönüştürür.
- Özellik adlarını HARF HARF aşağıdaki listeden kopyala. Yazım hatası yapma (örn. "Curicular" YANLIŞ, "Curricular" DOĞRU).

Format:
[DATA: {{"özellik_adı": sayısal_değer, ...}}]

Örnek: Öğrenci "2. dönem 5 ders aldım 3ünü geçtim notum 4 üzerinden 2.4" derse:
[DATA: {{"Curricular units 2nd sem (enrolled)": 5, "Curricular units 2nd sem (approved)": 3, "Curricular units 2nd sem (grade)": 2.4}}]

DOĞRU özellik adları listesi (bunları birebir kullan):
{json.dumps(feature_order, ensure_ascii=False)}
"""


def fix_feature_name(name, feature_order, tr_to_en_map):
    if name in feature_order:
        return name

    name_lower = normalize_lookup_key(name)
    if name_lower in tr_to_en_map:
        return tr_to_en_map[name_lower]

    for correct in feature_order:
        if correct.lower() == name_lower:
            return correct

    fixed_name = name_lower
    for typo, correction in COMMON_TYPOS.items():
        fixed_name = fixed_name.replace(typo, correction)

    if fixed_name != name_lower:
        for correct in feature_order:
            if correct.lower() == fixed_name:
                return correct

    for correct in feature_order:
        correct_words = set(correct.lower().split())
        name_words = set(name_lower.split())
        common = len(correct_words & name_words)
        if common >= 2 and common >= len(correct_words) - 1:
            return correct

    return None


def resolve_categorical_value(feature_name, value, categorical_value_map):
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        normalized_value = normalize_lookup_key(value)
        for candidate, candidate_value in categorical_value_map.items():
            candidate_feature, candidate_label = candidate
            if candidate_feature != feature_name:
                continue
            if normalize_lookup_key(candidate_label) == normalized_value:
                return candidate_value
    return None


def auto_convert_grade(feature_name, value):
    if feature_name not in GRADE_FEATURES:
        return value
    if value <= 4.0:
        return (value / 4.0) * 20.0
    if value > 20:
        return (value / 100.0) * 20.0
    return value


def auto_convert_turkish_scale(feature_name, value):
    if feature_name == "Admission grade" and value > 190:
        return 95.0 + ((value / 500.0) * 95.0)
    if feature_name == "Previous qualification (grade)" and 0 < value <= 100:
        return 95.0 + ((value / 100.0) * 95.0)
    return value


def get_model_numeric_range(feature_name, feature_config):
    if feature_name == "Admission grade":
        return (95.0, 190.0)
    if feature_name == "Previous qualification (grade)":
        return (95.0, 190.0)
    if feature_name in GRADE_FEATURES:
        return (0.0, 20.0)

    cfg = feature_config.get(feature_name, {})
    value_range = cfg.get("range")
    if value_range is None:
        return None
    return (float(value_range[0]), float(value_range[1]))


def validate_feature_value(feature_name, value, feature_config):
    cfg = feature_config.get(feature_name, {})
    if cfg.get("type") == "categorical":
        valid_values = set(cfg.get("options", {}).values())
        return value if value in valid_values else None

    value_range = get_model_numeric_range(feature_name, feature_config)
    if value_range is None:
        return value

    low, high = value_range
    if value < low:
        return low
    if value > high:
        return high
    return value


def clean_extracted_data(
    raw_data,
    feature_order,
    feature_config,
    tr_to_en_map,
    categorical_value_map,
):
    cleaned = {}
    for key, value in raw_data.items():
        if value is None:
            continue
        if isinstance(value, str) and value.lower() in ("null", "none", ""):
            continue

        fixed_key = fix_feature_name(key, feature_order, tr_to_en_map)
        if fixed_key is None:
            continue

        if isinstance(value, str):
            cat_val = resolve_categorical_value(fixed_key, value, categorical_value_map)
            if cat_val is not None:
                valid_cat = validate_feature_value(fixed_key, float(cat_val), feature_config)
                if valid_cat is not None:
                    cleaned[fixed_key] = float(valid_cat)
                continue

        try:
            num_val = float(value)
        except (ValueError, TypeError):
            continue

        num_val = auto_convert_grade(fixed_key, num_val)
        num_val = auto_convert_turkish_scale(fixed_key, num_val)
        num_val = validate_feature_value(fixed_key, num_val, feature_config)
        if num_val is not None:
            cleaned[fixed_key] = float(num_val)

    return cleaned


def extract_data_from_response(
    response_text,
    feature_order,
    feature_config,
    tr_to_en_map,
    categorical_value_map,
):
    data_match = re.search(r"\[DATA:\s*(\{.*?\})\]", response_text, re.DOTALL)
    if data_match:
        try:
            data = json.loads(data_match.group(1))
            cleaned = clean_extracted_data(
                data,
                feature_order,
                feature_config,
                tr_to_en_map,
                categorical_value_map,
            )
            clean_text = re.sub(r"\[DATA:.*?\]", "", response_text, flags=re.DOTALL).strip()
            return cleaned if cleaned else None, clean_text
        except json.JSONDecodeError:
            pass

    clean_text = re.sub(r"\[DATA:\s*\]", "", response_text).strip()
    return None, clean_text


def clean_non_turkish(text):
    text = re.sub(r"[一-鿿぀-ゟ゠-ヿ가-힯]+", "", text)
    text = re.sub(r"  +", " ", text)
    return text.strip()


def check_analysis_ready(response_text):
    return "ANALIZ_HAZIR" in response_text
