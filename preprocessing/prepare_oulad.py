from preprocessing.common import (
    OULAD_KEY_COLS,
    OULAD_TARGET_NAMES,
    apply_oulad_target_mapping,
    encode_oulad_categories,
    ensure_output_dir,
    fill_oulad_missing_values,
    load_oulad_tables,
    merge_oulad_tables,
)


def main():
    ensure_output_dir()
    tables = load_oulad_tables()
    mapped_student_info = apply_oulad_target_mapping(tables["studentInfo"])

    print("=" * 70)
    print("  OULAD — VERİ HAZIRLAMA (7 Tablo → 1 Dataset)")
    print("=" * 70)
    for name, frame in tables.items():
        print(f"\n  {name:<20}: {frame.shape}")

    print("\n" + "=" * 70)
    print("  HEDEF DEĞİŞKEN DÖNÜŞÜMÜ")
    print("=" * 70)
    print(f"\n  Orijinal (4 sınıf):")
    print(f"  {tables['studentInfo']['final_result'].value_counts().to_dict()}")
    print(f"\n  Yeni (3 sınıf):")
    for val in [0, 1, 2]:
        cnt = (mapped_student_info["target"] == val).sum()
        pct = cnt / len(mapped_student_info) * 100
        print(f"    {OULAD_TARGET_NAMES[val]}: {cnt} (%{pct:.1f})")

    print("\n" + "=" * 70)
    print("  ASSESSMENT ÖZELLİKLERİ")
    print("=" * 70)
    print("  Assessment ve VLE özellikleri yardımcı fonksiyonlarla oluşturuluyor.")

    print("\n" + "=" * 70)
    print("  TABLOLARI BİRLEŞTİR")
    print("=" * 70)
    merged = merge_oulad_tables(tables)
    print(f"  Birleştirme sonrası: {merged.shape[0]} satır × {merged.shape[1]} sütun")

    filled = fill_oulad_missing_values(merged)
    print(f"  Eksik veri sonrası: {filled.isnull().sum().sum()} eksik")

    print("\n" + "=" * 70)
    print("  KAYIT ÖZELLİKLERİ")
    print("=" * 70)
    print("  Kayıt özellikleri: date_registration")
    print("  NOT: 'unregistered' çıkarıldı (hedefe çok yakın özellik — target leakage riski)")

    print("\n" + "=" * 70)
    print("  ENCODING")
    print("=" * 70)
    encoded, encodings = encode_oulad_categories(filled)
    print(f"  Kategorik sütunlar: {list(encodings.keys())}")
    for col, mapping in encodings.items():
        print(f"    {col}: {mapping}")

    output_path = "preprocessing/oulad_processed.csv"
    encoded.to_csv(output_path, index=False)

    print("\n" + "=" * 70)
    print("  SONUÇ")
    print("=" * 70)
    print("  NOT: Normalizasyon ve MI feature selection modeling aşamasında")
    print("       train/test split sonrası yapılacak (data leakage önlemi).")
    print(f"  Final veri seti: {encoded.shape[0]} satır × {encoded.shape[1]} sütun")
    print(f"  Özellik sayısı: {encoded.shape[1] - 1}")
    print(f"  Hedef: target (0=Withdrawn, 1=Fail, 2=Pass)")
    print(f"\n  Sınıf dağılımı:")
    for val in [0, 1, 2]:
        cnt = (encoded["target"] == val).sum()
        pct = cnt / len(encoded) * 100
        print(f"    {OULAD_TARGET_NAMES[val]}: {cnt} (%{pct:.1f})")
    print(f"\n  Kaydedildi: {output_path}")

    print("\n" + "=" * 70)
    print("  VERİ HAZIRLAMA TAMAMLANDI")
    print("=" * 70)


if __name__ == "__main__":
    main()
