from preprocessing.common import (
    build_dropout_processed_frame,
    ensure_output_dir,
    load_dropout_dataset,
)


def main():
    ensure_output_dir()
    df = load_dropout_dataset()
    final_df, target_mapping = build_dropout_processed_frame(df)
    output_path = "preprocessing/dropout_processed.csv"

    print("=" * 60)
    print("  VERİ ÖN İŞLEME — Dropout UCI")
    print("=" * 60)

    print("\n--- 1. Genel Kontrol ---")
    print(f"  Boyut: {df.shape[0]} satır × {df.shape[1]} sütun")
    print(f"  Eksik veri: {df.isnull().sum().sum()} (yok)")

    print("\n--- 2. Hedef Değişken ---")
    print("  Orijinal dağılım:")
    for val, cnt in df["Target"].value_counts().items():
        print(f"    {val}: {cnt} ({cnt / len(df) * 100:.1f}%)")
    print(f"\n  Encoding: {target_mapping}")

    print("\n--- 3. Özellikler ---")
    print(f"  Toplam özellik sayısı: {final_df.shape[1] - 1}")

    print("\n" + "=" * 60)
    print("  SONUÇ")
    print("=" * 60)
    print("  NOT: Normalizasyon ve MI feature selection modeling aşamasında")
    print("       train/test split sonrası yapılacak (data leakage önlemi).")
    print(f"\n  Final veri seti: {final_df.shape[0]} satır × {final_df.shape[1]} sütun")
    print(f"  Özellik sayısı: {final_df.shape[1] - 1}")
    print(f"  Hedef değişken: Target ({target_mapping})")

    final_df.to_csv(output_path, index=False)
    print(f"\n  Kaydedildi: {output_path}")


if __name__ == "__main__":
    main()
