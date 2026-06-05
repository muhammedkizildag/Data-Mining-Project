from pathlib import Path
from zipfile import ZipFile

import pandas as pd
from sklearn.preprocessing import LabelEncoder


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PREPROCESSING_DIR = PROJECT_ROOT / "preprocessing"
DATASETS_DIR = PROJECT_ROOT / "datasets"
OULAD_DIR = DATASETS_DIR / "oulad"
DROPOUT_DATA_PATH = DATASETS_DIR / "dropout_academic_success" / "data.csv"

OULAD_TARGET_MAP = {
    "Withdrawn": 0,
    "Fail": 1,
    "Pass": 2,
    "Distinction": 2,
}
OULAD_TARGET_NAMES = {0: "Withdrawn", 1: "Fail", 2: "Pass"}
OULAD_KEY_COLS = ["code_module", "code_presentation", "id_student"]
OULAD_TOP_ACTIVITIES = [
    "resource",
    "oucontent",
    "url",
    "forumng",
    "quiz",
    "subpage",
    "homepage",
    "questionnaire",
    "page",
]
OULAD_REQUIRED_FILES = [
    "studentInfo.csv",
    "assessments.csv",
    "studentAssessment.csv",
    "vle.csv",
    "studentVle.csv",
    "studentRegistration.csv",
    "courses.csv",
]


def ensure_output_dir():
    PREPROCESSING_DIR.mkdir(parents=True, exist_ok=True)


def load_dropout_dataset():
    df = pd.read_csv(DROPOUT_DATA_PATH, sep=";")
    df.columns = df.columns.str.strip()
    return df


def encode_dropout_target(df):
    encoded = df.copy()
    encoder = LabelEncoder()
    encoded["Target_encoded"] = encoder.fit_transform(encoded["Target"])
    target_mapping = dict(
        zip(encoder.classes_, encoder.transform(encoder.classes_))
    )
    return encoded, target_mapping


def build_dropout_processed_frame(df):
    encoded, target_mapping = encode_dropout_target(df)
    X = encoded.drop(["Target", "Target_encoded"], axis=1)
    y = encoded["Target_encoded"]
    final_df = pd.concat([X, y.reset_index(drop=True)], axis=1).rename(
        columns={"Target_encoded": "Target"}
    )
    return final_df, target_mapping


def load_oulad_tables(base_dir=OULAD_DIR):
    ensure_oulad_tables_available(base_dir)
    return {
        "studentInfo": pd.read_csv(base_dir / "studentInfo.csv"),
        "assessments": pd.read_csv(base_dir / "assessments.csv"),
        "studentAssessment": pd.read_csv(base_dir / "studentAssessment.csv"),
        "vle": pd.read_csv(base_dir / "vle.csv"),
        "studentVle": pd.read_csv(base_dir / "studentVle.csv"),
        "studentRegistration": pd.read_csv(base_dir / "studentRegistration.csv"),
        "courses": pd.read_csv(base_dir / "courses.csv"),
    }


def ensure_oulad_tables_available(base_dir=OULAD_DIR):
    missing_files = [name for name in OULAD_REQUIRED_FILES if not (base_dir / name).exists()]
    if not missing_files:
        return

    archive_path = base_dir / "oulad.zip"
    if not archive_path.exists():
        missing = ", ".join(missing_files)
        raise FileNotFoundError(
            f"OULAD dosyalari eksik: {missing}. Ayrica {archive_path} bulunamadi."
        )

    with ZipFile(archive_path) as archive:
        archive.extractall(base_dir)


def apply_oulad_target_mapping(student_info):
    mapped = student_info.copy()
    mapped["target"] = mapped["final_result"].map(OULAD_TARGET_MAP)
    return mapped


def build_oulad_assessment_features(student_assessment, assessments):
    sa = student_assessment.merge(assessments, on="id_assessment", how="left")
    sa_agg = sa.groupby(OULAD_KEY_COLS).agg(
        avg_score=("score", "mean"),
        std_score=("score", "std"),
        min_score=("score", "min"),
        max_score=("score", "max"),
        num_assessments=("score", "count"),
        num_missing_score=("score", lambda x: x.isnull().sum()),
    ).reset_index()
    sa_agg["std_score"] = sa_agg["std_score"].fillna(0)

    for atype in ["TMA", "CMA", "Exam"]:
        subset = sa[sa["assessment_type"] == atype]
        type_agg = subset.groupby(OULAD_KEY_COLS).agg(
            **{
                f"avg_score_{atype}": ("score", "mean"),
                f"num_{atype}": ("score", "count"),
            }
        ).reset_index()
        sa_agg = sa_agg.merge(type_agg, on=OULAD_KEY_COLS, how="left")

    sa_agg = sa_agg.fillna(0)

    submit = sa.dropna(subset=["date_submitted", "date"]).copy()
    submit["submit_delay"] = submit["date_submitted"] - submit["date"]
    delay_agg = submit.groupby(OULAD_KEY_COLS).agg(
        avg_submit_delay=("submit_delay", "mean"),
        late_submissions=("submit_delay", lambda x: (x > 0).sum()),
        early_submissions=("submit_delay", lambda x: (x <= 0).sum()),
    ).reset_index()
    sa_agg = sa_agg.merge(delay_agg, on=OULAD_KEY_COLS, how="left")
    return sa_agg.fillna(0)


def build_oulad_vle_features(student_vle, vle):
    sv = student_vle.merge(vle[["id_site", "activity_type"]], on="id_site", how="left")
    vle_total = sv.groupby(OULAD_KEY_COLS).agg(
        total_clicks=("sum_click", "sum"),
        total_vle_days=("date", "nunique"),
        avg_daily_clicks=("sum_click", "mean"),
        num_distinct_activities=("id_site", "nunique"),
    ).reset_index()

    for act in OULAD_TOP_ACTIVITIES:
        act_sub = sv[sv["activity_type"] == act]
        act_agg = act_sub.groupby(OULAD_KEY_COLS).agg(
            **{f"clicks_{act}": ("sum_click", "sum")}
        ).reset_index()
        vle_total = vle_total.merge(act_agg, on=OULAD_KEY_COLS, how="left")

    return vle_total.fillna(0)


def merge_oulad_tables(tables):
    student_info = apply_oulad_target_mapping(tables["studentInfo"])
    sa_agg = build_oulad_assessment_features(
        tables["studentAssessment"], tables["assessments"]
    )
    vle_total = build_oulad_vle_features(tables["studentVle"], tables["vle"])
    reg_features = tables["studentRegistration"][
        OULAD_KEY_COLS + ["date_registration"]
    ].copy()
    course_features = tables["courses"].rename(
        columns={"module_presentation_length": "course_length"}
    )

    df = student_info.copy()
    df = df.merge(sa_agg, on=OULAD_KEY_COLS, how="left")
    df = df.merge(vle_total, on=OULAD_KEY_COLS, how="left")
    df = df.merge(reg_features, on=OULAD_KEY_COLS, how="left")
    df = df.merge(
        course_features, on=["code_module", "code_presentation"], how="left"
    )
    return df


def fill_oulad_missing_values(df):
    filled = df.copy()
    numeric_fill_cols = [
        c
        for c in filled.columns
        if filled[c].dtype in ["float64", "int64"] and c != "target"
    ]
    for col in numeric_fill_cols:
        if filled[col].isnull().sum() > 0:
            filled[col] = filled[col].fillna(0)

    if filled["imd_band"].isnull().any():
        filled["imd_band"] = filled["imd_band"].fillna(filled["imd_band"].mode()[0])
    return filled


def encode_oulad_categories(df):
    encoded = df.copy()
    drop_cols = ["code_module", "code_presentation", "id_student", "final_result"]
    encoded = encoded.drop(columns=drop_cols)
    cat_cols = encoded.select_dtypes(include="object").columns.tolist()

    encodings = {}
    for col in cat_cols:
        encoder = LabelEncoder()
        encoded[col] = encoder.fit_transform(encoded[col])
        encodings[col] = dict(
            zip(encoder.classes_, encoder.transform(encoder.classes_))
        )
    return encoded, encodings
