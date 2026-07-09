import argparse
import os
import re
import sys
import warnings
from typing import Dict, List

import joblib
import matplotlib
import nltk
import numpy as np
import pandas as pd
import requests
from requests.auth import HTTPBasicAuth

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

warnings.filterwarnings("ignore")


ASSIGNMENT_GROUP_COL = "Assignment group"
CATEGORY_COL = "Category"
SUBCATEGORY_COL = "Subcategory"
SHORT_DESCRIPTION_COL = "Short description"
DESCRIPTION_COL = "Description"

MODEL_FILE = "model.pkl"
VECTORIZER_FILE = "vectorizer.pkl"
LABEL_ENCODER_FILE = "label_encoder.pkl"
CATEGORY_MODEL_FILE = "category_model.pkl"
CATEGORY_LABEL_ENCODER_FILE = "category_label_encoder.pkl"
SUBCATEGORY_MODEL_FILE = "subcategory_model.pkl"
SUBCATEGORY_LABEL_ENCODER_FILE = "subcategory_label_encoder.pkl"
SVD_FILE = "svd_transformer.pkl"
VECTORIZER_LEM_FILE = "vectorizer_lem.pkl"


SERVICENOW_FIELD_MAP: Dict[str, str] = {
    "Sys ID": "sys_id",
    "Number": "number",
    "Opened": "opened_at",
    "Short description": "short_description",
    "Caller": "caller_id",
    "Priority": "priority",
    "State": "state",
    "Category": "category",
    "Assignment group": "assignment_group",
    "Assigned to": "assigned_to",
    "Updated": "sys_updated_on",
    "Updated by": "sys_updated_by",
    "Opened by": "opened_by",
    "Created by": "sys_created_by",
    "Description": "description",
    "Active": "active",
    "Activity due": "activity_due",
    "Actual end": "calendar_stc",
    "Actual start": "work_start",
    "Additional assignee list": "additional_assignee_list",
    "Approval": "approval",
    "Approval history": "approval_history",
    "Approval set": "approval_set",
    "Business duration": "business_duration",
    "Business impact": "business_impact",
    "Business resolve time": "business_stc",
    "Caused by Change": "caused_by",
    "Change Request": "rfc",
    "Channel": "contact_type",
    "Child Incidents": "child_incidents",
    "Closed": "closed_at",
    "Closed by": "closed_by",
    "Comments": "comments",
    "Comments and Work notes": "comments_and_work_notes",
    "Company": "company",
    "Configuration item": "cmdb_ci",
    "Contract": "contract",
    "Correlation ID": "correlation_id",
    "Correlation display": "correlation_display",
    "Created": "sys_created_on",
    "Delivery plan": "delivery_plan",
    "Delivery task": "delivery_task",
    "Domain": "sys_domain",
    "Domain Path": "sys_domain_path",
    "Due date": "due_date",
    "Duration": "calendar_duration",
    "Effective number": "number",
    "Escalation": "escalation",
    "Expected start": "expected_start",
    "Follow up": "follow_up",
    "Group list": "group_list",
    "Impact": "impact",
    "Incident state": "incident_state",
    "Knowledge": "knowledge",
    "Last reopened at": "reopened_time",
    "Last reopened by": "reopened_by",
    "Location": "location",
    "Made SLA": "made_sla",
    "Notify": "notify",
    "On hold reason": "hold_reason",
    "Order": "order",
    "Origin": "origin_id",
    "Origin table": "origin_table",
    "Parent": "parent",
    "Parent Incident": "parent_incident",
    "Probable cause": "cause",
    "Problem": "problem_id",
    "Reassignment count": "reassignment_count",
    "Reopen count": "reopen_count",
    "Resolution code": "close_code",
    "Resolution notes": "close_notes",
    "Resolve time": "calendar_stc",
    "Resolved": "resolved_at",
    "Resolved by": "resolved_by",
    "SLA due": "sla_due",
    "Service": "business_service",
    "Service offering": "service_offering",
    "Severity": "severity",
    "Skills": "skills",
    "Subcategory": "subcategory",
    "Tags": "sys_tags",
    "Task type": "sys_class_name",
    "Time worked": "time_worked",
    "Transfer reason": "transfer_reason",
    "Universal Request": "universal_request",
    "Updates": "sys_mod_count",
    "Upon approval": "upon_approval",
    "Upon reject": "upon_reject",
    "Urgency": "urgency",
    "User input": "user_input",
    "Watch list": "watch_list",
    "Work notes": "work_notes",
    "Work notes list": "work_notes_list",
}


try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")
    nltk.download("stopwords")

try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet")
    nltk.download("omw-1.4")

from nltk.stem import PorterStemmer, WordNetLemmatizer

stemmer = PorterStemmer()
use_lemmatizer = False
lemmatizer = None

try:
    lemmatizer = WordNetLemmatizer()
    use_lemmatizer = True
    print("WordNet lemmatizer initialized successfully")
except LookupError:
    print("WordNet not available, using stemming only")
    use_lemmatizer = False


CUSTOM_STOPWORDS = set(ENGLISH_STOP_WORDS).union(
    {
        "incident",
        "service",
        "request",
        "ticket",
        "issue",
        "problem",
        "user",
        "need",
        "help",
        "please",
        "hi",
        "hello",
        "thanks",
        "regards",
        "team",
        "support",
        "access",
        "system",
        "application",
        "server",
        "error",
        "fix",
        "resolve",
        "solution",
        "work",
    }
)


def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s\.\-_]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def remove_stopwords(text):
    words = text.split()
    filtered_words = [word for word in words if word not in CUSTOM_STOPWORDS and len(word) > 1]
    return " ".join(filtered_words)


def stem_text(text):
    words = text.split()
    stemmed = [stemmer.stem(w) for w in words]
    return " ".join(stemmed)


def lemmatize_text(text):
    if use_lemmatizer and lemmatizer:
        words = text.split()
        lemmatized = [lemmatizer.lemmatize(w) for w in words]
        return " ".join(lemmatized)
    return text


def preprocess_text(text, use_lemmatization=False):
    text = clean_text(text)
    text = remove_stopwords(text)
    if use_lemmatization and use_lemmatizer:
        text = lemmatize_text(text)
    else:
        text = stem_text(text)
    return text


def get_service_now_credentials() -> tuple[str, str, str]:
    instance_url = os.getenv("SERVICENOW_INSTANCE_URL", "").rstrip("/")
    username = os.getenv("SERVICENOW_USERNAME", "")
    password = os.getenv("SERVICENOW_PASSWORD", "")

    if not instance_url:
        raise RuntimeError("SERVICENOW_INSTANCE_URL is required.")
    if not username:
        raise RuntimeError("SERVICENOW_USERNAME is required.")
    if not password:
        raise RuntimeError("SERVICENOW_PASSWORD is required.")

    return instance_url, username, password


def fetch_servicenow_incidents(limit: int = 1000, query: str = "") -> pd.DataFrame:
    instance_url, username, password = get_service_now_credentials()
    api_fields = sorted(set(SERVICENOW_FIELD_MAP.values()))
    url = f"{instance_url}/api/now/table/incident"
    rows: List[dict] = []
    offset = 0
    page_size = min(100, max(1, limit))

    while len(rows) < limit:
        params = {
            "sysparm_display_value": "true",
            "sysparm_exclude_reference_link": "true",
            "sysparm_fields": ",".join(api_fields),
            "sysparm_limit": min(page_size, limit - len(rows)),
            "sysparm_offset": offset,
        }
        if query:
            params["sysparm_query"] = query

        response = requests.get(
            url,
            params=params,
            auth=HTTPBasicAuth(username, password),
            headers={"Accept": "application/json"},
            timeout=30,
        )
        response.raise_for_status()
        batch = response.json().get("result", [])
        if not batch:
            break

        rows.extend(batch)
        offset += len(batch)

    renamed_rows = [
        {excel_col: row.get(api_col, "") for excel_col, api_col in SERVICENOW_FIELD_MAP.items()}
        for row in rows
    ]
    return pd.DataFrame(renamed_rows, columns=list(SERVICENOW_FIELD_MAP.keys()))


def load_file_dataset(path: str) -> pd.DataFrame:
    if path.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(path)
    return pd.read_csv(path)


def load_dataset(args: argparse.Namespace) -> pd.DataFrame:
    if args.source == "servicenow":
        print("Fetching incidents from ServiceNow...")
        return fetch_servicenow_incidents(limit=args.limit, query=args.query)

    possible_files = [
        args.file,
        "incidents.xlsx",
        "incidents_1.xlsx",
        "incidents.csv",
        "incidents_1.csv",
        "Incidents.xlsx",
        "Incidents.csv",
    ]

    for file_name in possible_files:
        try:
            df = load_file_dataset(file_name)
            print(f"Successfully loaded '{file_name}'")
            return df
        except FileNotFoundError:
            continue
        except Exception as exc:
            print(f"Error loading '{file_name}': {exc}")

    raise FileNotFoundError("Could not find an Excel or CSV incidents file.")


def train_optional_text_classifier(df, X, target_col, model_file, encoder_file):
    if target_col not in df.columns:
        print(f"\nSkipping {target_col} model: column is missing.")
        return

    target = df[target_col].replace("", pd.NA).dropna()
    if target.empty:
        print(f"\nSkipping {target_col} model: no usable values.")
        return

    normalized_target = df[target_col].replace("", pd.NA).astype("string").str.strip().str.lower()
    counts = normalized_target.dropna().value_counts()
    valid_values = counts[counts >= 10].index
    mask = normalized_target.isin(valid_values).to_numpy()

    if len(valid_values) < 2 or mask.sum() < 20:
        print(f"\nSkipping {target_col} model: not enough training data.")
        return

    label_encoder = LabelEncoder()
    y_target = label_encoder.fit_transform(normalized_target[mask])

    model = LogisticRegression(max_iter=2500, C=0.3)
    model.fit(X[mask], y_target)

    joblib.dump(model, model_file)
    joblib.dump(label_encoder, encoder_file)

    print(f"\nSaved {target_col} model: {model_file}")
    print(f"Saved {target_col} label encoder: {encoder_file}")


def calculate_metrics(y_true, predictions):
    return {
        "Accuracy": accuracy_score(y_true, predictions),
        "Precision": precision_score(y_true, predictions, average="weighted", zero_division=0),
        "Recall": recall_score(y_true, predictions, average="weighted", zero_division=0),
        "F1 Score": f1_score(y_true, predictions, average="weighted", zero_division=0),
    }


def plot_confusion_matrix(cm, labels, title):
    file_name = f"{title.replace(' ', '_')}.png"

    plt.figure(figsize=(8, 7))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title(title, fontsize=14)
    plt.colorbar()

    tick_marks = range(len(labels))
    plt.xticks(tick_marks, labels, rotation=45, fontsize=10)
    plt.yticks(tick_marks, labels, fontsize=10)

    threshold = cm.max() / 2 if cm.size else 0
    for i in range(len(labels)):
        for j in range(len(labels)):
            plt.text(
                j,
                i,
                format(cm[i, j], "d"),
                horizontalalignment="center",
                color="white" if cm[i, j] > threshold else "black",
                fontsize=11,
                fontweight="bold",
            )

    plt.ylabel("Actual", fontsize=12)
    plt.xlabel("Predicted", fontsize=12)
    plt.tight_layout()
    plt.savefig(file_name, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved confusion matrix image: {file_name}")


def train(args: argparse.Namespace) -> None:
    print("\n" + "=" * 60)
    print("SERVICENOW ASSIGNMENT GROUP PREDICTION MODEL")
    print("=" * 60)

    print("\nLoading dataset...")
    df = load_dataset(args)

    print("\n" + "=" * 60)
    print("DATASET INFORMATION")
    print("=" * 60)
    print(f"Rows : {len(df)}")
    print(f"Columns : {len(df.columns)}")

    required_cols = [ASSIGNMENT_GROUP_COL, SHORT_DESCRIPTION_COL, DESCRIPTION_COL]
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        print(f"\nERROR: Missing columns: {missing_cols}")
        print(f"Available columns: {df.columns.tolist()}")
        sys.exit(1)

    df[ASSIGNMENT_GROUP_COL] = df[ASSIGNMENT_GROUP_COL].astype(str).str.strip().str.title()

    df["text"] = (
        df[SHORT_DESCRIPTION_COL].fillna("").astype(str)
        + " "
        + df[DESCRIPTION_COL].fillna("").astype(str)
    )

    print("\n" + "=" * 60)
    print("DATA STATISTICS")
    print("=" * 60)
    print(f"Assignment Groups : {df[ASSIGNMENT_GROUP_COL].nunique()}")
    print(f"Missing Assignment Groups : {df[ASSIGNMENT_GROUP_COL].isna().sum()}")
    print(f"Duplicate Texts : {df['text'].duplicated().sum()}")
    print(f"Missing Short Descriptions : {df[SHORT_DESCRIPTION_COL].isna().sum()}")
    print(f"Missing Descriptions : {df[DESCRIPTION_COL].isna().sum()}")
    print(f"Missing Assignment Groups : {df[ASSIGNMENT_GROUP_COL].isna().sum()}")

    df = df.dropna(subset=[ASSIGNMENT_GROUP_COL])

    print("\nAssignment Group Distribution (Top 20):")
    group_counts = df[ASSIGNMENT_GROUP_COL].value_counts()
    print(group_counts.head(20))

    valid_groups = group_counts[group_counts >= 10].index
    df = df[df[ASSIGNMENT_GROUP_COL].isin(valid_groups)]
    df = df.reset_index(drop=True)
    print(f"\nAfter filtering rare groups: {df[ASSIGNMENT_GROUP_COL].nunique()} groups remain")

    print("\n" + "=" * 60)
    print("TEXT PREPROCESSING")
    print("=" * 60)

    print("\nCleaning text with enhanced preprocessing...")
    df["clean_text"] = df["text"].apply(lambda x: preprocess_text(x, use_lemmatization=False))

    if use_lemmatizer:
        print("Also creating lemmatized version...")
        df["clean_text_lem"] = df["text"].apply(lambda x: preprocess_text(x, use_lemmatization=True))
    else:
        df["clean_text_lem"] = df["clean_text"]

    print("Text cleaning completed.")

    print("\nSample cleaned text:")
    sample_df = df[["text", "clean_text"]].head(2)
    for _, row in sample_df.iterrows():
        print(f"\nOriginal: {row['text'][:100]}...")
        print(f"Cleaned:  {row['clean_text'][:100]}...")

    print("\nEncoding labels...")
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df[ASSIGNMENT_GROUP_COL])
    print(f"Number of unique assignment groups: {len(label_encoder.classes_)}")

    print("\nVectorizing text with TF-IDF...")
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 1),
        min_df=3,
        max_df=0.80,
        max_features=5000,
        sublinear_tf=True,
    )

    X = vectorizer.fit_transform(df["clean_text"])
    print("TF-IDF matrix shape :", X.shape)

    vectorizer_lem = None
    if use_lemmatizer:
        vectorizer_lem = TfidfVectorizer(
            ngram_range=(1, 1),
            min_df=3,
            max_df=0.80,
            max_features=5000,
            sublinear_tf=True,
        )
        X_lem = vectorizer_lem.fit_transform(df["clean_text_lem"])
        print("TF-IDF matrix shape (lemmatized) :", X_lem.shape)

        from scipy.sparse import hstack

        X = hstack([X, X_lem])
        print("Combined feature matrix shape:", X.shape)

    print("\nReducing features for KNN...")
    n_features = X.shape[1]
    n_components = min(20, n_features - 1)

    svd = TruncatedSVD(n_components=n_components, random_state=42)
    X_reduced = svd.fit_transform(X)
    print("Reduced feature matrix shape :", X_reduced.shape)

    print("\nCreating train/test split...")
    (
        X_train,
        X_test,
        X_train_reduced,
        X_test_reduced,
        y_train,
        y_test,
        y_train_reduced,
        y_test_reduced,
    ) = train_test_split(
        X,
        X_reduced,
        y,
        y,
        test_size=0.40,
        random_state=123,
        stratify=y,
    )

    print("Training samples :", X_train.shape[0])
    print("Testing samples :", X_test.shape[0])
    print("KNN training samples :", X_train_reduced.shape[0])
    print("KNN testing samples :", X_test_reduced.shape[0])

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, C=0.04),
        "Naive Bayes": MultinomialNB(alpha=8.0),
        "Random Forest": RandomForestClassifier(n_estimators=40, max_depth=7, random_state=42),
        "SVM (RBF)": SVC(kernel="rbf", C=1, gamma=0.012),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=20,
            learning_rate=0.05,
            max_depth=1,
            random_state=42,
        ),
        "AdaBoost": AdaBoostClassifier(n_estimators=30, learning_rate=1.0, random_state=42),
        "Hybrid Voting": VotingClassifier(
            estimators=[
                ("lr", LogisticRegression(max_iter=1000, C=0.04)),
                ("nb", MultinomialNB(alpha=8.0)),
                ("svm", SVC(kernel="rbf", C=1, gamma=0.001)),
            ],
            voting="hard",
        ),
    }

    results = []
    trained_models = {}
    model_predictions = {}

    print("\n" + "=" * 60)
    print("         MODEL COMPARISON")
    print("=" * 60 + "\n")

    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model

        print(f"Evaluating {name}...")
        predictions = model.predict(X_test)
        model_predictions[name] = predictions

        metrics = calculate_metrics(y_test, predictions)
        results.append(
            {
                "Model": name,
                "Accuracy": round(metrics["Accuracy"], 4),
                "Precision": round(metrics["Precision"], 4),
                "Recall": round(metrics["Recall"], 4),
                "F1 Score": round(metrics["F1 Score"], 4),
            }
        )

        print(f"Completed {name}\n")

    print("Training KNN...")
    knn = KNeighborsClassifier(n_neighbors=150, weights="uniform")

    knn.fit(X_train_reduced, y_train_reduced)
    trained_models["KNN"] = knn

    print("Evaluating KNN...")
    knn_predictions = knn.predict(X_test_reduced)
    model_predictions["KNN"] = knn_predictions

    knn_metrics = calculate_metrics(y_test_reduced, knn_predictions)
    results.append(
        {
            "Model": "KNN",
            "Accuracy": round(knn_metrics["Accuracy"], 4),
            "Precision": round(knn_metrics["Precision"], 4),
            "Recall": round(knn_metrics["Recall"], 4),
            "F1 Score": round(knn_metrics["F1 Score"], 4),
        }
    )
    print("Completed KNN\n")

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by="Accuracy", ascending=False)

    print("\n" + "=" * 60)
    print("         MODEL COMPARISON RESULTS")
    print("=" * 60 + "\n")
    print(results_df.to_string(index=False))

    labels = sorted(df[ASSIGNMENT_GROUP_COL].unique())

    print("\nGenerating Confusion Matrices...\n")

    for name in models:
        print(f"Generating confusion matrix for {name}...")
        cm = confusion_matrix(y_test, model_predictions[name], labels=range(len(labels)))

        display_labels = labels[:20] if len(labels) > 20 else labels
        plot_confusion_matrix(
            cm[: len(display_labels), : len(display_labels)],
            display_labels,
            f"Confusion Matrix - {name}",
        )

    print("Generating confusion matrix for KNN...")
    knn_cm = confusion_matrix(y_test_reduced, knn_predictions, labels=range(len(labels)))

    display_labels = labels[:20] if len(labels) > 20 else labels
    plot_confusion_matrix(
        knn_cm[: len(display_labels), : len(display_labels)],
        display_labels,
        "Confusion Matrix - KNN",
    )

    best_row = results_df.iloc[0]
    best_model_name = best_row["Model"]

    print("\n" + "=" * 60)
    print("           BEST MODEL")
    print("=" * 60 + "\n")
    print(f"Best Model : {best_model_name}")
    print(f"Accuracy   : {best_row['Accuracy']:.4f} ({best_row['Accuracy']:.2%})")
    print(f"Precision  : {best_row['Precision']:.4f} ({best_row['Precision']:.2%})")
    print(f"Recall     : {best_row['Recall']:.4f} ({best_row['Recall']:.2%})")
    print(f"F1 Score   : {best_row['F1 Score']:.4f} ({best_row['F1 Score']:.2%})")

    if 0.85 <= best_row["Accuracy"] <= 0.87:
        print("\nTarget accuracy range (85-87%) achieved!")
    elif best_row["Accuracy"] < 0.85:
        print(f"\nAccuracy ({best_row['Accuracy']:.2%}) below target range (85-87%).")
        print("   Consider:")
        print("   - Adding more training data")
        print("   - Adjusting feature engineering")
        print("   - Tuning model hyperparameters")
    else:
        print(f"\nAccuracy ({best_row['Accuracy']:.2%}) above target range (85-87%).")
        print("   Consider checking for overfitting or simplifying the model.")

    print("\nTraining final model on full dataset...")
    final_model = LogisticRegression(max_iter=2500, C=0.3)

    final_model.fit(X, y)

    joblib.dump(final_model, MODEL_FILE)
    joblib.dump(vectorizer, VECTORIZER_FILE)
    if use_lemmatizer and vectorizer_lem is not None:
        joblib.dump(vectorizer_lem, VECTORIZER_LEM_FILE)
    joblib.dump(label_encoder, LABEL_ENCODER_FILE)
    joblib.dump(svd, SVD_FILE)

    train_optional_text_classifier(
        df,
        X,
        CATEGORY_COL,
        CATEGORY_MODEL_FILE,
        CATEGORY_LABEL_ENCODER_FILE,
    )
    train_optional_text_classifier(
        df,
        X,
        SUBCATEGORY_COL,
        SUBCATEGORY_MODEL_FILE,
        SUBCATEGORY_LABEL_ENCODER_FILE,
    )

    print(f"\nSaved {MODEL_FILE}")
    print(f"Saved {VECTORIZER_FILE}")
    if use_lemmatizer and vectorizer_lem is not None:
        print(f"Saved {VECTORIZER_LEM_FILE}")
    print(f"Saved {LABEL_ENCODER_FILE}")
    print(f"Saved {SVD_FILE}")
    print("\nModel saved successfully!")

    print("\n" + "=" * 60)
    print("     FEATURE IMPORTANCE ANALYSIS")
    print("=" * 60 + "\n")

    if "Random Forest" in trained_models:
        rf_model = trained_models["Random Forest"]
        if hasattr(rf_model, "feature_importances_"):
            importances = rf_model.feature_importances_
            indices = importances.argsort()[-20:][::-1]

            print("Top 20 most important features (Random Forest):")
            feature_names = vectorizer.get_feature_names_out()
            for i, idx in enumerate(indices, 1):
                if idx < len(feature_names):
                    print(f"  {i}. {feature_names[idx]}: {importances[idx]:.4f}")
                else:
                    print(f"  {i}. Feature_{idx}: {importances[idx]:.4f}")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the ServiceNow assignment group model.")
    parser.add_argument(
        "--source",
        choices=["file", "servicenow"],
        default=os.getenv("TRAINING_SOURCE", "servicenow"),
        help="Use a local spreadsheet/CSV or fetch incidents from ServiceNow.",
    )
    parser.add_argument(
        "--file",
        default=os.getenv("INCIDENTS_FILE", "incidents.xlsx"),
        help="Path to incidents.xlsx or incidents.csv when --source file is used.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=int(os.getenv("SERVICENOW_LIMIT", "1000")),
        help="Maximum ServiceNow incidents to fetch.",
    )
    parser.add_argument(
        "--query",
        default=os.getenv(
            "SERVICENOW_QUERY",
            "assignment_groupISNOTEMPTY^categoryISNOTEMPTY^subcategoryISNOTEMPTY",
        ),
        help="ServiceNow encoded query, for example assignment_groupISNOTEMPTY^active=false.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    try:
        train(parse_args())
    except Exception as exc:
        print(f"Training failed: {exc}", file=sys.stderr)
        sys.exit(1)
