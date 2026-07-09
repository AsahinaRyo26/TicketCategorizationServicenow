import os
import re
from typing import Dict, List

import joblib
import nltk
import pandas as pd
import requests
from flask import Flask, jsonify, request
from requests.auth import HTTPBasicAuth
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from nltk.stem import PorterStemmer, WordNetLemmatizer

NLTK_SETUP_ERROR = None
try:
    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        nltk.download("wordnet", quiet=True)
        nltk.download("omw-1.4", quiet=True)
except Exception as exc:
    NLTK_SETUP_ERROR = str(exc)


MODEL_FILE = "model.pkl"
VECTORIZER_FILE = "vectorizer.pkl"
LABEL_ENCODER_FILE = "label_encoder.pkl"

SHORT_DESCRIPTION_COL = "Short description"
DESCRIPTION_COL = "Description"


SERVICENOW_FIELD_MAP: Dict[str, str] = {
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


app = Flask(__name__)

model = None
vectorizer = None
label_encoder = None
vectorizer_lem = None
ARTIFACT_LOAD_ERROR = None

try:
    model = joblib.load(MODEL_FILE)
    vectorizer = joblib.load(VECTORIZER_FILE)
    label_encoder = joblib.load(LABEL_ENCODER_FILE)
    try:
        vectorizer_lem = joblib.load("vectorizer_lem.pkl")
    except FileNotFoundError:
        vectorizer_lem = None
except Exception as exc:
    ARTIFACT_LOAD_ERROR = str(exc)

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()


def clean_text(text: object) -> str:
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s\.\-_]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def remove_stopwords(text: str) -> str:
    words = [word for word in text.split() if word not in CUSTOM_STOPWORDS and len(word) > 1]
    return " ".join(words)


def stem_text(text: str) -> str:
    return " ".join(stemmer.stem(word) for word in text.split())


def lemmatize_text(text: str) -> str:
    if NLTK_SETUP_ERROR:
        return text
    try:
        return " ".join(lemmatizer.lemmatize(word) for word in text.split())
    except LookupError:
        return text


def preprocess_text(text: object, use_lemmatization: bool = False) -> str:
    text = clean_text(text)
    text = remove_stopwords(text)
    if use_lemmatization:
        return lemmatize_text(text)
    return stem_text(text)


def get_service_now_credentials() -> tuple[str, str, str]:
    instance_url = os.getenv("SERVICENOW_INSTANCE_URL", "").rstrip("/")
    username = os.getenv("SERVICENOW_USERNAME", "")
    password = os.getenv("SERVICENOW_PASSWORD")

    if not instance_url:
        raise RuntimeError("SERVICENOW_INSTANCE_URL is not configured on the server.")
    if not username:
        raise RuntimeError("SERVICENOW_USERNAME is not configured on the server.")
    if not password:
        raise RuntimeError("SERVICENOW_PASSWORD is not configured on the server.")

    return instance_url, username, password


def fetch_servicenow_incidents(limit: int = 20, query: str = "") -> List[dict]:
    instance_url, username, password = get_service_now_credentials()
    api_fields = sorted(set(SERVICENOW_FIELD_MAP.values()))

    params = {
        "sysparm_display_value": "true",
        "sysparm_exclude_reference_link": "true",
        "sysparm_fields": ",".join(api_fields),
        "sysparm_limit": max(1, min(limit, 100)),
    }
    if query:
        params["sysparm_query"] = query

    response = requests.get(
        f"{instance_url}/api/now/table/incident",
        params=params,
        auth=HTTPBasicAuth(username, password),
        headers={"Accept": "application/json"},
        timeout=30,
    )
    response.raise_for_status()

    rows = response.json().get("result", [])
    return [
        {excel_col: row.get(api_col, "") for excel_col, api_col in SERVICENOW_FIELD_MAP.items()}
        for row in rows
    ]


def predict_from_text(short_description: str = "", description: str = "") -> dict:
    if model is None or vectorizer is None or label_encoder is None:
        raise RuntimeError(
            "Model files are not loaded. Run train_model.py first and keep model.pkl, "
            "vectorizer.pkl, and label_encoder.pkl in the same folder as app.py."
        )

    combined_text = f"{short_description} {description}".strip()
    if not combined_text:
        raise ValueError("Provide short_description, description, or text.")

    processed_text = preprocess_text(combined_text, use_lemmatization=False)
    text_vector = vectorizer.transform([processed_text])

    if vectorizer_lem is not None:
        from scipy.sparse import hstack

        processed_text_lem = preprocess_text(combined_text, use_lemmatization=True)
        text_vector_lem = vectorizer_lem.transform([processed_text_lem])
        text_vector = hstack([text_vector, text_vector_lem])
    encoded_prediction = model.predict(text_vector)[0]
    prediction = label_encoder.inverse_transform([encoded_prediction])[0]

    response = {
        "predicted_assignment_group": prediction,
        "confidence": None,
        "top_3_predictions": [],
    }

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(text_vector)[0]
        top_indices = probabilities.argsort()[-3:][::-1]
        response["confidence"] = round(float(probabilities[top_indices[0]]), 4)
        response["top_3_predictions"] = [
            {
                "assignment_group": label_encoder.inverse_transform([idx])[0],
                "confidence": round(float(probabilities[idx]), 4),
            }
            for idx in top_indices
        ]

    return response


@app.route("/")
def home():
    return jsonify(
        {
            "service": "ServiceNow Assignment Group Prediction API",
            "endpoints": [
                "GET /health",
                "POST /predict",
                "POST /predict_text",
                "POST /predict_batch",
                "GET /servicenow/incidents",
                "POST /servicenow/predict",
            ],
        }
    )


@app.route("/health", methods=["GET"])
def health():
    return jsonify(
        {
            "status": "healthy",
            "model_loaded": model is not None,
            "vectorizer_loaded": vectorizer is not None,
            "label_encoder_loaded": label_encoder is not None,
            "artifact_load_error": ARTIFACT_LOAD_ERROR,
            "nltk_setup_error": NLTK_SETUP_ERROR,
            "servicenow_instance_configured": bool(os.getenv("SERVICENOW_INSTANCE_URL")),
            "servicenow_username_configured": bool(os.getenv("SERVICENOW_USERNAME")),
            "servicenow_password_configured": bool(os.getenv("SERVICENOW_PASSWORD")),
        }
    )


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(silent=True) or {}
        result = predict_from_text(
            short_description=data.get("short_description", ""),
            description=data.get("description", ""),
        )
        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/predict_text", methods=["POST"])
def predict_text():
    try:
        data = request.get_json(silent=True) or {}
        result = predict_from_text(description=data.get("text", ""))
        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/predict_batch", methods=["POST"])
def predict_batch():
    try:
        data = request.get_json(silent=True) or {}
        tickets = data.get("tickets", [])
        if not isinstance(tickets, list):
            return jsonify({"error": "tickets must be a list"}), 400

        results = []
        for ticket in tickets:
            short_description = ticket.get("short_description", "")
            description = ticket.get("description", "")
            prediction = predict_from_text(short_description, description)
            results.append(
                {
                    "short_description": short_description,
                    "description": description,
                    **prediction,
                }
            )

        return jsonify({"total": len(results), "results": results})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/servicenow/incidents", methods=["GET"])
def servicenow_incidents():
    try:
        limit = int(request.args.get("limit", 20))
        query = request.args.get("query", "")
        rows = fetch_servicenow_incidents(limit=limit, query=query)
        return jsonify({"total": len(rows), "incidents": rows})
    except requests.HTTPError as exc:
        return jsonify({"error": "ServiceNow request failed", "details": str(exc)}), 502
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/servicenow/predict", methods=["POST"])
def servicenow_predict():
    try:
        data = request.get_json(silent=True) or {}
        limit = int(data.get("limit", 20))
        query = data.get("query", "assignment_groupISEMPTY")
        rows = fetch_servicenow_incidents(limit=limit, query=query)

        results = []
        for row in rows:
            prediction = predict_from_text(
                short_description=row.get(SHORT_DESCRIPTION_COL, ""),
                description=row.get(DESCRIPTION_COL, ""),
            )
            results.append(
                {
                    "number": row.get("Number", ""),
                    "short_description": row.get(SHORT_DESCRIPTION_COL, ""),
                    "description": row.get(DESCRIPTION_COL, ""),
                    **prediction,
                }
            )

        return jsonify({"total": len(results), "results": results})
    except requests.HTTPError as exc:
        return jsonify({"error": "ServiceNow request failed", "details": str(exc)}), 502
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=False)
