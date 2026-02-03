from fastapi import FastAPI, UploadFile, File
import pandas as pd
import io

from preprocessing.preprocess import load_and_preprocess
from ml_engine.anomaly import detect_anomalies
from genai_engine.insight_generator import generate_insight

app = FastAPI(title="AI Data Analyst System")

# -------------------------------------------------
# Health Check
# -------------------------------------------------
@app.get("/")
def home():
    return {"message": "AI Data Analyst System Running"}

# -------------------------------------------------
# Analyze Default Dataset
# -------------------------------------------------
@app.get("/analyze")
def analyze():
    try:
        df = load_and_preprocess("data/sample_sales.csv")

        df = detect_anomalies(df)

        anomaly_count = int((df["anomaly"] == -1).sum())
        summary = f"Detected {anomaly_count} anomalies in recent sales data."

        try:
            insight = generate_insight(summary)
        except Exception as genai_error:
            insight = f"GenAI failed gracefully: {str(genai_error)}"

        # Prepare dashboard data
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        numeric_cols.remove("anomaly")

        dashboard_df = df[["date"] + numeric_cols + ["anomaly"]].head(1000)

        return {
            "anomalies": anomaly_count,
            "genai_insight": insight,
            "data": dashboard_df.to_dict(orient="records")
        }

    except Exception as e:
        return {"error": str(e)}

# -------------------------------------------------
# Analyze Uploaded CSV File
# -------------------------------------------------
@app.post("/analyze-file")
async def analyze_file(file: UploadFile = File(...)):
    try:
        contents = await file.read()

        # ---------- Safe CSV Reading ----------
        try:
            df = pd.read_csv(io.BytesIO(contents), encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_csv(io.BytesIO(contents), encoding="latin-1")

        # ---------- Auto-detect Date Column ----------
        date_col = None
        for col in df.columns:
            if "date" in col.lower() or "time" in col.lower():
                date_col = col
                break

        if date_col is None:
            return {
                "error": "CSV must contain a date-like column (date, order_date, timestamp)"
            }

        df["date"] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=["date"])

        # ---------- Feature Engineering ----------
        df["day"] = df["date"].dt.day
        df["month"] = df["date"].dt.month

        # ---------- Anomaly Detection ----------
        df = detect_anomalies(df)

        anomaly_count = int((df["anomaly"] == -1).sum())
        summary = f"Detected {anomaly_count} anomalies in uploaded data."

        try:
            insight = generate_insight(summary)
        except Exception as genai_error:
            insight = f"GenAI failed gracefully: {str(genai_error)}"

        # ---------- Prepare Dashboard Data ----------
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        numeric_cols.remove("anomaly")

        dashboard_df = df[["date"] + numeric_cols + ["anomaly"]].head(1000)

        return {
            "anomalies": anomaly_count,
            "genai_insight": insight,
            "data": dashboard_df.to_dict(orient="records")
        }

    except Exception as e:
        return {"error": str(e)}
