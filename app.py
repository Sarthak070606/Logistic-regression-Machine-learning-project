from flask import Flask, render_template_string, Response
import io
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

app = Flask(__name__)

def run_pipeline(random_state: int = 42):
    # Load Seaborn Titanic dataset
    df = sns.load_dataset("titanic")

    # Drop low-signal / duplicate info columns
    cols_to_drop = ["deck", "embark_town", "alive", "who", "adult_male", "class"]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Features/target
    y = df["survived"].astype(int)
    X = df.drop(columns=["survived"])  # keep: pclass, sex, age, sibsp, parch, fare, embarked, alone

    # Identify column types
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    # Treat bools + object/category as categoricals for one-hot
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    # Preprocess: impute + scale/encode
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]),
                numeric_cols,
            ),
            (
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]),
                categorical_cols,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("clf", LogisticRegression(max_iter=1000, n_jobs=None)),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=random_state
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    accuracy = metrics.accuracy_score(y_test, y_pred)
    precision = metrics.precision_score(y_test, y_pred, zero_division=0)
    recall = metrics.recall_score(y_test, y_pred, zero_division=0)
    f1 = metrics.f1_score(y_test, y_pred, zero_division=0)
    roc_auc = metrics.roc_auc_score(y_test, y_proba)
    cm = metrics.confusion_matrix(y_test, y_pred)

    report_dict = metrics.classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report_dict).T.reset_index().rename(columns={"index": "class"})

    comparison = pd.DataFrame({
        "Actual": y_test.reset_index(drop=True).astype(int),
        "Predicted": pd.Series(y_pred).astype(int),
        "Probability": pd.Series(y_proba)
    })
    head_preview = comparison.head(20).round(4)

    return {
        "metrics": {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_auc": roc_auc,
        },
        "cm": cm,
        "report_df": report_df,
        "preview": head_preview,
        "full": comparison,
    }


PAGE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Titanic – Logistic Regression (Seaborn)</title>
  <style>
    :root { --bg:#0b1220; --panel:#111a2b; --text:#e6eefc; --muted:#98a2b3; --accent:#4f8cff; --ok:#22c55e; --warn:#f59e0b; --danger:#ef4444; }
    * { box-sizing: border-box; }
    body { margin:0; font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, 'Helvetica Neue', Arial, 'Noto Sans', 'Apple Color Emoji', 'Segoe UI Emoji'; background: radial-gradient(1200px 800px at 80% -10%, #1a2a4a 0%, var(--bg) 60%); color: var(--text); }
    .container { max-width: 1100px; margin: 40px auto; padding: 0 16px; }
    header { display:flex; align-items:center; justify-content:space-between; gap:16px; margin-bottom:24px; }
    h1 { font-size: 28px; margin:0; letter-spacing:0.2px; }
    .btn { background: linear-gradient(135deg, var(--accent), #7aa7ff); color:white; border:none; padding:10px 16px; border-radius:16px; font-weight:600; cursor:pointer; box-shadow: 0 10px 24px rgba(79,140,255,.25); transition: transform .05s ease; }
    .btn:active { transform: translateY(1px); }
    .grid { display:grid; grid-template-columns: repeat(12, 1fr); gap:16px; }
    .card { grid-column: span 12; background: linear-gradient(180deg, rgba(255,255,255,.04), rgba(255,255,255,.02)); border:1px solid rgba(255,255,255,.06); border-radius:20px; padding:18px; box-shadow: 0 1px 0 rgba(255,255,255,.06) inset; }
    @media(min-width: 860px) { .span-4{grid-column: span 4;} .span-8{grid-column: span 8;} }
    .kpi { display:flex; align-items:center; justify-content:space-between; padding:12px 14px; border-radius:14px; background: rgba(255,255,255,.03); border:1px solid rgba(255,255,255,.05); margin-bottom:10px; }
    .kpi .label { color: var(--muted); font-size: 13px; }
    .kpi .value { font-weight: 700; font-size: 22px; }
    table { width:100%; border-collapse: collapse; font-size: 14px; }
    th, td { text-align:left; padding:10px 12px; border-bottom:1px solid rgba(255,255,255,.06); }
    th { color: var(--muted); font-weight:600; }
    .pill { display:inline-block; padding:4px 8px; border-radius:999px; font-size:12px; font-weight:600; }
    .pill.ok { background: rgba(34,197,94,.15); color:#86efac; }
    .pill.warn { background: rgba(245,158,11,.15); color:#fcd34d; }
    .pill.danger { background: rgba(239,68,68,.15); color:#fecaca; }
    footer { margin-top: 26px; color: var(--muted); font-size: 12px; text-align:center; }
    .note { color: var(--muted); font-size: 13px; margin: 6px 0 0; }
    .cm { display:grid; grid-template-columns: repeat(2, 1fr); gap:8px; margin-top:10px; }
    .cm div { padding:16px; text-align:center; border-radius:12px; background: rgba(255,255,255,.03); border:1px solid rgba(255,255,255,.05); font-weight:700; }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <h1> Titanic – Logistic Regression (Seaborn)</h1>
      <form method="get" action="/">
        <button class="btn" type="submit">Re-run model</button>
      </form>
    </header>

    <div class="grid">
      <section class="card span-4">
        <h3 style="margin-top:0">Metrics</h3>
        <div class="kpi"><span class="label">Accuracy</span><span class="value">{{ (metrics.accuracy*100)|round(2) }}%</span></div>
        <div class="kpi"><span class="label">Precision</span><span class="value">{{ (metrics.precision*100)|round(2) }}%</span></div>
        <div class="kpi"><span class="label">Recall</span><span class="value">{{ (metrics.recall*100)|round(2) }}%</span></div>
        <div class="kpi"><span class="label">F1-score</span><span class="value">{{ (metrics.f1*100)|round(2) }}%</span></div>
        <div class="kpi"><span class="label">ROC-AUC</span><span class="value">{{ (metrics.roc_auc*100)|round(2) }}%</span></div>
        <p class="note">Logistic Regression with one-hot encoded categoricals and scaled numerics.</p>
      </section>

      <section class="card span-8">
        <h3 style="margin-top:0">Confusion Matrix</h3>
        <div class="cm">
          <div>TN<br>{{ cm[0][0] }}</div>
          <div>FP<br>{{ cm[0][1] }}</div>
          <div>FN<br>{{ cm[1][0] }}</div>
          <div>TP<br>{{ cm[1][1] }}</div>
        </div>
      </section>

      <section class="card span-12">
        <h3 style="margin-top:0">Classification Report</h3>
        <table>
          <thead>
            <tr>
              {% for col in report_cols %}
              <th>{{ col }}</th>
              {% endfor %}
            </tr>
          </thead>
          <tbody>
            {% for row in report_rows %}
            <tr>
              {% for col in report_cols %}
              <td>
                {% if col in ["precision","recall","f1-score","support"] and row[col] is not none %}
                  {% if col == "support" %}{{ row[col]|round(0)|int }}{% else %}{{ (row[col]*100)|round(2) }}%{% endif %}
                {% else %}
                  {{ row[col] }}
                {% endif %}
              </td>
              {% endfor %}
            </tr>
            {% endfor %}
          </tbody>
        </table>
      </section>

      <section class="card span-12">
        <h3 style="margin-top:0">Preview: Predictions vs Actuals (first 20)</h3>
        <table>
          <thead>
            <tr>
              <th>#</th>
              <th>Actual</th>
              <th>Predicted</th>
              <th>Probability</th>
              <th>Status</th>
            </tr>
          </thead>
          <tbody>
            {% for i,row in preview.iterrows() %}
            <tr>
              <td>{{ i }}</td>
              <td>{{ row["Actual"] }}</td>
              <td>{{ row["Predicted"] }}</td>
              <td>{{ ('%.4f' % row["Probability"]) }}</td>
              <td>
                {% if row['Actual'] == row['Predicted'] %}
                  <span class="pill ok">correct</span>
                {% else %}
                  <span class="pill danger">incorrect</span>
                {% endif %}
              </td>
            </tr>
            {% endfor %}
          </tbody>
        </table>
        <p class="note">Need the whole table? <a style="color:#cbd5e1" href="/download" download>Download CSV</a></p>
      </section>
    </div>

    <footer>
      Built with Flask · Scikit-learn · Seaborn Titanic. Refresh to re-run with the same split.
    </footer>
  </div>
</body>
</html>
"""


@app.route("/")
def index():
    out = run_pipeline()

    # Prepare report table rows/cols
    report_df = out["report_df"].copy()
    report_df["precision"] = report_df["precision"].astype(float)
    report_df["recall"] = report_df["recall"].astype(float)
    report_df["f1-score"] = report_df["f1-score"].astype(float)
    report_df["support"] = report_df["support"].astype(float)

    report_cols = ["class", "precision", "recall", "f1-score", "support"]
    report_rows = report_df[report_cols].to_dict(orient="records")

    return render_template_string(
        PAGE,
        metrics=out["metrics"],
        cm=out["cm"],
        report_cols=report_cols,
        report_rows=report_rows,
        preview=out["preview"],
    )


@app.route("/download")
def download_csv():
    out = run_pipeline()
    csv_buf = io.StringIO()
    out["full"].to_csv(csv_buf, index=False)
    csv = csv_buf.getvalue()
    return Response(
        csv,
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=predictions_vs_actuals.csv"},
    )


if __name__ == "__main__":
    # For local dev; on Render/Heroku use gunicorn: `gunicorn app:app`
    app.run(host="0.0.0.0", port=5000, debug=True)
