# 🩺 Diabetes Risk Predictor

A machine learning web app that predicts diabetes risk from clinical
indicators, trained on the **Pima Indians Diabetes Dataset**. 100% Python
(scikit-learn for the model, Streamlit for the web app).

## What's in this folder

| File | Purpose |
|---|---|
| `diabetes.csv` | Raw dataset (768 patients, 8 features + outcome) |
| `train_model.py` | Cleans data, trains/compares 4 models, saves the best one |
| `app.py` | The Streamlit web application |
| `diabetes_model.joblib` | Trained model (Gradient Boosting Classifier) |
| `scaler.joblib` | Feature scaler used at inference time |
| `metrics.json` | Test-set performance metrics |
| `feature_stats.json` | Dataset statistics (shown in-app) |
| `requirements.txt` | Python dependencies |

## Model performance (held-out test set)

- **Accuracy:** 88.3%
- **Precision:** 82.1%
- **Recall:** 85.2%
- **ROC-AUC:** 0.955

Four models were compared with 5-fold cross-validation (Logistic Regression,
Random Forest, Gradient Boosting, SVM) — Gradient Boosting won and was used
for the final model.

## Run it locally (2 minutes)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Optional) Retrain the model from scratch
python train_model.py

# 3. Launch the web app
streamlit run app.py
```

Your browser will open automatically at `http://localhost:8501`.

## Deploy it live for free (Streamlit Community Cloud) — ~5 minutes

This is the fastest way to get a real public URL, no server management needed.

1. **Create a GitHub repo** and push this entire folder to it:
   ```bash
   git init
   git add .
   git commit -m "Diabetes risk predictor"
   git branch -M main
   git remote add origin https://github.com/<your-username>/<repo-name>.git
   git push -u origin main
   ```
2. Go to **[share.streamlit.io](https://share.streamlit.io)** and sign in with GitHub.
3. Click **"New app"**, select your repo, branch `main`, and set the main file
   path to `app.py`.
4. Click **Deploy**. Streamlit Cloud installs `requirements.txt` automatically
   and gives you a live URL like:
   `https://<your-app-name>.streamlit.app`
5. That's it — it's live, free, and auto-redeploys whenever you push new commits.

### Other free/low-cost hosting options
- **Hugging Face Spaces** (choose the Streamlit SDK) — also free, similar setup.
- **Render.com** — free web service tier, use `streamlit run app.py --server.port $PORT --server.address 0.0.0.0` as the start command.
- **Railway.app** — similar to Render, generous free tier.

## Important notes on the data cleaning

The raw dataset encodes missing values as `0` for `Glucose`, `BloodPressure`,
`SkinThickness`, `Insulin`, and `BMI` — which are not physiologically valid
zeros. `train_model.py` treats these as missing and imputes them using the
median (grouped by diabetes outcome for a better estimate). This is a
standard, well-documented preprocessing step for this dataset and
meaningfully improves model quality over leaving raw zeros in place.

## Disclaimer

This project is for **educational and demonstration purposes only**. It is
not a certified medical device and must never be used as a substitute for
professional medical diagnosis or advice.
