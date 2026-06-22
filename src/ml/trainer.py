# src/ml/trainer.py
# ============================================================
# Train & compare: Random Forest, XGBoost, LightGBM, LSTM
# Auto-selects best model by F1 Score
# ============================================================

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from loguru import logger
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report
)
import xgboost as xgb
import lightgbm as lgb

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config.settings import RANDOM_STATE, N_JOBS, MODELS_DIR, TRAIN_TEST_SPLIT
from src.data.database import SessionLocal, ModelMetric


class ModelTrainer:
    """Trains, evaluates, and persists all ML models."""

    MODELS = {
        "random_forest": RandomForestClassifier(
            n_estimators=200, max_depth=12, min_samples_split=10,
            random_state=RANDOM_STATE, n_jobs=N_JOBS, class_weight="balanced"
        ),
        "xgboost": xgb.XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=RANDOM_STATE, eval_metric="logloss",
            use_label_encoder=False, n_jobs=N_JOBS
        ),
        "lightgbm": lgb.LGBMClassifier(
            n_estimators=200, max_depth=8, learning_rate=0.05,
            num_leaves=63, subsample=0.8, colsample_bytree=0.8,
            random_state=RANDOM_STATE, n_jobs=N_JOBS, verbose=-1
        ),
    }

    def __init__(self, feature_cols: list):
        self.feature_cols = feature_cols
        self.scaler = StandardScaler()
        self.best_model_name = None
        self.best_model = None
        self.metrics = {}

    # ----------------------------------------------------------
    def prepare_data(self, df: pd.DataFrame):
        X = df[self.feature_cols].values
        y = df["target"].values
        split_idx = int(len(X) * TRAIN_TEST_SPLIT)
        return (
            X[:split_idx], X[split_idx:],
            y[:split_idx], y[split_idx:]
        )

    # ----------------------------------------------------------
    def train_all(self, df: pd.DataFrame) -> dict:
        """Train all classical models and return metrics dict."""
        X_train, X_test, y_train, y_test = self.prepare_data(df)

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        results = {}
        for name, model in self.MODELS.items():
            logger.info(f"Training {name}...")
            try:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                y_prob = model.predict_proba(X_test_scaled)[:, 1]

                metrics = {
                    "accuracy": round(accuracy_score(y_test, y_pred), 4),
                    "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
                    "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
                    "f1": round(f1_score(y_test, y_pred, zero_division=0), 4),
                    "roc_auc": round(roc_auc_score(y_test, y_prob), 4),
                }
                results[name] = {"model": model, "metrics": metrics}
                logger.info(f"{name} — F1: {metrics['f1']:.4f}, AUC: {metrics['roc_auc']:.4f}")
            except Exception as e:
                logger.error(f"{name} training failed: {e}")

        self.metrics = results
        self._select_best()
        return results

    # ----------------------------------------------------------
    def _select_best(self):
        """Select best model by F1 score."""
        best_f1 = -1
        for name, result in self.metrics.items():
            if result["metrics"]["f1"] > best_f1:
                best_f1 = result["metrics"]["f1"]
                self.best_model_name = name
                self.best_model = result["model"]
        logger.info(f"Best model: {self.best_model_name} (F1={best_f1:.4f})")

    # ----------------------------------------------------------
    def save_models(self, ticker: str = "global"):
        """Persist all models and scaler to disk."""
        safe = ticker.replace(".", "_")
        for name, result in self.metrics.items():
            path = MODELS_DIR / f"{safe}_{name}.pkl"
            joblib.dump(result["model"], path)
        scaler_path = MODELS_DIR / f"{safe}_scaler.pkl"
        joblib.dump(self.scaler, scaler_path)
        # Save best model name
        with open(MODELS_DIR / f"{safe}_best_model.txt", "w") as f:
            f.write(self.best_model_name or "xgboost")
        logger.info(f"Models saved for {ticker}")

    # ----------------------------------------------------------
    def load_models(self, ticker: str = "global"):
        """Load saved models from disk."""
        safe = ticker.replace(".", "_")
        scaler_path = MODELS_DIR / f"{safe}_scaler.pkl"
        if scaler_path.exists():
            self.scaler = joblib.load(scaler_path)

        best_path = MODELS_DIR / f"{safe}_best_model.txt"
        if best_path.exists():
            with open(best_path) as f:
                self.best_model_name = f.read().strip()
            model_path = MODELS_DIR / f"{safe}_{self.best_model_name}.pkl"
            if model_path.exists():
                self.best_model = joblib.load(model_path)
                logger.info(f"Loaded {self.best_model_name} for {ticker}")
                return True
        return False

    # ----------------------------------------------------------
    def save_metrics_to_db(self, ticker: str = "global"):
        """Store evaluation metrics in database."""
        db = SessionLocal()
        try:
            for name, result in self.metrics.items():
                m = result["metrics"]
                record = ModelMetric(
                    model_name=name, ticker=ticker,
                    accuracy=m["accuracy"], precision=m["precision"],
                    recall=m["recall"], f1_score=m["f1"], roc_auc=m["roc_auc"],
                    is_best=(name == self.best_model_name)
                )
                db.add(record)
            db.commit()
        finally:
            db.close()

    # ----------------------------------------------------------
    def train_lstm(self, df: pd.DataFrame, ticker: str = "global"):
        """Train LSTM model using PyTorch."""
        try:
            import torch
            import torch.nn as nn
            from torch.utils.data import DataLoader, TensorDataset

            X_arr = df[self.feature_cols].values
            y_arr = df["target"].values
            X_scaled = self.scaler.transform(X_arr)

            seq_len = 20
            X_seq, y_seq = [], []
            for i in range(seq_len, len(X_scaled)):
                X_seq.append(X_scaled[i - seq_len:i])
                y_seq.append(y_arr[i])
            X_seq = np.array(X_seq)
            y_seq = np.array(y_seq)

            split = int(len(X_seq) * TRAIN_TEST_SPLIT)
            X_train = torch.FloatTensor(X_seq[:split])
            y_train = torch.FloatTensor(y_seq[:split])
            X_test = torch.FloatTensor(X_seq[split:])
            y_test_np = y_seq[split:]

            class LSTMModel(nn.Module):
                def __init__(self, input_size, hidden=64, layers=2):
                    super().__init__()
                    self.lstm = nn.LSTM(input_size, hidden, layers,
                                        batch_first=True, dropout=0.2)
                    self.fc = nn.Linear(hidden, 1)
                    self.sigmoid = nn.Sigmoid()

                def forward(self, x):
                    out, _ = self.lstm(x)
                    return self.sigmoid(self.fc(out[:, -1, :])).squeeze()

            model = LSTMModel(len(self.feature_cols))
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            criterion = nn.BCELoss()
            dataset = TensorDataset(X_train, y_train)
            loader = DataLoader(dataset, batch_size=32, shuffle=False)

            model.train()
            for epoch in range(30):
                for xb, yb in loader:
                    optimizer.zero_grad()
                    out = model(xb)
                    loss = criterion(out, yb)
                    loss.backward()
                    optimizer.step()

            model.eval()
            with torch.no_grad():
                probs = model(X_test).numpy()
                preds = (probs > 0.5).astype(int)

            metrics = {
                "accuracy": round(accuracy_score(y_test_np, preds), 4),
                "precision": round(precision_score(y_test_np, preds, zero_division=0), 4),
                "recall": round(recall_score(y_test_np, preds, zero_division=0), 4),
                "f1": round(f1_score(y_test_np, preds, zero_division=0), 4),
                "roc_auc": round(roc_auc_score(y_test_np, probs), 4),
            }
            self.metrics["lstm"] = {"model": model, "metrics": metrics}
            torch.save(model.state_dict(),
                       MODELS_DIR / f"{ticker.replace('.', '_')}_lstm.pt")
            logger.info(f"LSTM — F1: {metrics['f1']:.4f}, AUC: {metrics['roc_auc']:.4f}")
            self._select_best()
            return metrics
        except Exception as e:
            logger.error(f"LSTM training failed: {e}")
            return {}
