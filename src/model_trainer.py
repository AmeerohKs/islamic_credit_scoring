import os
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def train_model(df, save_path):
  print("กำลังสร้างและเทรนโมเดล AI (Random Forest + SMOTE)...")

  # ตัดคอลัมน์ผลลัพธ์จาก Pre-screening ออก เพื่อป้องกันข้อมูลปนเปื้อนเข้าโมเดล
  cols_to_drop = ["Is_Passed_Filter", "Reject_Reasons", "Flags", "Target_NPF"]
  X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
  y = df["Target_NPF"]

  X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=0.2, random_state=42, stratify=y
  )

  num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
  cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

  preprocessor = ColumnTransformer(
      transformers=[
          ("num", StandardScaler(), num_cols),
          ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
      ]
  )

  pipeline = ImbPipeline(
      steps=[
          ("preprocessor", preprocessor),
          ("smote", SMOTE(random_state=42)),
          (
              "classifier",
              RandomForestClassifier(
                  max_depth=10,
                  min_samples_split=5,
                  n_estimators=200,
                  random_state=42,
              ),
          ),
      ]
  )

  pipeline.fit(X_train, y_train)

  os.makedirs(os.path.dirname(save_path), exist_ok=True)
  joblib.dump({"pipeline": pipeline, "optimal_threshold": 0.5}, save_path)

  return pipeline, X_test, y_test