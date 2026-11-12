from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
import joblib

def train_model(df, save_path):
    print("กำลังสร้างและเทรนโมเดล AI (Random Forest + SMOTE)...")
    
    X = df.drop('Target_NPF', axis=1)
    y = df['Target_NPF']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns
    cat_cols = X.select_dtypes(include=['object']).columns
    
    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), cat_cols)
    ])
    
    # ข้ามขั้นตอน GridSearchCV ไปก่อนเพื่อความรวดเร็วในการรัน ไปใช้ Best Params ที่หาได้จาก Notebook เลย
    pipeline = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('classifier', RandomForestClassifier(max_depth=5, min_samples_split=5, n_estimators=200, random_state=42))
    ])
    
    pipeline.fit(X_train, y_train)
    
    # บันทึกเป็นไฟล์ .pkl ชั่วคราว (จะถูกอัปเดตใหม่ในขั้นตอนถัดไป)
    joblib.dump({'pipeline': pipeline, 'optimal_threshold': 0.5}, save_path)
    
    return pipeline, X_test, y_test