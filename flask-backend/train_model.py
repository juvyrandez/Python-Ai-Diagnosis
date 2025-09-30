import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import joblib

# Standard numeric features used across datasets (Celsius and metric units)
STD_NUMERIC = [
    'age',
    'blood_pressure_systolic',
    'blood_pressure_diastolic',
    'temperature_c',
    'weight_kg',
    'heart_rate_bpm',
    'resp_rate_cpm',
]


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize possible input schemas to the standard schema expected by models.
    - Convert Fahrenheit to Celsius if needed.
    - Fill missing numeric columns with defaults.
    """
    df = df.copy()

    # Rename compatible columns into the standard names
    colmap = {
        'systolic_bp': 'blood_pressure_systolic',
        'diastolic_bp': 'blood_pressure_diastolic',
        'blood_pressure_systolic': 'blood_pressure_systolic',
        'blood_pressure_diastolic': 'blood_pressure_diastolic',
        'temperature': 'temperature_f',  # if Fahrenheit appears as 'temperature'
        'temperature_c': 'temperature_c',
        'weight': 'weight_kg',
        'weight_kg': 'weight_kg',
        'heart_rate': 'heart_rate_bpm',
        'heart_rate_bpm': 'heart_rate_bpm',
        'respiratory_rate': 'resp_rate_cpm',
        'resp_rate_cpm': 'resp_rate_cpm',
    }
    for src, dst in colmap.items():
        if src in df.columns and src != dst:
            df[dst] = df[src]

    # Fahrenheit to Celsius if applicable
    if 'temperature_f' in df.columns and 'temperature_c' not in df.columns:
        df['temperature_c'] = (df['temperature_f'] - 32.0) * (5.0 / 9.0)

    # Ensure all required numeric columns exist
    defaults = {
        'age': 30,
        'blood_pressure_systolic': 120,
        'blood_pressure_diastolic': 80,
        'temperature_c': 37.0,
        'weight_kg': 70.0,
        'heart_rate_bpm': 80.0,
        'resp_rate_cpm': 18.0,
    }
    for col, default in defaults.items():
        if col not in df.columns:
            df[col] = default

    # Keep only needed columns where present
    keep = ['complaint', 'diagnosis'] + STD_NUMERIC
    cols = [c for c in keep if c in df.columns]
    return df[cols]


def load_general_dataframe() -> pd.DataFrame:
    """Prefer expanded dataset if present; otherwise fall back to base complaints.csv."""
    if os.path.exists('complaints_expanded.csv'):
        base = pd.read_csv('complaints_expanded.csv')
    else:
        base = pd.read_csv('complaints.csv')
    return normalize_columns(base)


def train_general_diagnosis_model():
    print('Training general diagnosis model...')
    df = load_general_dataframe()

    if 'complaint' not in df.columns or 'diagnosis' not in df.columns:
        raise ValueError('Dataset must contain complaint and diagnosis columns')

    text_col = 'complaint'
    num_cols = STD_NUMERIC

    preprocessor = ColumnTransformer(
        transformers=[
            ('text', TfidfVectorizer(max_features=1200, stop_words='english'), text_col),
            ('num', StandardScaler(), num_cols),
        ]
    )

    model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=200, random_state=42))
    ])

    X = df[[text_col] + num_cols]
    y = df['diagnosis']

    model.fit(X, y)
    joblib.dump(model, 'diagnosis_model.pkl')

    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    print(f'General diagnosis model accuracy (train resubstitution): {acc:.2f}')
    return model


def train_animal_bite_category_model():
    print('Training animal bite CATEGORY model...')
    if not os.path.exists('animal_bites.csv'):
        print('animal_bites.csv not found, skipping animal bite model.')
        return None

    raw = pd.read_csv('animal_bites.csv')
    df = normalize_columns(raw)

    # Derive category label from either explicit category column or diagnosis text
    if 'category' in raw.columns:
        y = raw['category'].astype(str)
    elif 'diagnosis' in raw.columns:
        y = raw['diagnosis'].astype(str).str.extract(r'(Category\s*[123])', expand=False).fillna('Category 2')
    else:
        raise ValueError('animal_bites.csv must contain either category or diagnosis column')

    text_col = 'complaint'
    num_cols = STD_NUMERIC

    preprocessor = ColumnTransformer(
        transformers=[
            ('text', TfidfVectorizer(max_features=800, stop_words='english'), text_col),
            ('num', StandardScaler(), num_cols),
        ]
    )

    model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=250, random_state=42))
    ])

    X = df[[text_col] + num_cols]
    model.fit(X, y)
    joblib.dump(model, 'animal_bite_category_model.pkl')

    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    print(f'Animal bite category model accuracy (train resubstitution): {acc:.2f}')
    return model


if __name__ == '__main__':
    train_general_diagnosis_model()
    animal_model = train_animal_bite_category_model()
    print('✅ Training complete. Saved:')
    print('- diagnosis_model.pkl')
    if animal_model is not None:
        print('- animal_bite_category_model.pkl')

