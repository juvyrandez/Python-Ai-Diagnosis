import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
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


def clean_and_deduplicate_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean data by removing duplicates and invalid entries.
    """
    print(f"Original dataset size: {len(df)} rows")
    
    # Remove exact duplicates
    df = df.drop_duplicates()
    print(f"After removing exact duplicates: {len(df)} rows")
    
    # Remove rows with missing complaint or diagnosis
    df = df.dropna(subset=['complaint', 'diagnosis'])
    print(f"After removing missing values: {len(df)} rows")
    
    # Remove rows with empty strings
    df = df[df['complaint'].str.strip() != '']
    df = df[df['diagnosis'].str.strip() != '']
    print(f"After removing empty strings: {len(df)} rows")
    
    # Remove duplicate complaint-diagnosis pairs (keep first occurrence)
    df = df.drop_duplicates(subset=['complaint', 'diagnosis'], keep='first')
    print(f"After removing duplicate complaint-diagnosis pairs: {len(df)} rows")
    
    return df


def load_general_dataframe() -> pd.DataFrame:
    """Prefer expanded dataset if present; otherwise fall back to base complaints.csv."""
    if os.path.exists('complaints_expanded.csv'):
        base = pd.read_csv('complaints_expanded.csv')
    else:
        base = pd.read_csv('complaints.csv')
    
    # Clean and deduplicate
    base = clean_and_deduplicate_data(base)
    
    return normalize_columns(base)


def train_general_diagnosis_model():
    print('\n' + '='*60)
    print('Training general diagnosis model...')
    print('='*60)
    df = load_general_dataframe()

    if 'complaint' not in df.columns or 'diagnosis' not in df.columns:
        raise ValueError('Dataset must contain complaint and diagnosis columns')

    text_col = 'complaint'
    num_cols = STD_NUMERIC

    X = df[[text_col] + num_cols]
    y = df['diagnosis']
    
    # Check class distribution
    print(f"\nClass distribution:")
    class_counts = y.value_counts()
    print(class_counts)
    print(f"\nTotal unique diagnoses: {y.nunique()}")
    
    # Check if we can use stratified splitting (need at least 2 samples per class)
    min_class_count = class_counts.min()
    use_stratify = min_class_count >= 2
    
    if not use_stratify:
        print(f"\nWarning: Some classes have only 1 sample. Using non-stratified split.")
    
    # Split data: 70% train, 15% validation, 15% test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y if use_stratify else None
    )
    
    # Check if temp set can be stratified
    temp_class_counts = y_temp.value_counts()
    use_stratify_temp = temp_class_counts.min() >= 2
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp if use_stratify_temp else None
    )
    
    print(f"\nData split:")
    print(f"  Training set: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
    print(f"  Validation set: {len(X_val)} samples ({len(X_val)/len(X)*100:.1f}%)")
    print(f"  Test set: {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")

    preprocessor = ColumnTransformer(
        transformers=[
            ('text', TfidfVectorizer(max_features=1500, stop_words='english', ngram_range=(1, 2)), text_col),
            ('num', StandardScaler(), num_cols),
        ]
    )

    model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            n_estimators=300,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ))
    ])

    # Train on training set
    print("\nTraining model...")
    model.fit(X_train, y_train)
    
    # Evaluate on training set
    y_train_pred = model.predict(X_train)
    train_acc = accuracy_score(y_train, y_train_pred)
    print(f"\nTraining accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")
    
    # Evaluate on validation set
    y_val_pred = model.predict(X_val)
    val_acc = accuracy_score(y_val, y_val_pred)
    print(f"Validation accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")
    
    # Evaluate on test set
    y_test_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test, y_test_pred)
    print(f"Test accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    
    # Show classification report for test set
    print("\nTest Set Classification Report:")
    print(classification_report(y_test, y_test_pred, zero_division=0))
    
    # Save the model
    joblib.dump(model, 'diagnosis_model.pkl')
    print("\n✅ Model saved as 'diagnosis_model.pkl'")
    
    return model


def train_animal_bite_category_model():
    print('\n' + '='*60)
    print('Training animal bite CATEGORY model...')
    print('='*60)
    
    # Try both file names
    if os.path.exists('animal_bites_cebuano.csv'):
        raw = pd.read_csv('animal_bites_cebuano.csv')
    elif os.path.exists('animal_bites.csv'):
        raw = pd.read_csv('animal_bites.csv')
    else:
        print('No animal bites dataset found, skipping animal bite model.')
        return None
    
    # Clean and deduplicate
    raw = clean_and_deduplicate_data(raw)
    df = normalize_columns(raw)

    # Derive category label from either explicit category column or diagnosis text
    if 'category' in raw.columns:
        y = raw['category'].astype(str)
    elif 'diagnosis' in raw.columns:
        y = raw['diagnosis'].astype(str).str.extract(r'(Category\s*[123])', expand=False).fillna('Category 2')
    else:
        raise ValueError('animal_bites dataset must contain either category or diagnosis column')

    text_col = 'complaint'
    num_cols = STD_NUMERIC

    X = df[[text_col] + num_cols]
    
    # Check class distribution
    print(f"\nClass distribution:")
    class_counts = y.value_counts()
    print(class_counts)
    print(f"\nTotal unique categories: {y.nunique()}")
    
    # Check if we can use stratified splitting (need at least 2 samples per class)
    min_class_count = class_counts.min()
    use_stratify = min_class_count >= 2
    
    if not use_stratify:
        print(f"\nWarning: Some classes have only 1 sample. Using non-stratified split.")
    
    # Split data: 70% train, 15% validation, 15% test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y if use_stratify else None
    )
    
    # Check if temp set can be stratified
    temp_class_counts = y_temp.value_counts()
    use_stratify_temp = temp_class_counts.min() >= 2
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp if use_stratify_temp else None
    )
    
    print(f"\nData split:")
    print(f"  Training set: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
    print(f"  Validation set: {len(X_val)} samples ({len(X_val)/len(X)*100:.1f}%)")
    print(f"  Test set: {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")

    preprocessor = ColumnTransformer(
        transformers=[
            ('text', TfidfVectorizer(max_features=1000, stop_words='english', ngram_range=(1, 2)), text_col),
            ('num', StandardScaler(), num_cols),
        ]
    )

    model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            n_estimators=300,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ))
    ])

    # Train on training set
    print("\nTraining model...")
    model.fit(X_train, y_train)
    
    # Evaluate on training set
    y_train_pred = model.predict(X_train)
    train_acc = accuracy_score(y_train, y_train_pred)
    print(f"\nTraining accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")
    
    # Evaluate on validation set
    y_val_pred = model.predict(X_val)
    val_acc = accuracy_score(y_val, y_val_pred)
    print(f"Validation accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")
    
    # Evaluate on test set
    y_test_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test, y_test_pred)
    print(f"Test accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    
    # Show classification report for test set
    print("\nTest Set Classification Report:")
    print(classification_report(y_test, y_test_pred, zero_division=0))
    
    # Save the model
    joblib.dump(model, 'animal_bite_category_model.pkl')
    print("\n✅ Model saved as 'animal_bite_category_model.pkl'")
    
    return model


if __name__ == '__main__':
    print("\n" + "="*60)
    print("AI DIAGNOSIS MODEL TRAINING")
    print("="*60)
    
    # Train general diagnosis model
    train_general_diagnosis_model()
    
    # Train animal bite category model
    animal_model = train_animal_bite_category_model()
    
    print("\n" + "="*60)
    print('✅ TRAINING COMPLETE')
    print("="*60)
    print('Saved models:')
    print('  - diagnosis_model.pkl')
    if animal_model is not None:
        print('  - animal_bite_category_model.pkl')
    print("="*60)

