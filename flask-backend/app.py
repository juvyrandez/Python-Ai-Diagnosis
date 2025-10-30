from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import joblib
import sys
import os
import csv
import re
from datetime import datetime, timezone
import pandas as pd
import numpy as np

app = Flask(__name__)
CORS(app)

# Compatibility shim: some models saved when ComplaintIndicators lived under __main__
# Expose the class there to satisfy unpickling of older artifacts
try:
    from train_model import ComplaintIndicators as _CI
    setattr(sys.modules['__main__'], 'ComplaintIndicators', _CI)
except Exception:
    pass

# Resolve paths relative to this file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load trained models (file-relative)
model_path = os.path.join(BASE_DIR, 'diagnosis_model.pkl')
model = joblib.load(model_path)
try:
    animal_model_path = os.path.join(BASE_DIR, 'animal_bite_category_model.pkl')
    animal_bite_category_model = joblib.load(animal_model_path)
except Exception:
    animal_bite_category_model = None

# Log file path (file-relative)
LOG_PATH = os.path.join(BASE_DIR, 'predictions_log.csv')

DEFAULTS = {
    'age': 40,
    'blood_pressure_systolic': 120,
    'blood_pressure_diastolic': 80,
    'temperature_c': 37.0,
    'weight_kg': 70,
    'heart_rate_bpm': 80,
    'resp_rate_cpm': 16,
}

VITALS_PATTERN = {
    'bp': re.compile(r"\b(?:bp|blood pressure)\s*(\d{2,3})\s*[\/\\-]\s*(\d{2,3})\b", re.I),
    'temp': re.compile(r"\b(?:temp|temperature)\s*(\d{2}(?:\.\d)?)\s*(?:c|celsius)?\b", re.I),
    'hr': re.compile(r"\b(?:hr|heart rate|pulse)\s*(\d{2,3})\s*(?:bpm)?\b", re.I),
    'rr': re.compile(r"\b(?:rr|resp(?:iratory)? rate)\s*(\d{1,2})\s*(?:cpm|rpm)?\b", re.I),
    'weight': re.compile(r"\b(?:weight|wt)\s*(\d{2,3})\s*(?:kg)?\b", re.I),
}

def extract_vitals_from_text(text: str):
    found = {}
    if not text:
        return found
    m = VITALS_PATTERN['bp'].search(text)
    if m:
        found['blood_pressure_systolic'] = int(m.group(1))
        found['blood_pressure_diastolic'] = int(m.group(2))
    m = VITALS_PATTERN['temp'].search(text)
    if m:
        found['temperature_c'] = float(m.group(1))
    m = VITALS_PATTERN['hr'].search(text)
    if m:
        found['heart_rate_bpm'] = int(m.group(1))
    m = VITALS_PATTERN['rr'].search(text)
    if m:
        found['resp_rate_cpm'] = int(m.group(1))
    m = VITALS_PATTERN['weight'].search(text)
    if m:
        found['weight_kg'] = int(m.group(1))
    return found
def apply_rule_based_adjustments(classes, proba, complaint, features):
    """Adjust probabilities with simple clinical priors to improve plausibility.
    Returns new list of probabilities aligned with classes.
    """
    text = (complaint or '').lower()
    age = float(features.get('age', 40))
    temp = float(features.get('temperature_c', 37.0))
    rr = float(features.get('resp_rate_cpm', 16))
    hr = float(features.get('heart_rate_bpm', 80))

    has_fever = bool(re.search(r"\bfever|febrile|hilanat\b", text)) or temp >= 38.0
    has_cough = bool(re.search(r"\bcough|coughing|ubo\b", text))
    has_sob = bool(re.search(r"\bshortness of breath|dyspnea|sob|difficulty breathing\b", text))
    has_chest_pain = bool(re.search(r"\bchest pain|pressure|tightness|sakit sa dughan\b", text))
    has_rash = bool(re.search(r"\brash|itch|hives|urticaria|skin\b", text))
    has_throat = bool(re.search(r"\bsore throat|pharyng|tonsillitis\b", text))
    has_abd = bool(re.search(r"\babdominal|stomach|epigastr|abd pain|tummy|sakit sa tiyan\b", text))
    has_urinary = bool(re.search(r"\bdysuria|urination|frequency|urgency|burning\b", text))
    has_dm_kw = bool(re.search(r"\b(diabetes|hypergly|polydipsia|polyuria|weight loss|excessive thirst)\b", text))
    has_headache = bool(re.search(r"\bheadache|head pain|sakit sa ulo\b", text))
    has_dizziness = bool(re.search(r"\bdizziness|dizzy|vertigo|pagkaluya\b", text))
    has_body_ache = bool(re.search(r"\bbody ache|muscle pain|myalgia|joint pain\b", text))
    has_runny_nose = bool(re.search(r"\brunny nose|nasal|congestion|colds\b", text))
    
    # Duration extraction (e.g., "3 days", "2 weeks")
    dur_days = None
    m_d = re.search(r"\b(\d{1,3})\s*day[s]?\b", text)
    m_w = re.search(r"\b(\d{1,2})\s*week[s]?\b", text)
    if m_w:
        try:
            dur_days = int(m_w.group(1)) * 7
        except Exception:
            dur_days = None
    if m_d and dur_days is None:
        try:
            dur_days = int(m_d.group(1))
        except Exception:
            dur_days = None
    # Rough symptom count via delimiters
    symptom_count = text.count(',') + text.count(' and ')
    
    # Count total words in complaint to assess specificity
    word_count = len(text.split())

    # Start with original probs
    weights = [float(p) for p in proba]

    def mult(label_substrs, factor):
        for i, cls in enumerate(classes):
            if any(s.lower() in str(cls).lower() for s in label_substrs):
                weights[i] *= factor

    # CRITICAL: If complaint is too vague (few words, single symptom), penalize specific diagnoses
    if word_count <= 3 and symptom_count == 0:
        # Very vague complaint like "fever" or "headache" - strongly penalize specific diagnoses
        mult(['Influenza', 'Bronchitis', 'Pneumonia', 'COVID-19'], 0.3)
        # Boost more general/symptom-based diagnoses
        mult(['Viral Upper Respiratory Infection', 'Fever of Unknown Origin', 'Viral Syndrome'], 1.5)
    
    # If ONLY fever mentioned (no other respiratory symptoms), don't jump to Influenza
    if has_fever and not has_cough and not has_sob and not has_throat and not has_runny_nose and not has_body_ache:
        mult(['Influenza'], 0.2)  # Strongly penalize Influenza for isolated fever
        mult(['Viral Upper Respiratory Infection'], 1.3)  # Prefer more general URI
    
    # Influenza requires multiple symptoms (fever + cough + body aches OR fever + multiple respiratory)
    if has_fever and has_cough and (has_body_ache or symptom_count >= 2):
        mult(['Influenza'], 1.8)  # Boost Influenza when multiple symptoms present
    elif has_fever and has_cough:
        mult(['Influenza'], 1.2)  # Mild boost for fever + cough
        mult(['Viral Upper Respiratory Infection', 'Bronchitis'], 1.5)  # But prefer these
    
    # Age-based priors: child/teen
    if age < 18:
        # Strongly suppress adult diseases
        mult(['COPD Exacerbation','Benign Prostatic Hyperplasia','Stable Angina','Acute Coronary Syndrome','Osteoarthritis','Type 2 Diabetes Mellitus','Gout'], 0.05)
        # Pediatric fever with multiple symptoms
        if has_fever and symptom_count >= 2:
            mult(['Viral Upper Respiratory Infection','Acute Viral Pharyngitis'], 1.6)
        # Stronger when fever + cough + other symptoms
        if has_fever and has_cough and symptom_count >= 2:
            mult(['Influenza','Viral Upper Respiratory Infection','Community-acquired Pneumonia'], 1.8)
        # If fever without chest wall pain wording, down-weight costochondritis
        if has_fever and not has_chest_pain:
            mult(['Costochondritis'], 0.2)
        # If pediatric fever without diabetes keywords, strongly suppress diabetes classes
        if has_fever and not has_dm_kw:
            for i, cls in enumerate(classes):
                if any(s in str(cls) for s in ['Diabetes Mellitus','Type 2 Diabetes Mellitus']):
                    weights[i] = 0.0

    # Respiratory pattern: fever + cough + high RR -> boost pneumonia
    if has_fever and has_cough and rr >= 22:
        mult(['Community-acquired Pneumonia'], 1.6)
    # Wheeze or sob terms could boost asthma
    if has_sob:
        mult(['Asthma Exacerbation'], 1.3)

    # General plausibility: if only fever present (no chest pain keywords), reduce chest wall / cardiac
    if has_fever and not has_chest_pain:
        mult(['Costochondritis','Stable Angina','Acute Coronary Syndrome'], 0.6)
        # And reduce chronic metabolic (e.g., T2DM) for isolated fever
        mult(['Type 2 Diabetes Mellitus'], 0.5)

    # If fever and NO rash-related words, down-weight dermatologic diagnoses
    if has_fever and not has_rash:
        mult(['Allergic Dermatitis','Urticaria','Cellulitis','Allergic Reaction'], 0.3)

    # If fever and NO abdominal terms, down-weight GI causes
    if has_fever and not has_abd:
        mult(['Gastritis','Gastroesophageal Reflux Disease','Peptic Ulcer Disease'], 0.3)

    # If fever + sore throat keywords, boost viral/bacterial pharyngitis
    if has_fever and has_throat:
        mult(['Acute Viral Pharyngitis','Acute Bacterial Tonsillitis','Mononucleosis'], 1.5)

    # Duration-based refinements
    if dur_days is not None:
        # Prolonged fever + cough (>7 days): shift toward pneumonia/URI, away from influenza
        if has_fever and has_cough and dur_days >= 7:
            mult(['Community-acquired Pneumonia','Viral Upper Respiratory Infection'], 1.5)
            mult(['Influenza'], 0.6)
        # Sore throat > 3 days: consider bacterial tonsillitis
        if has_throat and dur_days >= 3:
            mult(['Acute Bacterial Tonsillitis'], 1.6)
        # UTI-like symptoms > 2 days
        if has_urinary and dur_days >= 2:
            mult(['Urinary Tract Infection'], 1.6)
        # Many symptoms + prolonged duration suggests chronic inflammatory over allergic dermatitis
        if symptom_count >= 2 and dur_days >= 7:
            mult(['Allergic Dermatitis','Urticaria'], 0.5)

    # If fever without clear rash/urinary/abdomen/chest pain context and (child), gently prefer URI
    if has_fever and (not has_rash and not has_abd and not has_urinary and not has_chest_pain):
        # Stronger preference in children for viral URI/pharyngitis on isolated fever
        boost = 2.5 if age < 18 else 1.8
        mult(['Viral Upper Respiratory Infection','Influenza','Acute Viral Pharyngitis'], boost)
        # Gently penalize diabetes in isolated fever for all ages
        mult(['Diabetes Mellitus','Type 2 Diabetes Mellitus'], 0.3)

    # Fever with tachycardia (HR > 100) mildly boosts infectious etiologies
    if has_fever and hr > 100:
        mult(['Influenza','Viral Upper Respiratory Infection','Community-acquired Pneumonia','Acute Viral Pharyngitis'], 1.2)

    # Renormalize
    total = sum(weights)
    if total > 0:
        return [w/total for w in weights]
    return proba

def get_urgency_level(category: str) -> str:
    c = (category or '').lower()
    if 'category 3' in c:
        return 'EMERGENCY - Seek immediate medical attention'
    if 'category 2' in c:
        return 'URGENT - Medical evaluation needed within 24 hours'
    return 'ROUTINE - Basic wound care and monitoring'

def map_treatment_from_category(category: str) -> str:
    c = (category or '').lower()
    if 'category 3' in c:
        return ('Immediate wound washing; start Rabies Post-Exposure Prophylaxis (RPEP) + Rabies Immunoglobulin as indicated; '
                'tetanus update; antibiotics; urgent surgical evaluation if deep wounds.')
    if 'category 2' in c:
        return ('Wound cleaning; start Rabies Post-Exposure Prophylaxis (RPEP); tetanus update; '
                'consider oral antibiotics if high-risk wounds.')
    return ('Wash with soap and water; no RPEP needed if skin intact and animal healthy; observe; tetanus as per schedule.')

def normalize_animal_category_label(label: str) -> str:
    s = str(label or '')
    # If it already contains 'Animal Bite', keep it (ensure proper spacing)
    if re.search(r"animal\s*bite", s, flags=re.I):
        # Ensure 'Category N' is present
        m = re.search(r"Category\s*([123])", s, flags=re.I)
        if m:
            return f"Animal Bite Category {m.group(1)}"
        return "Animal Bite Category 2"
    # Extract just the category number and standardize the label
    m = re.search(r"Category\s*([123])", s, flags=re.I)
    if m:
        return f"Animal Bite Category {m.group(1)}"
    return "Animal Bite Category 2"

def analyze_vital_signs(age, bp_systolic, bp_diastolic, temperature_c, heart_rate_bpm, resp_rate_cpm):
    alerts = []
    try:
        age = float(age)
    except Exception:
        pass
    try:
        sbp = float(bp_systolic)
        dbp = float(bp_diastolic)
        temp = float(temperature_c)
        hr = float(heart_rate_bpm)
        rr = float(resp_rate_cpm)
    except Exception:
        return alerts

    # Blood pressure
    if sbp >= 140 or dbp >= 90:
        alerts.append('Elevated blood pressure')
    if sbp < 90 or dbp < 60:
        alerts.append('Low blood pressure')

    # Temperature (Celsius)
    if temp >= 38.0:
        alerts.append('Fever')
    if temp <= 35.0:
        alerts.append('Hypothermia risk')

    # Heart rate (bpm)
    if hr > 100:
        alerts.append('Tachycardia')
    if hr < 50:
        alerts.append('Bradycardia')

    # Respiratory rate (cycles per minute)
    if rr > 20:
        alerts.append('Tachypnea')
    if rr < 10:
        alerts.append('Bradypnea')

    return alerts

def ensure_log_header():
    if not os.path.exists(LOG_PATH):
        with open(LOG_PATH, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp','complaint','age','blood_pressure_systolic','blood_pressure_diastolic','temperature_c',
                'weight_kg','heart_rate_bpm','resp_rate_cpm',
                'top1','p1','top2','p2','top3','p3'
            ])

def log_prediction(payload, top3):
    ensure_log_header()
    with open(LOG_PATH, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now(timezone.utc).isoformat(),
            payload.get('complaint',''),
            payload['age'], payload['blood_pressure_systolic'], payload['blood_pressure_diastolic'], payload['temperature_c'],
            payload['weight_kg'], payload['heart_rate_bpm'], payload['resp_rate_cpm'],
            top3[0][0], f"{top3[0][1]:.4f}", top3[1][0], f"{top3[1][1]:.4f}", top3[2][0], f"{top3[2][1]:.4f}"
        ])

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json or {}
    complaint = data.get('complaint', '') or ''

    # Validation: require at least a complaint or any vital/age field explicitly provided
    has_complaint = bool(complaint.strip())
    provided_numeric = [k for k in DEFAULTS.keys() if str(data.get(k, '')).strip() != '']
    if not has_complaint and not provided_numeric:
        return jsonify({'error': 'No input provided. Please enter a chief complaint and/or vitals.'}), 400

    # Normalize incoming keys and gather numeric features with defaults
    features = {}
    features.update(DEFAULTS)

    # Accept both legacy keys (systolic_bp/diastolic_bp) and new keys (blood_pressure_*), prefer new
    if 'blood_pressure_systolic' in data:
        try:
            features['blood_pressure_systolic'] = int(data['blood_pressure_systolic'])
        except Exception:
            pass
    elif 'systolic_bp' in data:
        try:
            features['blood_pressure_systolic'] = int(data['systolic_bp'])
        except Exception:
            pass

    if 'blood_pressure_diastolic' in data:
        try:
            features['blood_pressure_diastolic'] = int(data['blood_pressure_diastolic'])
        except Exception:
            pass
    elif 'diastolic_bp' in data:
        try:
            features['blood_pressure_diastolic'] = int(data['diastolic_bp'])
        except Exception:
            pass

    # Temperature: accept temperature_c directly or convert Fahrenheit 'temperature'
    if 'temperature_c' in data:
        try:
            features['temperature_c'] = float(data['temperature_c'])
        except Exception:
            pass
    elif 'temperature' in data:
        try:
            features['temperature_c'] = (float(data['temperature']) - 32.0) * (5.0/9.0)
        except Exception:
            pass

    # Other numeric fields
    for k in ['age','weight_kg','heart_rate_bpm','resp_rate_cpm']:
        if k in data and data[k] not in (None, ''):
            try:
                features[k] = float(data[k]) if k in ['weight_kg','heart_rate_bpm','resp_rate_cpm'] else int(data[k])
            except Exception:
                pass

    # Aliases from frontend payload
    if 'weight' in data and data['weight'] not in (None, ''):
        try:
            features['weight_kg'] = float(data['weight'])
        except Exception:
            pass
    if 'heart_rate' in data and data['heart_rate'] not in (None, ''):
        try:
            features['heart_rate_bpm'] = float(data['heart_rate'])
        except Exception:
            pass
    if 'respiratory_rate' in data and data['respiratory_rate'] not in (None, ''):
        try:
            features['resp_rate_cpm'] = float(data['respiratory_rate'])
        except Exception:
            pass

    # Try to extract from complaint text if missing
    extracted = extract_vitals_from_text(complaint)
    for k, v in extracted.items():
        if k in features and (k not in data or data.get(k) in (None, '')):
            features[k] = v

    X = [{
        'complaint': complaint,
        'age': features['age'],
        'blood_pressure_systolic': features['blood_pressure_systolic'],
        'blood_pressure_diastolic': features['blood_pressure_diastolic'],
        'temperature_c': features['temperature_c'],
        'weight_kg': features['weight_kg'],
        'heart_rate_bpm': features['heart_rate_bpm'],
        'resp_rate_cpm': features['resp_rate_cpm'],
    }]

    # Predict top-3 (support both new model requiring DataFrame and legacy text-only model)
    try:
        df = pd.DataFrame(X)
        proba = model.predict_proba(df)[0]
        model_obj_for_classes = model
    except Exception:
        # Fallback: legacy pipeline expects list of complaint strings
        proba = model.predict_proba([complaint])[0]
        model_obj_for_classes = model
    # Support both Pipeline and direct estimator, and any classifier step name
    classes = None
    try:
        # Try direct attribute first
        classes = list(model_obj_for_classes.classes_)
    except Exception:
        try:
            # If Pipeline, get last step's classes_
            if hasattr(model_obj_for_classes, 'steps') and len(model_obj_for_classes.steps) > 0:
                classes = list(model_obj_for_classes.steps[-1][1].classes_)
        except Exception:
            classes = None
    if classes is None:
        return jsonify({'error': 'Model not ready: classes_ not found'}), 500
    # If the complaint text clearly lacks animal bite context, suppress those categories
    lower_text = (complaint or '').lower()
    bite_keywords = ['bite', 'bitten', 'rabies', 'dog', 'cat', 'monkey', 'bat', 'animal bite']
    has_bite_context = any(k in lower_text for k in bite_keywords)
    if not has_bite_context:
        # Zero probabilities for Animal Bite Category classes
        new_pairs = []
        total = 0.0
        for cls, p in zip(classes, proba):
            if isinstance(cls, str) and cls.lower().startswith('animal bite category'):
                new_pairs.append((cls, 0.0))
            else:
                new_pairs.append((cls, float(p)))
                total += float(p)
        # Renormalize if total > 0
        if total > 0:
            proba = [p/total for (_, p) in new_pairs]
            classes = [cls for (cls, _) in new_pairs]
        else:
            # If everything zero (shouldn't happen), keep original
            pass
    # Apply rule-based adjustments (e.g., pediatric fever -> boost influenza/URI)
    proba = apply_rule_based_adjustments(classes, proba, complaint, features)
    ranked = sorted(zip(classes, proba), key=lambda x: x[1], reverse=True)[:3]

    # Server-side fallback: if text indicates animal bite and dedicated model is available,
    # compute category and inject as top result so UI shows Category 1-3 reliably.
    if has_bite_context and animal_bite_category_model is not None:
        try:
            bite_df = pd.DataFrame([{
                'complaint': complaint,
                'age': features['age'],
                'blood_pressure_systolic': features['blood_pressure_systolic'],
                'blood_pressure_diastolic': features['blood_pressure_diastolic'],
                'temperature_c': features['temperature_c'],
                'weight_kg': features['weight_kg'],
                'heart_rate_bpm': features['heart_rate_bpm'],
                'resp_rate_cpm': features['resp_rate_cpm'],
            }])
            bite_proba = animal_bite_category_model.predict_proba(bite_df)[0]
            bite_classes = list(animal_bite_category_model.classes_)
            idx = int(np.argmax(bite_proba))
            bite_cat = bite_classes[idx]
            bite_p = float(bite_proba[idx])
        except Exception:
            bite_cat = animal_bite_category_model.predict(bite_df)[0]
            bite_p = 0.9

        # Normalize label to full form
        bite_cat = normalize_animal_category_label(bite_cat)
        # Prepend category if not already present as top-1
        ranked = [(bite_cat, bite_p)] + [pair for pair in ranked if pair[0] != bite_cat]
        ranked = ranked[:3]

    # Log
    to_log_payload = dict(X[0])
    log_prediction(to_log_payload, ranked)

    # Compatibility fields: top1 prediction and confidence, plus vitals analysis
    top1_diagnosis, top1_prob = ranked[0]
    return jsonify({
        'prediction': top1_diagnosis,
        'confidence': float(top1_prob),
        'vital_signs_analysis': analyze_vital_signs(
            features['age'], features['blood_pressure_systolic'], features['blood_pressure_diastolic'],
            features['temperature_c'], features['heart_rate_bpm'], features['resp_rate_cpm']
        ),
        'top3': [
            {'diagnosis': d, 'probability': float(p)} for d, p in ranked
        ]
    })

@app.route('/predict-animal-bite', methods=['POST'])
def predict_animal_bite():
    if animal_bite_category_model is None:
        return jsonify({'error': 'Animal bite category model not available'}), 500

    data = request.json or {}
    complaint = data.get('complaint', '') or ''

    # Reuse normalization path from /predict
    features = {}
    features.update(DEFAULTS)

    if 'blood_pressure_systolic' in data:
        try:
            features['blood_pressure_systolic'] = int(data['blood_pressure_systolic'])
        except Exception:
            pass
    elif 'systolic_bp' in data:
        try:
            features['blood_pressure_systolic'] = int(data['systolic_bp'])
        except Exception:
            pass

    if 'blood_pressure_diastolic' in data:
        try:
            features['blood_pressure_diastolic'] = int(data['blood_pressure_diastolic'])
        except Exception:
            pass
    elif 'diastolic_bp' in data:
        try:
            features['blood_pressure_diastolic'] = int(data['diastolic_bp'])
        except Exception:
            pass

    if 'temperature_c' in data:
        try:
            features['temperature_c'] = float(data['temperature_c'])
        except Exception:
            pass
    elif 'temperature' in data:
        try:
            features['temperature_c'] = (float(data['temperature']) - 32.0) * (5.0/9.0)
        except Exception:
            pass

    for k in ['age','weight_kg','heart_rate_bpm','resp_rate_cpm']:
        if k in data and data[k] not in (None, ''):
            try:
                features[k] = float(data[k]) if k in ['weight_kg','heart_rate_bpm','resp_rate_cpm'] else int(data[k])
            except Exception:
                pass

    # Try extract from complaint text
    extracted = extract_vitals_from_text(complaint)
    for k, v in extracted.items():
        if k in features and (k not in data or data.get(k) in (None, '')):
            features[k] = v

    df = pd.DataFrame([{
        'complaint': complaint,
        'age': features['age'],
        'blood_pressure_systolic': features['blood_pressure_systolic'],
        'blood_pressure_diastolic': features['blood_pressure_diastolic'],
        'temperature_c': features['temperature_c'],
        'weight_kg': features['weight_kg'],
        'heart_rate_bpm': features['heart_rate_bpm'],
        'resp_rate_cpm': features['resp_rate_cpm'],
    }])

    try:
        proba = animal_bite_category_model.predict_proba(df)[0]
        classes = list(animal_bite_category_model.classes_)
        idx = int(np.argmax(proba))
        category = classes[idx]
        confidence = float(proba[idx])
    except Exception:
        category = animal_bite_category_model.predict(df)[0]
    category = normalize_animal_category_label(category)
    confidence = 0.8

    return jsonify({
        'category': category,
        'category_confidence': confidence,
        'treatment': map_treatment_from_category(category),
        'urgency_level': get_urgency_level(category),
        'vital_signs_analysis': analyze_vital_signs(
            features['age'], features['blood_pressure_systolic'], features['blood_pressure_diastolic'],
            features['temperature_c'], features['heart_rate_bpm'], features['resp_rate_cpm']
        ),
    })

@app.route('/logs', methods=['GET'])
def download_logs():
    ensure_log_header()
    return send_file(LOG_PATH, as_attachment=True)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.getenv('PORT', 5000)))
