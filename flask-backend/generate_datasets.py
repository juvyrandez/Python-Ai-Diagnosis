import os
import random
import csv

random.seed(42)

COMPLAINTS_SOURCE = 'complaints.csv'
EXPANDED_OUT = 'complaints_expanded.csv'
ANIMAL_BITE_OUT = 'animal_bites.csv'

# Basic seed complaints if the source is minimal
SEED_COMPLAINTS = [
    ("fever and cough", "Flu or Respiratory Infection"),
    ("chest pain", "Possible Heart Issue or Angina"),
    ("headache and dizziness", "Migraine or Hypertension"),
    ("stomach pain after eating", "Gastritis or Acid Reflux"),
    ("shortness of breath", "Asthma or Pneumonia"),
    ("rash on skin", "Allergic Reaction or Skin Infection"),
    ("frequent urination", "Urinary Tract Infection or Diabetes"),
    ("sore throat", "Tonsillitis or Viral Infection"),
    ("joint pain", "Arthritis or Autoimmune Disorder"),
    ("fatigue and weight loss", "Diabetes or Thyroid Disorder"),
]

# Augmentation variants to produce richer text
AUG_PREFIXES = [
    "patient reports", "complains of", "noted", "with", "experiencing",
]
AUG_SUFFIXES = [
    "for 2 days", "for a week", "sudden onset", "gradual onset", "worse at night",
    "with mild chills", "with nausea", "no vomiting", "after exercise", "at rest",
]

# Diagnosis alternatives map to produce 3-way variety
DX_ALTERNATIVES = {
    "Flu or Respiratory Infection": [
        "Influenza", "Viral Upper Respiratory Infection", "Bronchitis"
    ],
    "Possible Heart Issue or Angina": [
        "Stable Angina", "Acute Coronary Syndrome", "Costochondritis"
    ],
    "Migraine or Hypertension": [
        "Migraine", "Tension Headache", "Hypertensive Urgency"
    ],
    "Gastritis or Acid Reflux": [
        "Gastritis", "Gastroesophageal Reflux Disease", "Peptic Ulcer Disease"
    ],
    "Asthma or Pneumonia": [
        "Asthma Exacerbation", "Community-acquired Pneumonia", "COPD Exacerbation"
    ],
    "Allergic Reaction or Skin Infection": [
        "Allergic Dermatitis", "Urticaria", "Cellulitis"
    ],
    "Urinary Tract Infection or Diabetes": [
        "Urinary Tract Infection", "Benign Prostatic Hyperplasia", "Diabetes Mellitus"
    ],
    "Tonsillitis or Viral Infection": [
        "Acute Viral Pharyngitis", "Acute Bacterial Tonsillitis", "Mononucleosis"
    ],
    "Arthritis or Autoimmune Disorder": [
        "Osteoarthritis", "Rheumatoid Arthritis", "Gout"
    ],
    "Diabetes or Thyroid Disorder": [
        "Type 2 Diabetes Mellitus", "Hypothyroidism", "Hyperthyroidism"
    ],
}

# Utility to sample realistic vitals by age bucket

def sample_vitals(age: int):
    # Simple synthetic distributions
    if age < 12:
        sys = random.randint(90, 110)
        dia = random.randint(55, 70)
        hr = random.randint(80, 110)
        rr = random.randint(20, 28)
        temp = round(random.uniform(36.5, 38.5), 1)
        w = random.randint(15, 45)
    elif age < 60:
        sys = random.randint(105, 135)
        dia = random.randint(65, 85)
        hr = random.randint(60, 95)
        rr = random.randint(12, 20)
        temp = round(random.uniform(36.3, 38.2), 1)
        w = random.randint(45, 110)
    else:
        sys = random.randint(110, 150)
        dia = random.randint(60, 90)
        hr = random.randint(55, 90)
        rr = random.randint(12, 22)
        temp = round(random.uniform(36.2, 38.0), 1)
        w = random.randint(45, 100)
    return sys, dia, temp, w, hr, rr


def read_source_or_seed():
    rows = []
    if os.path.exists(COMPLAINTS_SOURCE):
        with open(COMPLAINTS_SOURCE, newline='', encoding='utf-8') as f:
            r = csv.DictReader(f)
            for row in r:
                comp = row.get('complaint', '').strip()
                dx = row.get('diagnosis', '').strip()
                if comp and dx:
                    rows.append((comp, dx))
    if not rows:
        rows = SEED_COMPLAINTS
    return rows


def generate_expanded():
    rows = read_source_or_seed()
    out_rows = []

    for comp, dx in rows:
        alts = DX_ALTERNATIVES.get(dx, [dx])
        for _ in range(30):  # 30 variants per seed -> ~300 rows
            age = random.randint(1, 90)
            sys, dia, temp, w, hr, rr = sample_vitals(age)
            prefix = random.choice(AUG_PREFIXES)
            suffix = random.choice(AUG_SUFFIXES)
            text = f"{prefix} {comp} {suffix}"
            # pick a likely diagnosis among three options for training diversity
            label = random.choice(alts)
            out_rows.append({
                'complaint': text,
                'diagnosis': label,
                'age': age,
                'systolic_bp': sys,
                'diastolic_bp': dia,
                'temperature_c': temp,
                'weight_kg': w,
                'heart_rate_bpm': hr,
                'resp_rate_cpm': rr,
            })

    with open(EXPANDED_OUT, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'complaint','diagnosis','age','systolic_bp','diastolic_bp','temperature_c',
            'weight_kg','heart_rate_bpm','resp_rate_cpm'
        ])
        writer.writeheader()
        writer.writerows(out_rows)


def generate_animal_bite():
    # WHO categories roughly:
    # Cat 1: touching/feeding animals, licks on intact skin
    # Cat 2: nibbling of uncovered skin, minor scratches without bleeding
    # Cat 3: single or multiple transdermal bites or scratches, mucous membrane exposures
    templates = [
        ("dog bite superficial on hand, no bleeding", 2),
        ("dog bite with bleeding on leg", 3),
        ("cat scratch minor, superficial", 2),
        ("lick by stray dog on intact skin", 1),
        ("bat exposure in room, unsure of contact", 3),
        ("multiple dog bites with deep wounds", 3),
        ("puppy lick on intact skin while playing", 1),
        ("monkey scratch minor without bleeding", 2),
        ("fox bite through clothing", 3),
        ("rat bite with puncture wound", 3),
    ]

    out_rows = []
    for text, cat in templates:
        for _ in range(60):  # 600 rows
            age = random.randint(2, 80)
            sys, dia, temp, w, hr, rr = sample_vitals(age)
            prefix = random.choice(["reports", "complains of", "came with", "presented after"])
            suffix = random.choice(["today", "yesterday", "2 hours ago", "this morning", "last night"])            
            comp = f"{prefix} {text} {suffix}"
            out_rows.append({
                'complaint': comp,
                'diagnosis': f"Animal Bite Category {cat}",
                'age': age,
                'systolic_bp': sys,
                'diastolic_bp': dia,
                'temperature_c': temp,
                'weight_kg': w,
                'heart_rate_bpm': hr,
                'resp_rate_cpm': rr,
            })

    with open(ANIMAL_BITE_OUT, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'complaint','diagnosis','age','systolic_bp','diastolic_bp','temperature_c',
            'weight_kg','heart_rate_bpm','resp_rate_cpm'
        ])
        writer.writeheader()
        writer.writerows(out_rows)


if __name__ == '__main__':
    generate_expanded()
    generate_animal_bite()
    print(f"Wrote {EXPANDED_OUT} and {ANIMAL_BITE_OUT}")
