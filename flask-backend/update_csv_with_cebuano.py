"""
Script to add Cebuano translations to CSV training data files
This will make the AI understand both English and Cebuano/Bisaya complaints
"""

import pandas as pd
import re

def translate_complaint_to_cebuano(complaint):
    """
    Translate English complaint to Cebuano/Bisaya
    Returns the complaint with Cebuano translation appended
    """
    # Remove quotes
    text = complaint.strip().strip('"\'')
    
    # Create Cebuano translation
    cebuano = text.lower()
    
    # Translate action verbs (beginning of sentence)
    cebuano = re.sub(r'^reports ', 'nag-report og ', cebuano)
    cebuano = re.sub(r'^came with ', 'miabot nga adunay ', cebuano)
    cebuano = re.sub(r'^presented after ', 'nagpakita pagkahuman sa ', cebuano)
    cebuano = re.sub(r'^complains of ', 'nagreklamo og ', cebuano)
    cebuano = re.sub(r'^experiencing ', 'nakasinati og ', cebuano)
    cebuano = re.sub(r'^patient reports ', 'nag-report ang pasyente og ', cebuano)
    cebuano = re.sub(r'^noted ', 'nakit-an nga adunay ', cebuano)
    cebuano = re.sub(r'^with ', 'adunay ', cebuano)
    
    # Translate animal bite symptoms
    cebuano = re.sub(r'dog bite superficial on hand, no bleeding', 'gipaak sa iro sa kamot nga dili lawom, walay dugo', cebuano)
    cebuano = re.sub(r'dog bite with bleeding on leg', 'gipaak sa iro sa tiil nga naay dugo', cebuano)
    cebuano = re.sub(r'dog bite', 'gipaak sa iro', cebuano)
    cebuano = re.sub(r'cat scratch minor, superficial', 'gikud-aw sa iring nga gamay ug dili lawom', cebuano)
    cebuano = re.sub(r'lick by stray dog on intact skin', 'gilam-an sa suroy nga iro sa panit nga walay samad', cebuano)
    cebuano = re.sub(r'bat exposure in room, unsure of contact', 'naay kabog sa kwarto, dili sigurado kon naigo', cebuano)
    cebuano = re.sub(r'multiple dog bites with deep wounds', 'daghang paak sa iro nga lawom ang samad', cebuano)
    
    # Translate general symptoms
    cebuano = re.sub(r'fever and cough', 'hilanat ug ubo', cebuano)
    cebuano = re.sub(r'chest pain', 'sakit sa dughan', cebuano)
    cebuano = re.sub(r'headache and dizziness', 'sakit sa ulo ug pagkaluya', cebuano)
    cebuano = re.sub(r'stomach pain after eating', 'sakit sa tiyan pagkahuman og kaon', cebuano)
    cebuano = re.sub(r'shortness of breath', 'lisod ginhawa', cebuano)
    cebuano = re.sub(r'rash on skin', 'pantal sa panit', cebuano)
    cebuano = re.sub(r'frequent urination', 'kanunay nga pag-ihi', cebuano)
    cebuano = re.sub(r'sore throat', 'sakit sa tutunlan', cebuano)
    cebuano = re.sub(r'joint pain', 'sakit sa lutahan', cebuano)
    cebuano = re.sub(r'fatigue and weight loss', 'kakapoy ug pagkunhod sa timbang', cebuano)
    
    # Translate time expressions
    cebuano = re.sub(r'2 hours ago', '2 ka oras na ang milabay', cebuano)
    cebuano = re.sub(r'last night', 'kagabii', cebuano)
    cebuano = re.sub(r'this morning', 'karon buntag', cebuano)
    cebuano = re.sub(r'yesterday', 'kagahapon', cebuano)
    cebuano = re.sub(r'today', 'karon', cebuano)
    cebuano = re.sub(r'for 2 days', 'sulod sa 2 ka adlaw', cebuano)
    cebuano = re.sub(r'for a week', 'sulod sa usa ka semana', cebuano)
    
    # Translate qualifiers
    cebuano = re.sub(r'worse at night', 'mas grabe sa gabii', cebuano)
    cebuano = re.sub(r'with mild chills', 'adunay kaguol', cebuano)
    cebuano = re.sub(r'no vomiting', 'walay pagsuka', cebuano)
    cebuano = re.sub(r'with nausea', 'adunay pagkaluya sa tiyan', cebuano)
    cebuano = re.sub(r'at rest', 'samtang nagpahulay', cebuano)
    cebuano = re.sub(r'gradual onset', 'hinay-hinay', cebuano)
    cebuano = re.sub(r'sudden onset', 'kalit lang', cebuano)
    cebuano = re.sub(r'after exercise', 'pagkahuman og ehersisyo', cebuano)
    
    # Combine English and Cebuano
    return f"{text} / {cebuano}"

def process_csv_file(input_file):
    """Process CSV file and add Cebuano translations"""
    print(f"\nProcessing {input_file}...")
    
    # Read CSV
    df = pd.read_csv(input_file)
    
    # Add Cebuano translations to complaint column
    df['complaint'] = df['complaint'].apply(translate_complaint_to_cebuano)
    
    # Save back to original file
    df.to_csv(input_file, index=False)
    
    print(f"✓ Updated {len(df)} rows with Cebuano translations")
    print(f"Sample: {df['complaint'].iloc[0][:100]}...")

# Process both files
if __name__ == "__main__":
    print("="*80)
    print("Adding Cebuano/Bisaya translations to AI training data")
    print("="*80)
    
    process_csv_file('animal_bites.csv')
    process_csv_file('complaints_expanded.csv')
    
    print("\n" + "="*80)
    print("✓ COMPLETE! Both files now have English/Cebuano bilingual data")
    print("="*80)
    print("\nNext step: Retrain the AI model to recognize Cebuano complaints")
    print("Run: python train_model.py")
