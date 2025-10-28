import csv
import re

# Cebuano translation dictionary
CEBUANO_TRANSLATIONS = {
    # Common verbs/phrases
    'patient reports': 'nag-report ang pasyente og',
    'experiencing': 'nakasinati og',
    'complains of': 'nagreklamo og',
    'noted': 'nakit-an nga adunay',
    'with': 'adunay',
    'came with': 'miabot nga adunay',
    'presented after': 'nagpakita pagkahuman sa',
    'reports': 'nag-report og',
    
    # Symptoms
    'fever and cough': 'hilanat ug ubo',
    'fever': 'hilanat',
    'cough': 'ubo',
    'chest pain': 'sakit sa dughan',
    'headache and dizziness': 'sakit sa ulo ug pagkaluya',
    'headache': 'sakit sa ulo',
    'dizziness': 'pagkaluya',
    'stomach pain after eating': 'sakit sa tiyan pagkahuman og kaon',
    'stomach pain': 'sakit sa tiyan',
    'shortness of breath': 'lisod ginhawa',
    'rash on skin': 'pantal sa panit',
    'rash': 'pantal',
    'frequent urination': 'kanunay nga pag-ihi',
    'sore throat': 'sakit sa tutunlan',
    'joint pain': 'sakit sa lutahan',
    'fatigue and weight loss': 'kakapoy ug pagkunhod sa timbang',
    'fatigue': 'kakapoy',
    'weight loss': 'pagkunhod sa timbang',
    
    # Animal bites
    'dog bite superficial on hand, no bleeding': 'gipaak sa iro nga dili lawom sa kamot, walay dugo',
    'dog bite with bleeding on leg': 'gipaak sa iro nga naay dugo sa tiil',
    'dog bite': 'gipaak sa iro',
    'cat scratch minor, superficial': 'gikud-aw sa iring nga gamay ug dili lawom',
    'cat scratch': 'gikud-aw sa iring',
    'lick by stray dog on intact skin': 'gilam-an sa iro nga walay puloy sa panit',
    'bat exposure in room, unsure of contact': 'naay kabog sa kwarto, dili sigurado kon naigo',
    'multiple dog bites with deep wounds': 'daghang paak sa iro nga lawom ang samad',
    
    # Qualifiers
    'after exercise': 'pagkahuman og ehersisyo',
    'for 2 days': 'sulod sa 2 ka adlaw',
    'for a week': 'sulod sa usa ka semana',
    'sudden onset': 'kalit lang',
    'worse at night': 'mas grabe sa gabii',
    'with mild chills': 'adunay kaguol',
    'no vomiting': 'walay pagsuka',
    'with nausea': 'adunay pagkaluya sa tiyan',
    'at rest': 'samtang nagpahulay',
    'gradual onset': 'hinay-hinay',
    'yesterday': 'kagahapon',
    'last night': 'kagabii',
    'this morning': 'karon buntag',
    'today': 'karon',
    '2 hours ago': '2 ka oras na ang milabay',
}

def translate_to_cebuano(english_text):
    """Translate English complaint to Cebuano"""
    text = english_text.strip().strip('"')
    
    # Start with the original text
    cebuano = text.lower()
    
    # Apply translations in order (more specific first)
    # Animal bites (specific patterns)
    cebuano = re.sub(r'dog bite superficial on hand, no bleeding', 'gipaak sa iro nga dili lawom sa kamot, walay dugo', cebuano)
    cebuano = re.sub(r'dog bite with bleeding on leg', 'gipaak sa iro nga naay dugo sa tiil', cebuano)
    cebuano = re.sub(r'cat scratch minor, superficial', 'gikud-aw sa iring nga gamay ug dili lawom', cebuano)
    cebuano = re.sub(r'lick by stray dog on intact skin', 'gilam-an sa iro nga walay puloy sa panit', cebuano)
    cebuano = re.sub(r'bat exposure in room, unsure of contact', 'naay kabog sa kwarto, dili sigurado kon naigo', cebuano)
    cebuano = re.sub(r'multiple dog bites with deep wounds', 'daghang paak sa iro nga lawom ang samad', cebuano)
    
    # Multi-word symptoms
    cebuano = re.sub(r'stomach pain after eating', 'sakit sa tiyan pagkahuman og kaon', cebuano)
    cebuano = re.sub(r'headache and dizziness', 'sakit sa ulo ug pagkaluya', cebuano)
    cebuano = re.sub(r'fever and cough', 'hilanat ug ubo', cebuano)
    cebuano = re.sub(r'chest pain', 'sakit sa dughan', cebuano)
    cebuano = re.sub(r'shortness of breath', 'lisod ginhawa', cebuano)
    cebuano = re.sub(r'rash on skin', 'pantal sa panit', cebuano)
    cebuano = re.sub(r'frequent urination', 'kanunay nga pag-ihi', cebuano)
    cebuano = re.sub(r'sore throat', 'sakit sa tutunlan', cebuano)
    cebuano = re.sub(r'joint pain', 'sakit sa lutahan', cebuano)
    cebuano = re.sub(r'fatigue and weight loss', 'kakapoy ug pagkunhod sa timbang', cebuano)
    
    # Verbs and phrases (at beginning)
    cebuano = re.sub(r'^patient reports', 'nag-report ang pasyente og', cebuano)
    cebuano = re.sub(r'^experiencing', 'nakasinati og', cebuano)
    cebuano = re.sub(r'^complains of', 'nagreklamo og', cebuano)
    cebuano = re.sub(r'^noted', 'nakit-an nga adunay', cebuano)
    cebuano = re.sub(r'^came with', 'miabot nga adunay', cebuano)
    cebuano = re.sub(r'^presented after', 'nagpakita pagkahuman sa', cebuano)
    cebuano = re.sub(r'^reports', 'nag-report og', cebuano)
    cebuano = re.sub(r'^with', 'adunay', cebuano)
    
    # Qualifiers (at end)
    cebuano = re.sub(r'after exercise$', 'pagkahuman og ehersisyo', cebuano)
    cebuano = re.sub(r'for 2 days$', 'sulod sa 2 ka adlaw', cebuano)
    cebuano = re.sub(r'for a week$', 'sulod sa usa ka semana', cebuano)
    cebuano = re.sub(r'sudden onset$', 'nga kalit lang', cebuano)
    cebuano = re.sub(r'worse at night$', 'nga mas grabe sa gabii', cebuano)
    cebuano = re.sub(r'with mild chills$', 'nga adunay kaguol', cebuano)
    cebuano = re.sub(r'no vomiting$', 'walay pagsuka', cebuano)
    cebuano = re.sub(r'with nausea$', 'nga adunay pagkaluya sa tiyan', cebuano)
    cebuano = re.sub(r'at rest$', 'samtang nagpahulay', cebuano)
    cebuano = re.sub(r'gradual onset$', 'nga hinay-hinay', cebuano)
    
    # Time expressions
    cebuano = re.sub(r'yesterday$', 'kagahapon', cebuano)
    cebuano = re.sub(r'last night$', 'kagabii', cebuano)
    cebuano = re.sub(r'this morning$', 'karon buntag', cebuano)
    cebuano = re.sub(r'today$', 'karon', cebuano)
    cebuano = re.sub(r'2 hours ago$', '2 ka oras na ang milabay', cebuano)
    
    return cebuano

def process_csv_file(input_file, output_file):
    """Add Cebuano translations to CSV file"""
    with open(input_file, 'r', encoding='utf-8', newline='') as infile:
        reader = csv.reader(infile)
        rows = list(reader)
    
    # Process each row (skip header)
    for i in range(1, len(rows)):
        if len(rows[i]) > 0:
            complaint = rows[i][0]
            # Check if already has translation (contains '/')
            if ' / ' not in complaint:
                cebuano = translate_to_cebuano(complaint)
                rows[i][0] = f"{complaint} / {cebuano}"
    
    # Write output
    with open(output_file, 'w', encoding='utf-8', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerows(rows)
    
    print(f"Processed {len(rows)-1} rows")
    print(f"Output saved to: {output_file}")

# Process both files
print("Adding Cebuano translations to complaints_expanded.csv...")
process_csv_file('complaints_expanded.csv', 'complaints_expanded_cebuano.csv')

print("\nAdding Cebuano translations to animal_bites.csv...")
process_csv_file('animal_bites.csv', 'animal_bites_cebuano.csv')

print("\nDone! Please review the output files and replace the originals if satisfied.")
