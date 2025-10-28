"""
Test script to verify the AI understands Cebuano/Bisaya complaints
"""
import requests
import json

API_URL = "http://localhost:5000/predict"

# Test cases with Cebuano complaints
test_cases = [
    {
        "name": "Test 1: Cebuano fever and cough",
        "payload": {
            "complaint": "nagreklamo og hilanat ug ubo nga adunay kaguol",
            "age": 30,
            "blood_pressure_systolic": 132,
            "blood_pressure_diastolic": 68,
            "temperature_c": 37.2,
            "weight_kg": 91,
            "heart_rate_bpm": 84,
            "resp_rate_cpm": 16
        }
    },
    {
        "name": "Test 2: Cebuano dog bite",
        "payload": {
            "complaint": "gipaak sa iro sa kamot nga dili lawom, walay dugo karon buntag",
            "age": 25,
            "blood_pressure_systolic": 120,
            "blood_pressure_diastolic": 80,
            "temperature_c": 37.0,
            "weight_kg": 65,
            "heart_rate_bpm": 75,
            "resp_rate_cpm": 16
        }
    },
    {
        "name": "Test 3: Cebuano headache and dizziness",
        "payload": {
            "complaint": "nakasinati og sakit sa ulo ug pagkaluya sulod sa 2 ka adlaw",
            "age": 45,
            "blood_pressure_systolic": 148,
            "blood_pressure_diastolic": 88,
            "temperature_c": 36.6,
            "weight_kg": 72,
            "heart_rate_bpm": 69,
            "resp_rate_cpm": 19
        }
    },
    {
        "name": "Test 4: Mixed English-Cebuano",
        "payload": {
            "complaint": "patient reports hilanat ug ubo for 3 days",
            "age": 12,
            "blood_pressure_systolic": 110,
            "blood_pressure_diastolic": 70,
            "temperature_c": 38.0,
            "weight_kg": 40,
            "heart_rate_bpm": 90,
            "resp_rate_cpm": 20
        }
    },
    {
        "name": "Test 5: Cebuano stomach pain",
        "payload": {
            "complaint": "adunay sakit sa tiyan pagkahuman og kaon sulod sa usa ka semana",
            "age": 35,
            "blood_pressure_systolic": 125,
            "blood_pressure_diastolic": 80,
            "temperature_c": 37.0,
            "weight_kg": 70,
            "heart_rate_bpm": 75,
            "resp_rate_cpm": 16
        }
    }
]

print("="*80)
print("TESTING AI DIAGNOSIS WITH CEBUANO/BISAYA LANGUAGE")
print("="*80)

for test in test_cases:
    print(f"\n{test['name']}")
    print(f"Complaint: {test['payload']['complaint']}")
    print("-" * 60)
    
    try:
        response = requests.post(API_URL, json=test['payload'], timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            
            # Display top 3 diagnoses
            if 'top3' in result:
                print("✅ AI Diagnosis Results:")
                for i, dx in enumerate(result['top3'], 1):
                    print(f"   {i}. {dx['diagnosis']}: {dx['probability']*100:.1f}%")
            else:
                print(f"   Primary: {result.get('prediction', 'N/A')}")
                print(f"   Confidence: {result.get('confidence', 0)*100:.1f}%")
                
            # Display vital signs analysis
            if result.get('vital_signs_analysis'):
                print(f"   Alerts: {', '.join(result['vital_signs_analysis']) if result['vital_signs_analysis'] else 'None'}")
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"   {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Error: Cannot connect to Flask backend")
        print("   Please make sure the Flask server is running:")
        print("   python app.py")
        break
    except Exception as e:
        print(f"❌ Error: {str(e)}")

print("\n" + "="*80)
print("Testing complete!")
print("="*80)
