"""
Flask Backend — MedFlow No-Show Prediction System
"""
from flask import Flask, jsonify, request, send_from_directory
import mysql.connector
import numpy as np
import pandas as pd
import pickle
import os
import traceback

app = Flask(__name__)

# ── DB Config ─────────────────────────────────────────────────────────────────
DB_CONFIG = {
    'host':     'centerbeam.proxy.rlwy.net',
    'user':     'root',
    'password': 'uqqTtnLRCTBzVhUNDorIPByIknjemOAG', 
    'database': 'hospital_db',
    'port':     46001
}

# ── Load ML model ─────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'rf_model.pkl')
try:
    rf_model = pickle.load(open(MODEL_PATH, 'rb'))
    print("✅ RF model loaded")
except FileNotFoundError:
    rf_model = None
    print("⚠️  rf_model.pkl not found — put it in the same folder as app.py")

DEPARTMENTS = ['Cardiology','Dermatology','General','Oncology','Orthopaedics','Psychiatry']
DAYS        = ['Monday','Tuesday','Wednesday','Thursday','Friday']

def get_db():
    return mysql.connector.connect(**DB_CONFIG)

# ── The Smarter Encoder (Uses DataFrames) ────────────────────────────────────
def encode_patient(age, lead_time, past_noshow, past_cancel, hour, dept, day):
    columns = [
        'age', 'lead_time_days', 'past_noshow_count', 'past_cancellations', 'hour_of_day',
        'Cardiology', 'Dermatology', 'General', 'Oncology', 'Orthopaedics', 'Psychiatry',
        'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'
    ]
    features = [age, lead_time, past_noshow, past_cancel, hour]
    for d in DEPARTMENTS: features.append(1 if dept==d else 0)
    for d in DAYS:        features.append(1 if day==d else 0)
    return pd.DataFrame([features], columns=columns)

# ── ROUTES ────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory('static', filename)

@app.route('/api/dashboard')
def dashboard():
    try:
        conn = get_db(); cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT COUNT(*) AS total FROM synthetic_patients")
        total = cursor.fetchone()['total']
        cursor.execute("SELECT COUNT(*) AS high FROM synthetic_predictions WHERE risk_label='High'")
        high = cursor.fetchone()['high']
        cursor.execute("SELECT COUNT(*) AS medium FROM synthetic_predictions WHERE risk_label='Medium'")
        medium = cursor.fetchone()['medium']
        cursor.execute("SELECT COUNT(*) AS low FROM synthetic_predictions WHERE risk_label='Low'")
        low = cursor.fetchone()['low']
        cursor.execute("SELECT SUM(correct='1') AS correct, COUNT(*) AS total FROM synthetic_predictions")
        acc = cursor.fetchone()
        accuracy = round(acc['correct'] / acc['total'] * 100, 1) if acc['total'] else 0
        cursor.execute("SELECT department, ROUND(AVG(no_show)*100,1) AS noshow_pct FROM synthetic_patients GROUP BY department ORDER BY noshow_pct DESC")
        by_dept = cursor.fetchall()
        cursor.execute("SELECT day_of_week, ROUND(AVG(no_show)*100,1) AS noshow_pct FROM synthetic_patients GROUP BY day_of_week ORDER BY noshow_pct DESC")
        by_day = cursor.fetchall()
        cursor.close(); conn.close()
        return jsonify({'total': total, 'high': high, 'medium': medium, 'low': low, 'accuracy': accuracy, 'by_dept': by_dept, 'by_day': by_day})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ── API: Predict single patient (SHIELD ENABLED) ─────────────────────────────
@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.json
    if rf_model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    try:
        def safe_int(val, default=0):
            try:
                if val is None or str(val).strip() == '': return default
                return int(float(val))
            except (ValueError, TypeError):
                return default

        age = safe_int(data.get('age'), 30)
        lead_time = safe_int(data.get('lead_time_days'), 0)
        noshows = safe_int(data.get('past_noshow_count'), 0)
        cancellations = safe_int(data.get('past_cancellations'), 0)
        
        hour_val = data.get('hour_of_day', data.get('hour', 10))
        hour = int(str(hour_val).split(':')[0]) if ':' in str(hour_val) and hour_val else safe_int(hour_val, 10)

        features_df = encode_patient(age, lead_time, noshows, cancellations, hour, data.get('department', 'General'), data.get('day_of_week', 'Monday'))
        prob = float(rf_model.predict_proba(features_df)[0][1])

        risk = 'High' if prob>=0.6 else ('Medium' if prob>=0.35 else 'Low')
        rec = "High no-show risk." if prob >= 0.6 else ("Moderate risk." if prob >= 0.35 else "Low risk.")
        return jsonify({'probability': round(prob, 3), 'risk': risk, 'recommendation': rec})

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# ── API: Add New Patient to Database ──────────────────────────────────────────
@app.route('/api/patient/add', methods=['POST'])
def add_patient():
    print("\n➕ --- ADDING NEW PATIENT TO DATABASE ---")
    if rf_model is None:
        return jsonify({'error': 'Model not loaded'}), 500
        
    try:
        data = request.json
        conn = get_db()
        cursor = conn.cursor(dictionary=True)
        
        # 1. Generate a new Patient ID
        cursor.execute("SELECT COALESCE(MAX(patient_id), 0) + 1 AS next_id FROM synthetic_patients")
        new_id = cursor.fetchone()['next_id']
        
        # 2. Extract inputs securely
        age = int(data.get('age', 30))
        lead_time = int(data.get('lead_time_days', 0))
        noshows = int(data.get('past_noshow_count', 0))
        cancellations = int(data.get('past_cancellations', 0))
        hour = int(str(data.get('hour_of_day', 10)).split(':')[0])
        dept = data.get('department', 'General')
        day = data.get('day_of_week', 'Monday')
        
        # 3. Run the ML Prediction
        features_df = encode_patient(age, lead_time, noshows, cancellations, hour, dept, day)
        prob = float(rf_model.predict_proba(features_df)[0][1])
        risk = 'High' if prob >= 0.6 else ('Medium' if prob >= 0.35 else 'Low')
        
        default_actual_status = 0 # Default to 0 (Showed) for a new, future appointment
        is_correct = '1' if (prob < 0.5) else '0' 

        # 4. Save to Patients Table
        cursor.execute("""
            INSERT INTO synthetic_patients 
            (patient_id, age, department, day_of_week, hour_of_day, lead_time_days, past_noshow_count, past_cancellations, true_risk_score, no_show)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (new_id, age, dept, day, hour, lead_time, noshows, cancellations, prob, default_actual_status))

        # 5. Save to Predictions Table
        cursor.execute("""
            INSERT INTO synthetic_predictions 
            (patient_id, noshow_probability, risk_label, model_used, correct)
            VALUES (%s, %s, %s, %s, %s)
        """, (new_id, prob, risk, 'RandomForest v2 (Live)', is_correct))
        
        conn.commit()
        cursor.close()
        conn.close()
        
        print(f"✅ Patient {new_id} saved successfully!")
        return jsonify({'success': True, 'new_patient_id': new_id, 'probability': prob, 'risk': risk})
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/predictions')
def predictions():
    try:
        conn = get_db(); cursor = conn.cursor(dictionary=True)
        cursor.execute("""
            SELECT sp.patient_id, sp.age, sp.department, sp.day_of_week, sp.past_noshow_count, sp.lead_time_days, sp.no_show AS actual, sq.noshow_probability, sq.risk_label, sq.model_used, sq.correct
            FROM synthetic_patients sp JOIN synthetic_predictions sq ON sp.patient_id = sq.patient_id ORDER BY sq.noshow_probability DESC
        """)
        rows = cursor.fetchall()
        cursor.close(); conn.close()
        return jsonify(rows)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ── API: Update Patient Attendance Live ────────────────────────────────────────
@app.route('/api/patient/<int:patient_id>/status', methods=['POST'])
def update_status(patient_id):
    print(f"\n🚀 --- INCOMING UPDATE FOR PATIENT ID: {patient_id} ---")
    try:
        data = request.json
        new_no_show = int(data['no_show']) 
        
        conn = get_db()
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE synthetic_patients 
            SET no_show = %s 
            WHERE patient_id = %s
        """, (new_no_show, patient_id))
        
        cursor.execute("""
            UPDATE synthetic_predictions
            SET correct = CASE 
                WHEN noshow_probability >= 0.5 AND %s = 1 THEN '1'
                WHEN noshow_probability < 0.5 AND %s = 0 THEN '1'
                ELSE '0'
            END
            WHERE patient_id = %s
        """, (new_no_show, new_no_show, patient_id))
        
        conn.commit()
        cursor.close()
        conn.close()
        
        print(f"✅ Patient {patient_id} updated successfully!")
        return jsonify({'success': True, 'message': 'Database updated!'})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/overbook', methods=['POST'])
def overbook():
    data = request.json
    p1, p2 = float(data['p1']), float(data['p2'])
    show1, show2 = round(1-p1, 3), round(1-p2, 3)
    E = round(show1+show2, 3)
    both_high = p1>=0.60 and p2>=0.60
    safe = E<=int(data.get('capacity', 1))
    return jsonify({'p1': p1, 'p2': p2, 'show1': show1, 'show2': show2, 'expected_attendance': E, 'both_high_risk': both_high, 'safe': safe, 'decision': "ALLOW OVERBOOKING" if both_high and safe else "DENY"})

@app.route('/api/reappoint/<int:patient_id>')
def reappoint(patient_id):
    if rf_model is None: return jsonify({'error': 'Model not loaded'}), 500
    try:
        conn = get_db(); cursor = conn.cursor(dictionary=True)
        cursor.execute(f"SELECT age, past_noshow_count, past_cancellations, department, day_of_week, hour_of_day FROM synthetic_patients WHERE patient_id={patient_id} LIMIT 1")
        patient = cursor.fetchone()
        cursor.close(); conn.close()
        if not patient: return jsonify({'error': 'Patient not found'}), 404

        candidates = []
        for day in DAYS:
            for hour in [8,9,10,11,13,14,15]:
                for lead in [3,7,14,21]:
                    features_df = encode_patient(patient['age'], lead, patient['past_noshow_count'], patient['past_cancellations'], hour, patient['department'], day)
                    prob = float(rf_model.predict_proba(features_df)[0][1])
                    candidates.append({'day': day, 'time': f"{hour:02d}:00", 'lead_days': lead, 'noshow_prob': round(prob, 3), 'show_prob': round(1-prob, 3)})
        candidates.sort(key=lambda x: x['noshow_prob'])
        return jsonify({'patient': patient, 'recommendations': candidates[:5]})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/analysis/<query_name>')
def analysis(query_name):
    queries = {
        'by_dept': "SELECT department, COUNT(*) AS total, SUM(no_show) AS noshows, ROUND(AVG(no_show)*100,1) AS noshow_pct FROM synthetic_patients GROUP BY department ORDER BY noshow_pct DESC",
        'by_day': "SELECT day_of_week, COUNT(*) AS total, ROUND(AVG(no_show)*100,1) AS noshow_pct FROM synthetic_patients GROUP BY day_of_week ORDER BY noshow_pct DESC",
        'risk_dist': "SELECT risk_label, COUNT(*) AS total, ROUND(AVG(noshow_probability)*100,1) AS avg_prob_pct, ROUND(MIN(noshow_probability),3) AS min_prob, ROUND(MAX(noshow_probability),3) AS max_prob FROM synthetic_predictions GROUP BY risk_label ORDER BY avg_prob_pct DESC",
        'accuracy': "SELECT COUNT(*) AS total, SUM(correct='1') AS correct, ROUND(SUM(correct='1')*100.0/COUNT(*),1) AS accuracy_pct FROM synthetic_predictions",
        'high_risk': "SELECT sp.patient_id, sp.age, sp.department, sp.past_noshow_count, sp.lead_time_days, sq.noshow_probability, sq.risk_label FROM synthetic_patients sp JOIN synthetic_predictions sq ON sp.patient_id = sq.patient_id WHERE sq.risk_label = 'High' ORDER BY sq.noshow_probability DESC"
    }
    if query_name not in queries: return jsonify({'error': 'Unknown query'}), 400
    try:
        conn = get_db(); cursor = conn.cursor(dictionary=True)
        cursor.execute(queries[query_name])
        rows = cursor.fetchall()
        cursor.close(); conn.close()
        return jsonify(rows)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("\n🏥 MedFlow Intelligence Platform")
    print("   Running on http://localhost:5000\n")
    app.run(debug=True, port=5000)