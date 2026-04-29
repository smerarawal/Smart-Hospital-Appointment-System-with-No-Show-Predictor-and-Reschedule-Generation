# Smart-Hospital-Appointment-System-with-No-Show-Predictor-and-Reschedule-Generation

---

# 🏥 MedFlow: AI-Powered Hospital Intelligence Platform

MedFlow is a full-stack web application built to solve a major operational problem in healthcare: **patient no-shows**.

By combining a **Random Forest ML model** with a real-time dashboard, MedFlow helps hospitals:

* Predict patient attendance
* Optimize scheduling with safe overbooking
* Improve patient reappointment strategies

---

## ✨ Features

### 🔹 1. Live ML Predictor (Inference Engine)

* Interactive UI with sliders for:

  * Age
  * Past no-shows
  * Cancellations
  * Lead time

* Backend:

  * Flask API processes inputs
  * Dynamic **One-Hot Encoding** using Pandas
  * Model inference using `.pkl` file

* Output:

  * Probability score (e.g., `0.854`)
  * Risk classification:

    * 🔴 High Risk
    * 🟡 Medium Risk
    * 🟢 Low Risk
  * Actionable recommendations

---

### 🔹 2. Full CRUD + Live Retraining Logic

* Editable predictions table

* Staff can update actual outcomes:

  * **Showed / No-show**

* Backend logic:

  * SQL `CASE` statement recalculates correctness
  * Real-time accuracy updates

* Frontend:

  * Uses `fetch` with `no-store`
  * No page reload required

* Additional:

  * Add new patients
  * Auto-generate primary keys

---

### 🔹 3. Smart Scheduling & Overbooking Engine

* Computes:

  * Expected attendance: `E(attendance)`

* Decision system:

  * Uses probabilities of two patients
  * Ensures clinic capacity is not exceeded

* Output:

  * ✅ ALLOW
  * ❌ DENY

---

### 🔹 4. Reappointment AI Optimizer

* Fetches patient history from MySQL

* Tests multiple combinations:

  * Days
  * Time slots
  * Lead times

* Runs model on all permutations

* Returns:

  * 🏆 Top 5 best appointment slots

---

### 🔹 5. Live SQL Analytics Dashboard

Real-time analytics using SQL queries (`JOIN`, `GROUP BY`, `AVG`, `SUM`):

* No-show rates by:

  * Department
  * Day of week

* Risk distribution:

  * Min / Max / Average

* Overall ML accuracy

---

## 🛠️ Tech Stack

### 💻 Frontend

* HTML5 / CSS3 (Dark Mode UI)
* Vanilla JavaScript (ES6+)
* Fetch API

### ⚙️ Backend

* Python 3
* Flask
* Pandas & NumPy
* Scikit-learn

### 🗄️ Database

* MySQL (Railway hosted)
* mysql-connector-python

---

## 🗄️ Database Design (ER Overview)

### 🔹 1. `synthetic_patients` (Strong Entity)

* **Primary Key:** `patient_id`

**Attributes:**

* age
* department
* day_of_week
* hour_of_day
* lead_time_days
* past_noshow_count
* past_cancellations

**Target Variable:**

* `no_show` (0 = Showed, 1 = Missed)

---

### 🔹 2. `synthetic_predictions` (Weak Entity)

* **Foreign Key:** `patient_id`

**Attributes:**

* noshow_probability
* risk_label
* model_used

**Derived Attribute:**

* `correct`

  * Computed dynamically using actual outcome

---

## 🚀 Setup Instructions

### ✅ Prerequisites

* Python 3.8+
* Internet connection (for Railway DB)

---

### 📥 1. Clone Repository

```bash
git clone https://github.com/yourusername/MedFlow.git
cd MedFlow
```

---

### 📦 2. Install Dependencies

```bash
pip install flask mysql-connector-python pandas numpy scikit-learn
```

---

### 🤖 3. Add ML Model

Place your trained model file:

```
rf_model.pkl
```

In the root directory (same level as `app.py`)

---

### ▶️ 4. Run Server

```bash
python app.py
```

---

### 🌐 5. Launch App

Open in browser:

```
http://localhost:5000
```

---

## 🧠 Key Strengths

* Real-time ML inference
* Closed feedback loop (prediction → correction → accuracy)
* Practical hospital use-case
* End-to-end system (UI + ML + DB + Analytics)

---

## 📌 Future Improvements

* Deep learning models (LSTM / Transformers)
* Patient notification system
* Integration with hospital EMR systems
* Deployment (Docker + Cloud)

---


