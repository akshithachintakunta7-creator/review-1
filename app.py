from flask import Flask, request, jsonify
import os
import requests

from datetime import datetime, timedelta
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_jwt_extended import (
    JWTManager,
    create_access_token,
    jwt_required,
    get_jwt_identity
)
from itsdangerous import URLSafeTimedSerializer
from flask_cors import CORS
from flask_mail import Mail, Message
from sqlalchemy import func

# =========================================================
# APP CONFIG
# =========================================================

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv("DATABASE_URL")
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config["JWT_SECRET_KEY"] = os.getenv("JWT_SECRET_KEY")
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY")

# Email config
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = os.getenv("MAIL_USERNAME")
app.config['MAIL_PASSWORD'] = os.getenv("MAIL_PASSWORD")

db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
jwt = JWTManager(app)
serializer = URLSafeTimedSerializer(app.config["SECRET_KEY"])
mail = Mail(app)
CORS(app)

# =========================================================
# HUGGINGFACE CONFIG
# =========================================================

HF_TOKEN = os.getenv("HF_TOKEN")

HF_CLASSIFY_URL = "https://api-inference.huggingface.co/models/distilbert-base-uncased"
HF_CHAT_URL = "https://api-inference.huggingface.co/models/google/flan-t5-base"

hf_headers = {
    "Authorization": f"Bearer {HF_TOKEN}"
}

def hf_classify(text):
    response = requests.post(HF_CLASSIFY_URL, headers=hf_headers, json={"inputs": text})
    result = response.json()
    if not isinstance(result, list):
        return 0, 0.5
    top = result[0]
    prediction = 1 if top["label"] == "LABEL_1" else 0
    confidence = float(top["score"])
    return prediction, confidence

def hf_chat(prompt):
    response = requests.post(HF_CHAT_URL, headers=hf_headers, json={"inputs": prompt})
    result = response.json()
    if isinstance(result, list):
        return result[0]["generated_text"]
    return "I'm unable to respond right now."

# =========================================================
# EMAIL ALERT FUNCTION
# =========================================================

def send_health_alert(user_email, condition, risk_score):
    try:
        msg = Message(
            subject="⚠ High Health Risk Detected",
            sender=app.config['MAIL_USERNAME'],
            recipients=[user_email]
        )
        msg.body = f"""
High Risk Alert

Condition: {condition}
Risk Score: {risk_score}

Please consult a certified medical professional.

Women Support AI Platform
"""
        mail.send(msg)
    except Exception as e:
        print("Email Error:", e)

# =========================================================
# DATABASE MODELS
# =========================================================

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100))
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(20), default="user")
    is_verified = db.Column(db.Boolean, default=False)
    cyber_strikes = db.Column(db.Integer, default=0)
    is_blocked = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class CyberLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer)
    text = db.Column(db.Text)
    prediction = db.Column(db.String(50))
    confidence = db.Column(db.Float)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class LegalLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    text = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class HealthLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    symptoms = db.Column(db.Text)
    prediction = db.Column(db.String(100))
    confidence = db.Column(db.Float)
    risk_score = db.Column(db.Float)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class PeriodLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer)
    last_period = db.Column(db.String(20))
    avg_cycle = db.Column(db.Integer)
    next_period = db.Column(db.String(20))
    ovulation_day = db.Column(db.String(20))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class SkillLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer)
    skills = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

# =========================================================
# CYBER DETECTION
# =========================================================

@app.route("/api/cyber", methods=["POST"])
@jwt_required()
def detect_cyber():
    text = request.json.get("text", "")
    identity = get_jwt_identity()
    user = User.query.get(identity["id"])

    if user.is_blocked:
        return jsonify({"error": "Account blocked"}), 403

    prediction, confidence = hf_classify(text)
    label = "Cyberbullying" if prediction == 1 else "Safe"

    if label == "Cyberbullying" and confidence >= 0.75:
        user.cyber_strikes += 1
        if user.cyber_strikes >= 3:
            user.is_blocked = True

    db.session.add(CyberLog(
        user_id=user.id,
        text=text,
        prediction=label,
        confidence=confidence
    ))
    db.session.commit()

    return jsonify({
        "prediction": label,
        "confidence": confidence,
        "strikes": user.cyber_strikes,
        "blocked": user.is_blocked
    })

# =========================================================
# LEGAL CHATBOT
# =========================================================

@app.route("/api/legal", methods=["POST"])
def legal_chat():
    user_text = request.json.get("text", "")
    reply = hf_chat(f"You are a women's legal assistant. {user_text}")
    db.session.add(LegalLog(text=user_text))
    db.session.commit()
    return jsonify({"reply": reply})

# =========================================================
# SKILL TO INCOME
# =========================================================

@app.route("/api/skill-income", methods=["POST"])
@jwt_required()
def skill_income():
    identity = get_jwt_identity()
    skills = request.json.get("skills", [])
    suggestions = []

    for skill in skills:
        suggestions.append({
            "skill": skill,
            "suggestion": "Freelancing / Online Consulting",
            "income_range": "$200 - $2000/month"
        })

    db.session.add(SkillLog(
        user_id=identity["id"],
        skills=", ".join(skills)
    ))
    db.session.commit()

    return jsonify({"recommendations": suggestions})

# =========================================================
# HEALTH ANALYSIS
# =========================================================

CONDITION_RULES = {
    "PCOS": ["irregular periods", "acne", "weight gain"],
    "Thyroid Disorder": ["fatigue", "hair loss"],
    "Anemia": ["low energy", "heavy bleeding"]
}

@app.route("/api/health", methods=["POST"])
@jwt_required()
def analyze_health():
    identity = get_jwt_identity()
    user = User.query.get(identity["id"])
    symptoms = request.json.get("symptoms", [])

    score_map = {}
    for condition, rule_symptoms in CONDITION_RULES.items():
        match = len(set(symptoms) & set(rule_symptoms))
        score_map[condition] = match

    best_condition = max(score_map, key=score_map.get)
    weighted_score = score_map[best_condition] / len(CONDITION_RULES[best_condition])
    risk_score = round(0.5 + weighted_score * 0.5, 2)

    if risk_score >= 0.80:
        send_health_alert(user.email, best_condition, risk_score)

    db.session.add(HealthLog(
        symptoms=", ".join(symptoms),
        prediction=best_condition,
        confidence=risk_score,
        risk_score=risk_score
    ))
    db.session.commit()

    return jsonify({
        "prediction": best_condition,
        "risk_score": risk_score
    })

# =========================================================
# PERIOD TRACKER
# =========================================================

@app.route("/api/period", methods=["POST"])
@jwt_required()
def period_tracker():
    identity = get_jwt_identity()
    last_period = request.json.get("last_period")
    avg_cycle = int(request.json.get("avg_cycle"))

    last_date = datetime.strptime(last_period, "%Y-%m-%d")
    next_period = last_date + timedelta(days=avg_cycle)
    ovulation_day = next_period - timedelta(days=14)

    db.session.add(PeriodLog(
        user_id=identity["id"],
        last_period=last_period,
        avg_cycle=avg_cycle,
        next_period=str(next_period.date()),
        ovulation_day=str(ovulation_day.date())
    ))
    db.session.commit()

    return jsonify({
        "next_period": str(next_period.date()),
        "ovulation_day": str(ovulation_day.date())
    })

# =========================================================
# ADMIN DASHBOARD
# =========================================================

@app.route("/api/admin/dashboard", methods=["GET"])
@jwt_required()
def admin_dashboard():
    identity = get_jwt_identity()
    user = User.query.get(identity["id"])

    if user.role != "admin":
        return jsonify({"error": "Admin access required"}), 403

    return jsonify({
        "total_users": User.query.count(),
        "blocked_users": User.query.filter_by(is_blocked=True).count(),
        "total_cyber_reports": CyberLog.query.count(),
        "cyberbullying_cases": CyberLog.query.filter_by(prediction="Cyberbullying").count(),
        "legal_queries": LegalLog.query.count(),
        "health_checks": HealthLog.query.count(),
        "high_risk_cases": HealthLog.query.filter(HealthLog.risk_score >= 0.75).count(),
        "period_logs": PeriodLog.query.count(),
        "skill_recommendations": SkillLog.query.count()
    })