import os, pickle, sqlite3, uuid, csv, io, json, functools
from datetime import datetime
from flask import Flask, render_template, request, jsonify, session, redirect, url_for, send_file, flash
from werkzeug.security import generate_password_hash, check_password_hash
import pandas as pd
import numpy as np

app = Flask(__name__)
app.secret_key = 'churnguard-secret-key-2024'

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, 'churn.db')

# === Load Model ===
model = pickle.load(open(os.path.join(BASE_DIR, 'model_new.pkl'), 'rb'))
scaler = pickle.load(open(os.path.join(BASE_DIR, 'scaler_new.pkl'), 'rb'))

ENCODING_MAP = {
    'gender': {'Male': 1, 'Female': 0},
    'Partner': {'Yes': 1, 'No': 0},
    'Dependents': {'Yes': 1, 'No': 0},
    'PhoneService': {'Yes': 1, 'No': 0},
    'MultipleLines': {'No phone service': 0, 'No': 1, 'Yes': 2},
    'InternetService': {'No': 0, 'DSL': 1, 'Fiber optic': 2},
    'OnlineSecurity': {'No internet service': 0, 'No': 1, 'Yes': 2},
    'OnlineBackup': {'No internet service': 0, 'No': 1, 'Yes': 2},
    'DeviceProtection': {'No internet service': 0, 'No': 1, 'Yes': 2},
    'TechSupport': {'No internet service': 0, 'No': 1, 'Yes': 2},
    'StreamingTV': {'No internet service': 0, 'No': 1, 'Yes': 2},
    'StreamingMovies': {'No internet service': 0, 'No': 1, 'Yes': 2},
    'Contract': {'Month-to-month': 0, 'One year': 1, 'Two year': 2},
    'PaperlessBilling': {'Yes': 1, 'No': 0},
    'PaymentMethod': {'Bank transfer (automatic)': 0, 'Credit card (automatic)': 1, 'Electronic check': 2, 'Mailed check': 3}
}
FEATURE_COLUMNS = ['gender','SeniorCitizen','Partner','Dependents','tenure','PhoneService','MultipleLines','InternetService','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies','Contract','PaperlessBilling','PaymentMethod','MonthlyCharges','TotalCharges']
NUMERICAL_COLUMNS = ['tenure', 'MonthlyCharges', 'TotalCharges']
FEATURE_IMPORTANCES = dict(zip(FEATURE_COLUMNS, model.feature_importances_))

# === Auth Decorator ===
def login_required(f):
    @functools.wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated

# === Database ===
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    conn.execute('''CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        email TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL,
        created_at TEXT)''')
    conn.execute('''CREATE TABLE IF NOT EXISTS customers (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        customer_id TEXT, gender TEXT,
        senior_citizen INTEGER, partner TEXT, dependents TEXT, tenure INTEGER,
        phone_service TEXT, multiple_lines TEXT, internet_service TEXT,
        online_security TEXT, online_backup TEXT, device_protection TEXT,
        tech_support TEXT, streaming_tv TEXT, streaming_movies TEXT,
        contract TEXT, paperless_billing TEXT, payment_method TEXT,
        monthly_charges REAL, total_charges REAL, churn_prediction TEXT,
        churn_probability REAL, risk_category TEXT, created_at TEXT,
        FOREIGN KEY (user_id) REFERENCES users(id))''')
    conn.commit()
    conn.close()

# === Prediction ===
def predict_from_dict(data):
    encoded = []
    for col in FEATURE_COLUMNS:
        if col in ENCODING_MAP:
            val = ENCODING_MAP[col].get(str(data.get(col, '')), 0)
        elif col == 'SeniorCitizen':
            val = int(data.get('SeniorCitizen', 0))
        else:
            val = float(data.get(col, 0))
        encoded.append(val)
    arr = np.array([encoded])
    num_idx = [FEATURE_COLUMNS.index(c) for c in NUMERICAL_COLUMNS]
    arr_scaled = arr.copy().astype(float)
    arr_scaled[:, num_idx] = scaler.transform(arr[:, num_idx])
    prob = model.predict_proba(arr_scaled)[0][1]
    pred = 'Yes' if prob >= 0.5 else 'No'
    risk = 'High' if prob >= 0.7 else ('Medium' if prob >= 0.4 else 'Low')
    return pred, round(prob * 100, 2), risk

# === Auth Routes ===
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '')
        if not username or not email or not password:
            return render_template('login.html', error='All fields are required', mode='signup')
        if len(password) < 4:
            return render_template('login.html', error='Password must be at least 4 characters', mode='signup')
        conn = get_db()
        existing = conn.execute('SELECT id FROM users WHERE email=? OR username=?', (email, username)).fetchone()
        if existing:
            conn.close()
            return render_template('login.html', error='Username or email already exists', mode='signup')
        hashed = generate_password_hash(password)
        conn.execute('INSERT INTO users (username, email, password, created_at) VALUES (?,?,?,?)',
            (username, email, hashed, datetime.now().isoformat()))
        # Delete ALL existing customer data for fresh start
        conn.execute('DELETE FROM customers')
        conn.commit()
        user = conn.execute('SELECT id, username FROM users WHERE email=?', (email,)).fetchone()
        conn.close()
        session['user_id'] = user['id']
        session['username'] = user['username']
        return redirect('/')
    return render_template('login.html', mode='signup')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '')
        conn = get_db()
        user = conn.execute('SELECT * FROM users WHERE email=?', (email,)).fetchone()
        conn.close()
        if user and check_password_hash(user['password'], password):
            session['user_id'] = user['id']
            session['username'] = user['username']
            return redirect('/')
        return render_template('login.html', error='Invalid email or password', mode='login')
    return render_template('login.html', mode='login')

@app.route('/logout')
def logout():
    session.clear()
    return redirect('/login')

# === Page Routes ===
@app.route('/')
@login_required
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    data = {
        'gender': request.form.get('gender','Male'),
        'SeniorCitizen': int(request.form.get('SeniorCitizen', 0)),
        'Partner': request.form.get('Partner','No'),
        'Dependents': request.form.get('Dependents','No'),
        'tenure': float(request.form.get('tenure', 0)),
        'PhoneService': request.form.get('PhoneService','No'),
        'MultipleLines': request.form.get('MultipleLines','No'),
        'InternetService': request.form.get('InternetService','No'),
        'OnlineSecurity': request.form.get('OnlineSecurity','No'),
        'OnlineBackup': request.form.get('OnlineBackup','No'),
        'DeviceProtection': request.form.get('DeviceProtection','No'),
        'TechSupport': request.form.get('TechSupport','No'),
        'StreamingTV': request.form.get('StreamingTV','No'),
        'StreamingMovies': request.form.get('StreamingMovies','No'),
        'Contract': request.form.get('Contract','Month-to-month'),
        'PaperlessBilling': request.form.get('PaperlessBilling','No'),
        'PaymentMethod': request.form.get('PaymentMethod','Electronic check'),
        'MonthlyCharges': float(request.form.get('MonthlyCharges', 0)),
        'TotalCharges': float(request.form.get('TotalCharges', 0)),
    }
    pred, prob, risk = predict_from_dict(data)
    cid = f"CG-{uuid.uuid4().hex[:8].upper()}"
    uid = session['user_id']
    conn = get_db()
    conn.execute('''INSERT INTO customers (user_id,customer_id,gender,senior_citizen,partner,dependents,tenure,phone_service,multiple_lines,internet_service,online_security,online_backup,device_protection,tech_support,streaming_tv,streaming_movies,contract,paperless_billing,payment_method,monthly_charges,total_charges,churn_prediction,churn_probability,risk_category,created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
    (uid,cid,data['gender'],data['SeniorCitizen'],data['Partner'],data['Dependents'],data['tenure'],data['PhoneService'],data['MultipleLines'],data['InternetService'],data['OnlineSecurity'],data['OnlineBackup'],data['DeviceProtection'],data['TechSupport'],data['StreamingTV'],data['StreamingMovies'],data['Contract'],data['PaperlessBilling'],data['PaymentMethod'],data['MonthlyCharges'],data['TotalCharges'],pred,prob,risk,datetime.now().isoformat()))
    conn.commit(); conn.close()
    top_features = sorted(FEATURE_IMPORTANCES.items(), key=lambda x: x[1], reverse=True)[:8]
    session['last_prediction'] = {'pred': pred, 'prob': prob, 'risk': risk, 'data': data}
    recs = get_recommendations(risk, data)
    return render_template('result.html', pred=pred, prob=prob, risk=risk, data=data, cid=cid, features=top_features, recommendations=recs)

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html')

@app.route('/customers')
@login_required
def customers():
    return render_template('customers.html')

@app.route('/customer/<int:cid>')
@login_required
def customer_profile(cid):
    uid = session['user_id']
    conn = get_db()
    customer = conn.execute('SELECT * FROM customers WHERE id=? AND user_id=?', (cid, uid)).fetchone()
    conn.close()
    if not customer:
        return redirect('/customers')
    c = dict(customer)
    # Generate risk factors
    risk_factors = []
    if c.get('contract') == 'Month-to-month': risk_factors.append({'factor': 'Month-to-month contract', 'impact': 'High', 'detail': 'Highest churn risk - no commitment'})
    if (c.get('tenure') or 0) < 12: risk_factors.append({'factor': f"Short tenure ({c.get('tenure',0)} months)", 'impact': 'High', 'detail': 'New customers churn more frequently'})
    if (c.get('monthly_charges') or 0) > 70: risk_factors.append({'factor': f"High monthly charges (${c.get('monthly_charges',0)})", 'impact': 'Medium', 'detail': 'Price sensitivity increases churn'})
    if c.get('internet_service') == 'Fiber optic': risk_factors.append({'factor': 'Fiber optic internet', 'impact': 'Medium', 'detail': 'Higher expectations, more alternatives'})
    if c.get('tech_support') in ['No', None]: risk_factors.append({'factor': 'No tech support', 'impact': 'Medium', 'detail': 'Unresolved issues drive churn'})
    if c.get('online_security') in ['No', None]: risk_factors.append({'factor': 'No online security', 'impact': 'Low', 'detail': 'Missing value-add service'})
    if c.get('payment_method') == 'Electronic check': risk_factors.append({'factor': 'Electronic check payment', 'impact': 'Low', 'detail': 'Less committed payment method'})
    if c.get('paperless_billing') == 'Yes': risk_factors.append({'factor': 'Paperless billing', 'impact': 'Low', 'detail': 'Slightly correlated with churn'})
    recs = get_recommendations(c.get('risk_category','Low'), {
        'Contract': c.get('contract'), 'TechSupport': c.get('tech_support'),
        'OnlineSecurity': c.get('online_security'), 'PaymentMethod': c.get('payment_method')
    })
    top_features = sorted(FEATURE_IMPORTANCES.items(), key=lambda x: x[1], reverse=True)[:8]
    return render_template('profile.html', c=c, risk_factors=risk_factors, recommendations=recs, features=top_features)

@app.route('/alerts')
@login_required
def alerts():
    return render_template('alerts.html')

@app.route('/api/alerts')
@login_required
def api_alerts():
    uid = session['user_id']
    conn = get_db()
    high = conn.execute("SELECT * FROM customers WHERE user_id=? AND risk_category='High' ORDER BY churn_probability DESC", (uid,)).fetchall()
    medium = conn.execute("SELECT * FROM customers WHERE user_id=? AND risk_category='Medium' ORDER BY churn_probability DESC LIMIT 20", (uid,)).fetchall()
    total = conn.execute('SELECT COUNT(*) FROM customers WHERE user_id=?', (uid,)).fetchone()[0]
    high_count = len(high)
    medium_count = conn.execute("SELECT COUNT(*) FROM customers WHERE user_id=? AND risk_category='Medium'", (uid,)).fetchone()[0]
    churned = conn.execute("SELECT COUNT(*) FROM customers WHERE user_id=? AND churn_prediction='Yes'", (uid,)).fetchone()[0]
    conn.close()
    return jsonify({
        'high_risk': [dict(r) for r in high],
        'medium_risk': [dict(r) for r in medium],
        'stats': {'total': total, 'high_count': high_count, 'medium_count': medium_count, 'churned': churned}
    })

# === API Routes (all filtered by user_id) ===
@app.route('/api/dashboard-data')
@login_required
def dashboard_data():
    uid = session['user_id']
    conn = get_db()
    total = conn.execute('SELECT COUNT(*) FROM customers WHERE user_id=?', (uid,)).fetchone()[0]
    if total == 0:
        conn.close()
        return jsonify({'total':0,'churned':0,'churn_rate':0,'avg_tenure':0,'avg_monthly':0,'high_risk':0,'by_contract':[],'by_internet':[],'by_risk':[]})
    churned = conn.execute("SELECT COUNT(*) FROM customers WHERE user_id=? AND churn_prediction='Yes'", (uid,)).fetchone()[0]
    avg_tenure = conn.execute('SELECT AVG(tenure) FROM customers WHERE user_id=?', (uid,)).fetchone()[0] or 0
    avg_monthly = conn.execute('SELECT AVG(monthly_charges) FROM customers WHERE user_id=?', (uid,)).fetchone()[0] or 0
    by_contract = conn.execute("SELECT contract, churn_prediction, COUNT(*) as cnt FROM customers WHERE user_id=? GROUP BY contract, churn_prediction", (uid,)).fetchall()
    by_internet = conn.execute("SELECT internet_service, churn_prediction, COUNT(*) as cnt FROM customers WHERE user_id=? GROUP BY internet_service, churn_prediction", (uid,)).fetchall()
    by_risk = conn.execute("SELECT risk_category, COUNT(*) as cnt FROM customers WHERE user_id=? GROUP BY risk_category", (uid,)).fetchall()
    high_risk = conn.execute("SELECT COUNT(*) FROM customers WHERE user_id=? AND risk_category='High'", (uid,)).fetchone()[0]
    conn.close()
    return jsonify({
        'total': total, 'churned': churned, 'churn_rate': round(churned/total*100,1) if total else 0,
        'avg_tenure': round(avg_tenure,1), 'avg_monthly': round(avg_monthly,2), 'high_risk': high_risk,
        'by_contract': [dict(r) for r in by_contract], 'by_internet': [dict(r) for r in by_internet],
        'by_risk': [dict(r) for r in by_risk]
    })

@app.route('/api/customers')
@login_required
def api_customers():
    uid = session['user_id']
    conn = get_db()
    page = int(request.args.get('page', 1)); per_page = 15
    search = request.args.get('search', '')
    risk_filter = request.args.get('risk', '')
    q = 'SELECT * FROM customers WHERE user_id=?'
    params = [uid]
    if search:
        q += ' AND (customer_id LIKE ? OR gender LIKE ?)'; params += [f'%{search}%', f'%{search}%']
    if risk_filter:
        q += ' AND risk_category=?'; params.append(risk_filter)
    total = conn.execute(q.replace('SELECT *', 'SELECT COUNT(*)'), params).fetchone()[0]
    q += f' ORDER BY id DESC LIMIT {per_page} OFFSET {(page-1)*per_page}'
    rows = conn.execute(q, params).fetchall()
    conn.close()
    return jsonify({'customers': [dict(r) for r in rows], 'total': total, 'pages': max(1,(total + per_page - 1) // per_page), 'page': page})

@app.route('/api/customers/<int:cid>', methods=['DELETE'])
@login_required
def delete_customer(cid):
    conn = get_db(); conn.execute('DELETE FROM customers WHERE id=? AND user_id=?', (cid, session['user_id'])); conn.commit(); conn.close()
    return jsonify({'success': True})

@app.route('/api/customers/delete-all', methods=['DELETE'])
@login_required
def delete_all_customers():
    conn = get_db(); conn.execute('DELETE FROM customers WHERE user_id=?', (session['user_id'],)); conn.commit(); conn.close()
    return jsonify({'success': True})

@app.route('/api/customers/<int:cid>', methods=['PUT'])
@login_required
def update_customer(cid):
    d = request.json; uid = session['user_id']
    conn = get_db()
    conn.execute('UPDATE customers SET gender=?,senior_citizen=?,partner=?,dependents=?,tenure=?,contract=?,monthly_charges=?,total_charges=? WHERE id=? AND user_id=?',
        (d.get('gender'),d.get('senior_citizen'),d.get('partner'),d.get('dependents'),d.get('tenure'),d.get('contract'),d.get('monthly_charges'),d.get('total_charges'),cid,uid))
    conn.commit(); conn.close()
    return jsonify({'success': True})

@app.route('/upload-csv', methods=['POST'])
@login_required
def upload_csv():
    f = request.files.get('file')
    if not f: return jsonify({'error': 'No file'}), 400
    uid = session['user_id']
    df = pd.read_csv(f)
    df['TotalCharges'] = pd.to_numeric(df.get('TotalCharges', 0), errors='coerce').fillna(0)
    results = []; conn = get_db()
    for _, row in df.iterrows():
        d = row.to_dict()
        pred, prob, risk = predict_from_dict(d)
        cid = d.get('customerID', f"CSV-{uuid.uuid4().hex[:6].upper()}")
        conn.execute('''INSERT INTO customers (user_id,customer_id,gender,senior_citizen,partner,dependents,tenure,phone_service,multiple_lines,internet_service,online_security,online_backup,device_protection,tech_support,streaming_tv,streaming_movies,contract,paperless_billing,payment_method,monthly_charges,total_charges,churn_prediction,churn_probability,risk_category,created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
        (uid,cid,d.get('gender'),d.get('SeniorCitizen',0),d.get('Partner'),d.get('Dependents'),d.get('tenure',0),d.get('PhoneService'),d.get('MultipleLines'),d.get('InternetService'),d.get('OnlineSecurity'),d.get('OnlineBackup'),d.get('DeviceProtection'),d.get('TechSupport'),d.get('StreamingTV'),d.get('StreamingMovies'),d.get('Contract'),d.get('PaperlessBilling'),d.get('PaymentMethod'),d.get('MonthlyCharges',0),d.get('TotalCharges',0),pred,prob,risk,datetime.now().isoformat()))
        results.append({'id': cid, 'prediction': pred, 'probability': prob, 'risk': risk})
    conn.commit(); conn.close()
    return jsonify({'results': results, 'count': len(results)})

@app.route('/export-csv')
@login_required
def export_csv():
    uid = session['user_id']
    conn = get_db(); rows = conn.execute('SELECT * FROM customers WHERE user_id=? ORDER BY id DESC', (uid,)).fetchall(); conn.close()
    output = io.StringIO()
    w = csv.writer(output)
    if rows:
        w.writerow([k for k in rows[0].keys()])
        for r in rows: w.writerow([r[k] for k in r.keys()])
    else:
        w.writerow(['No data'])
    output.seek(0)
    return send_file(io.BytesIO(output.getvalue().encode()), mimetype='text/csv', as_attachment=True, download_name='customers_export.csv')

# === Chatbot ===
@app.route('/chat', methods=['POST'])
@login_required
def chat():
    msg = request.json.get('message', '').lower().strip()
    ctx = session.get('last_prediction')
    reply = chatbot_response(msg, ctx)
    return jsonify({'reply': reply})

def chatbot_response(msg, ctx):
    if any(w in msg for w in ['hello','hi','hey','greetings']):
        return "Hello! I'm ChurnGuard AI Assistant. I can help with:\n- Understanding churn predictions\n- Risk analysis & retention strategies\n- Platform navigation\n\nWhat would you like to know?"
    if 'what is churn' in msg or 'define churn' in msg:
        return "Customer churn is when customers stop using a service. In telecom, it means switching providers or canceling. High churn directly impacts revenue and growth."
    if ('why' in msg and ('risk' in msg or 'churn' in msg)) or 'explain' in msg:
        if ctx:
            d = ctx['data']; reasons = []
            if d.get('Contract') == 'Month-to-month': reasons.append("Month-to-month contract (highest churn risk)")
            if float(d.get('tenure',0)) < 12: reasons.append(f"Short tenure ({d.get('tenure')} months)")
            if float(d.get('MonthlyCharges',0)) > 70: reasons.append(f"High monthly charges (${d.get('MonthlyCharges')})")
            if d.get('InternetService') == 'Fiber optic': reasons.append("Fiber optic service (higher churn rate)")
            if d.get('TechSupport') in ['No','No internet service']: reasons.append("No tech support")
            if d.get('OnlineSecurity') in ['No','No internet service']: reasons.append("No online security")
            if d.get('PaymentMethod') == 'Electronic check': reasons.append("Electronic check payment (less committed)")
            return f"Risk: {ctx['risk']} ({ctx['prob']}%)\n\nKey factors:\n" + "\n".join(f"- {r}" for r in (reasons or ["No major risk factors identified"]))
        return "Please make a prediction first, then I can explain the risk factors."
    if any(w in msg for w in ['reduce','prevent','retention','strategy','retain']):
        return "Top retention strategies:\n1. Offer long-term contract discounts\n2. Bundle services at lower rates\n3. Provide dedicated support for at-risk customers\n4. Loyalty rewards programs\n5. Proactive outreach before contract renewal\n6. Address service quality issues quickly"
    if 'discount' in msg or 'offer' in msg:
        return "Recommended discount strategies:\n- 15-20% off for 1-year commitment\n- Free premium add-on for 6 months\n- Bundle discount (Internet + TV + Phone)\n- Referral bonuses\n- Loyalty milestone rewards"
    if any(w in msg for w in ['feature','important','factor']):
        return "Top churn prediction factors:\n1. Contract type (most important)\n2. Total charges & tenure\n3. Monthly charges\n4. Internet service type\n5. Tech support & online security\n6. Payment method"
    if any(w in msg for w in ['help','can you','what can']):
        return "I can help with:\n- 'What is churn?' - Learn about churn\n- 'Why is this customer high risk?' - Analyze predictions\n- 'How to reduce churn?' - Retention strategies\n- 'Suggest a discount' - Pricing strategies\n- 'Important features' - Key prediction factors"
    if any(w in msg for w in ['dashboard','chart','analytics']):
        return "The Dashboard shows:\n- KPI cards (total customers, churn rate, etc.)\n- Churn distribution charts\n- Analysis by contract type & internet service\n- Risk category breakdown\n\nVisit the Dashboard page for full analytics!"
    if any(w in msg for w in ['bye','thanks','thank']):
        return "You're welcome! Feel free to ask anytime. Happy analyzing!"
    return "I'm not sure about that. Try asking:\n- 'What is churn?'\n- 'Why is this customer high risk?'\n- 'How to reduce churn?'\n- Type 'help' for all options."

def get_recommendations(risk, data):
    recs = []
    if risk == 'High':
        recs = ["Immediate outreach - contact within 24 hours", "Offer 20% discount on next 3 months", "Assign dedicated account manager", "Free service upgrade for loyalty retention", "Schedule a satisfaction survey call"]
    elif risk == 'Medium':
        recs = ["Send personalized retention email", "Offer contract upgrade with 10% discount", "Provide value-added services trial", "Monitor usage patterns closely"]
    else:
        recs = ["Continue excellent service delivery", "Invite to loyalty rewards program", "Request referral with bonus incentive", "Send appreciation communication"]
    if data.get('Contract') == 'Month-to-month': recs.append("Suggest switching to annual contract with discount")
    if data.get('TechSupport') in ['No','No internet service']: recs.append("Recommend adding Tech Support package")
    return recs

init_db()

if __name__ == '__main__':
    app.run(debug=True, port=5000)
