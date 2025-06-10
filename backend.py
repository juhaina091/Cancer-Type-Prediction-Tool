from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from xgboost import XGBClassifier 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import pickle
import os
import matplotlib.pyplot as plt
import xgboost as xgb

app = Flask(__name__)

data, model, scaler, encoders = None, None, None, None

# Load dataset
def load_data():
    try:
        data = pd.read_csv(r'C:\Users\LENOVO\Desktop\g\merged_cancer_data.csv')
        if data.empty:
            print("Error: Data file is empty or corrupted.")
            return None
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

# Preprocess Data
def preprocess_data(data):
    data = data.fillna('Unknown')
    data['Cancer type'] = data['Cancer type'].astype(str)

    le_cancer, le_gene, le_status = LabelEncoder(), LabelEncoder(), LabelEncoder()
    
    data['Cancer_Type_Encoded'] = le_cancer.fit_transform(data['Cancer type'])
    data['Gene_Symbol_Encoded'] = le_gene.fit_transform(data['Gene symbol'].astype(str))
    
    data['Expression_Status'] = data['Differential expression status_up'].fillna('Unknown')
    data.loc[data['Differential expression status_down'] == 'Down', 'Expression_Status'] = 'Down'
    data.loc[data['Differential expression status_up'] == 'Up', 'Expression_Status'] = 'Up'
    data['Expression_Status_Encoded'] = le_status.fit_transform(data['Expression_Status'])

    encoders = {'cancer_type': le_cancer, 'gene_symbol': le_gene, 'expression_status': le_status}

    with open('encoders_2.pkl', 'wb') as f:
        pickle.dump(encoders, f)

    return data, encoders

# Model Building
# Feature Importance visualization with Matplotlib
def build_cancer_type_prediction_model(data, use_random_forest=False):
    features = ['Gene_Symbol_Encoded', 'Expression_Status_Encoded']
    numeric_features = [
        'The number of tumor types_pccup',
        'Number of tumor types (PCDGs)',
        'The number of tumor types_pccdp',
        'Number of tumor types (PCCDPs)',
        'Risk coefficient'
    ]

    for feature in numeric_features:
        if feature in data.columns:
            features.append(feature)

    X = data[features].apply(pd.to_numeric, errors='coerce').fillna(0)
    y = data['Cancer_Type_Encoded']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    if use_random_forest:
        model = RandomForestClassifier(n_estimators=300, random_state=42)
    else:
        model = XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=7, subsample=0.8, colsample_bytree=0.8)

    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    print(f"Model Accuracy: {accuracy}")

    # Save model and scaler
    with open('cancer_prediction_model_2.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('scaler_2.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    # Feature importance visualization
    if not use_random_forest:
        ax = xgb.plot_importance(model)
        plt.savefig(r'C:\Users\LENOVO\Desktop\miniproject\Cancer\webtool_6\static\feature_importance.png')

    return model, scaler, accuracy


# Initialization
@app.before_request
def initialize():
    global data, model, scaler, encoders
    if data is None or data.empty:
        data = load_data()
        if data is not None:
            data, encoders = preprocess_data(data)
            model, scaler, _ = build_cancer_type_prediction_model(data)
            print("Model initialized successfully!")
        else:
            print("Failed to initialize model - data loading error")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = request.get_json(force=True, silent=True)
        if not input_data:
            return jsonify({'error': 'Empty request body'}), 400

        gene_symbol = input_data.get('gene_symbol')
        expression_status = input_data.get('expression_status', 'Unknown')

        if not gene_symbol:
            return jsonify({'error': 'Gene symbol is required'}), 400

        with open('encoders_2.pkl', 'rb') as f:
            encoders = pickle.load(f)

        gene_encoded = encoders['gene_symbol'].transform([gene_symbol])[0]
        expression_encoded = encoders['expression_status'].transform([expression_status])[0] if expression_status else 0

        with open('cancer_prediction_model_2.pkl', 'rb') as f:
            model = pickle.load(f)

        with open('scaler_2.pkl', 'rb') as f:
            scaler = pickle.load(f)

        features = np.array([gene_encoded, expression_encoded, 0, 0, 0, 0, 0]).reshape(1, -1)
        features_df = pd.DataFrame(features, columns=scaler.feature_names_in_)
        features_scaled = scaler.transform(features_df)

        prediction = model.predict(features_scaled)[0]
        prediction_proba = model.predict_proba(features_scaled)[0]

        cancer_type = encoders['cancer_type'].inverse_transform([prediction])[0]
        probabilities = {ct: float(prob) for ct, prob in zip(encoders['cancer_type'].classes_, prediction_proba)}
        top_predictions = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)[:5]

        return jsonify({
            'predicted_cancer_type': cancer_type,
            'probability': float(max(prediction_proba)),
            'top_predictions': dict(top_predictions)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
