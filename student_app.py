from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import streamlit as st
import pandas as pd
import joblib
import os
from datetime import datetime
import requests
from dotenv import load_dotenv

# --- CONFIG ---
st.set_page_config(page_title="Student Insights", layout="wide", page_icon="🎓", initial_sidebar_state="collapsed")

st.sidebar.empty()

# --- LOAD BRAIN ---
@st.cache_resource
def load_model():
    return joblib.load('student_grade_predictor.pkl')

model = load_model()

load_dotenv() # Load your HF Key

def get_ai_feedback(prediction, hours, attendance, prev_score):
    """Generates a human-like explanation using Groq (Llama 3)."""
    api_key = None
    try:
        st.secret = st.secrets["GROQ_API_KEY"]
    except:
        api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        return "❌ Error: Missing GROQ_API_KEY in .env file."

    # 2. Define the Prompt
    prompt = f"""You are an academic advisor. 
    A student has the following stats:
    - Predicted Final Grade: {prediction:.1f}/100
    - Study Hours: {hours}/10
    - Attendance: {attendance}%
    - Previous Score: {prev_score}/100

    Provide ONE sentence of specific, actionable advice to help them improve."""

    # 3. Call Groq API (Standard OpenAI Format)
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "llama-3.3-70b-versatile", # Using Llama 3
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=5)
        
        if response.status_code != 200:
            return f"⚠️ API Error: {response.status_code} - {response.text}"
            
        result = response.json()
        return result['choices'][0]['message']['content'].strip()
        
    except Exception as e:
        return f"⚠️ Connection Error: {str(e)}"

def retrain_model(new_df):
    """
    Takes new data, combines it with old knowledge, 
    and saves a smarter 'Edition 2' model.
    """
    # 1. Handle missing columns if the user uploaded a messy CSV
    required_cols = ['hours_studied', 'attendance_rate', 'previous_score', 'tutoring_sessions', 'final_grade']
    if not all(col in new_df.columns for col in required_cols):
        return "❌ Error: CSV is missing columns!"

    # 2. Define X and y from the NEW data
    X_new = new_df[['hours_studied', 'attendance_rate', 'previous_score', 'tutoring_sessions']]
    y_new = new_df['final_grade']
    
    # 3. Train a NEW model on just the new data
    new_model = LinearRegression()
    new_model.fit(X_new, y_new)
    
    # 4. Save it (Overwrite the old brain)
    joblib.dump(new_model, 'student_grade_predictor.pkl')
    return "✅ Success! The model has learned from the new data."

# --- UI ---
st.title("🎓 Student Performance Insight Lab")

tab1, tab2= st.tabs(["📊 Batch Analysis", "🔮 Grade Predictor"])

# === TAB 1: DASHBOARD ===
with tab1:
    file = st.file_uploader("Upload Student CSV", type=["csv"])
    
    # 1. OUTER CHECK: Did the user upload anything?
    if file is not None:
        df = pd.read_csv(file)
        # 2. INNER CHECK:
        
        # (Dashboard Mode) when grades are present
        if 'final_grade' in df.columns:
            st.success("✅ Historical Data Detected (Grades Included)")
            st.dataframe(df.head())
            
            # Some basic stats
            st.header("📈 Performance Overview")
            col1, col2, col3 = st.columns(3)
            col1.metric("Avg Score", f"{df['final_grade'].mean():.1f}")
            col2.metric("Max Score", f"{df['final_grade'].max():.1f}")
            col3.metric("Min Score", f"{df['final_grade'].min():.1f}")
            st.bar_chart(df['final_grade'])

            st.header("🏆 Top Performing Students")
            top_students = df.sort_values(by='final_grade', ascending=False).head(5)
            st.table(top_students[['hours_studied', 'attendance_rate', 'final_grade']])

        # (Prediction Mode) when grades are missing
        else:
            st.warning("⚠️ New Data Detected (Grades Missing). Running AI Predictions...")
            try:
                # Ensure we have the input columns needed for the model
                X_new = df[['hours_studied', 'attendance_rate', 'previous_score', 'tutoring_sessions']]
                
                # Predict!
                df['predicted_grade'] = model.predict(X_new)
                # Ensure grades are within 0-100
                df['predicted_grade'] = df['predicted_grade'].clip(0, 100)
                
                df['predicted_grade'] = df['predicted_grade'].round(1) # Round to 1 decimal place
                
                st.success("🎉 Predictions Generated!")
                st.dataframe(df)

                # Download Button
                csv = df.to_csv(index=False).encode('utf-8')
                timestamp = datetime.now().strftime("%Y%m%d-%H-%M-%S")
                st.download_button("⬇️ Download Predictions", csv, f"predicted_grades_{timestamp}.csv", "text/csv") 

                st.bar_chart(df['predicted_grade'])

                st.header("🏆 Top Performing Students")
                top_students = df.sort_values(by='predicted_grade', ascending=False).head(5)
                st.table(top_students[['hours_studied', 'attendance_rate', 'predicted_grade']])  
            except Exception as e:
                st.error(f"❌ Error: Your CSV is missing input columns. Details: {e}")

    # 3. OUTER ELSE: No file uploaded yet
    else:
        st.info("👆 Please upload a CSV file to begin!")

# === TAB 2: PREDICTION ENGINE ===
with tab2:
    st.header("Predict a Student's Final Grade")

    col1,col2,col3,col4 = st.columns(4)
    # 1. USER INPUTS
    with col1:
        hours = st.number_input("Hours Studied", 1, 10, 5)
    with col2:
        attendance = st.number_input("Attendance Rate (%)", 50, 100, 80)
    with col3:
        prev_score = st.number_input("Previous Score", 40, 100, 70)
    with col4:
        tutoring = st.number_input("Tutoring Sessions", 0, 10, 2)

    
    # 2. PREDICT BUTTON
    if st.button("Calculate Grade"):
        # Prepare input data
        input_data = pd.DataFrame([[hours, attendance, prev_score, tutoring]], 
                                  columns=['hours_studied', 'attendance_rate', 'previous_score', 'tutoring_sessions'])
        
        raw_prediction = model.predict(input_data)[0]

        # Clip the prediction to be between 0 and 100
        prediction = max(0, min(100, raw_prediction))

        st.success(f"Predicted Final Grade: {prediction:.1f} / 100")
        
        # Simple Interpretation
        if prediction >= 90:
            st.balloons()
            st.write("🌟 Top of the class!")
        elif prediction < 60:
            st.error("⚠️ At Risk! Suggest immediate intervention.")

        # AI-Generated Feedback
        with st.spinner("Generating feedback..."):
            feedback = get_ai_feedback(prediction, hours, attendance, prev_score)
            st.info(f"📝 **Advisor's Note:** {feedback}")

# === ADMIN ZONE ===
with st.sidebar:
    st.header("⚙️ Admin Dashboard")
    
    # --- FEATURE 1: HEALTH CHECK ---
    with st.expander("🩺 Model Health Check"):
        st.write("Is the model still accurate?")
        check_file = st.file_uploader("Upload 'Actual vs Predicted' CSV", type=["csv"], key="health_check")
        
        if check_file:
            check_df = pd.read_csv(check_file)
            
            # INNER CHECK: Does it have 'final_grade'?
            if 'final_grade' in check_df.columns:
                X_check = check_df[['hours_studied', 'attendance_rate', 'previous_score', 'tutoring_sessions']]
                check_df['predicted_grade'] = model.predict(X_check)
                
                # Calculate Error (MAE)
                from sklearn.metrics import mean_absolute_error
                mae = mean_absolute_error(check_df['final_grade'], check_df['predicted_grade'])
                
                st.metric("Current Error (MAE)", f"{mae:.2f} pts")
                
                if mae < 5:
                    st.success("✅ Healthy")
                else:
                    st.error("⚠️ Drift Detected! Retrain recommended.")
            else:
                st.warning("CSV needs 'final_grade' column.")

    st.divider()

    # --- FEATURE 2: RETRAINER ---
    with st.expander("🚀 Retrain Model"):
        st.write("Upload fresh data to update the AI.")
        training_file = st.file_uploader("Upload New Training Data", key="train")
        
        if training_file:
            new_data = pd.read_csv(training_file)
            if st.button("Start Retraining"):
                with st.spinner("Training new brain..."):
                    message = retrain_model(new_data)
                    if "Success" in message:
                        st.success("Done! Reload the page to use the new model.")
                        st.cache_resource.clear() 
                    else:
                        st.error(message)
