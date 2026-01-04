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
    # Initialize to avoid UnboundLocalError
    api_key = None
    try:
        st.secret = st.secrets["GROQ_API_KEY"]
    except Exception:
        pass
    # If Cloud failed, try local .env (Laptop)
    if not api_key:
        api_key = os.getenv("GROQ_API_KEY")

    # Final Check
    if not api_key:
        return "❌ Error: Missing GROQ_API_KEY. Set it in Secrets (Cloud) or .env (Local)."

    if not api_key:
        return "❌ Error: Missing GROQ_API_KEY in .env file."

    # Define the Prompt
    prompt = f"""You are an academic advisor. 
    A student has the following stats:
    - Predicted Final Grade: {prediction:.1f}/100
    - Study Hours: {hours}/10
    - Attendance: {attendance}%
    - Previous Score: {prev_score}/100

    Provide ONE sentence of specific, actionable advice to help them improve."""

    # Call Groq API (Standard OpenAI Format)
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
    # Handle missing columns if the user uploaded a messy CSV
    required_cols = ['hours_studied', 'attendance_rate', 'previous_score', 'tutoring_sessions', 'final_grade']
    if not all(col in new_df.columns for col in required_cols):
        return "❌ Error: CSV is missing columns!"

    # Define X and y from the NEW data
    X_new = new_df[['hours_studied', 'attendance_rate', 'previous_score', 'tutoring_sessions']]
    y_new = new_df['final_grade']
    
    # Train a NEW model on just the new data
    new_model = LinearRegression()
    new_model.fit(X_new, y_new)
    
    # Save it (Overwrite the old brain)
    joblib.dump(new_model, 'student_grade_predictor.pkl')
    return "✅ Success! The model has learned from the new data."

# --- UI ---
st.title("🎓 Student Performance Insight Lab")

tab1, tab2= st.tabs(["📊 Batch Analysis", "🔮 Grade Predictor"])

# === TAB 1: BATCH ANALYSIS & INSIGHTS ===
with tab1:
    st.header("📊 Batch Analysis & Data Insights")
    file = st.file_uploader("Upload Student CSV", type=["csv"])
    
    if file is not None:
        df = pd.read_csv(file)
        
        # --- FEATURE 1: DATA CLEANING ---
        st.subheader("Data Health & Cleaning")
        missing_count = df.isnull().sum().sum()
        
        if missing_count > 0:
            st.warning(f"⚠️ Found {missing_count} missing values.")
            st.write("Missing data by column:", df.isnull().sum()[df.isnull().sum() > 0])
            
            st.markdown("### 🛠️ Choose a Cleaning Strategy")
            
            clean_method = st.radio(
                "Select a method:",
                ["Mean (Average)", "Median (Middle)", "Mode (Most Frequent)", "Fill with 0 (Zero)", "Drop Rows"],
                horizontal=True
            )
            
            # 1. Identify columns to clean (Everything EXCEPT final_grade)
            cols_to_clean = [col for col in df.columns if col != 'final_grade']

            # 2. Apply Strategy
            
            # A. MEAN
            if clean_method == "Mean (Average)":
                st.write("ℹ️ **Good for:** Normal data. **Bad for:** Outliers.")
                fill_values = df[cols_to_clean].mean(numeric_only=True).round()
                df[cols_to_clean] = df[cols_to_clean].fillna(fill_values).astype(int)
                
            # B. MEDIAN
            elif clean_method == "Median (Middle)":
                st.write("ℹ️ **Good for:** Skewed data. **Bad for:** Precise totals.")
                df[cols_to_clean] = df[cols_to_clean].fillna(df[cols_to_clean].median(numeric_only=True))
            
            # C. MODE
            elif clean_method == "Mode (Most Frequent)":
                st.write("ℹ️ **Good for:** Categories. **Bad for:** Continuous numbers.")
                for col in cols_to_clean:
                    if df[col].isnull().sum() > 0:
                        df[col] = df[col].fillna(df[col].mode()[0])
            
            # D. FILL WITH 0 
            elif clean_method == "Fill with 0 (Zero)":
                st.write("ℹ️ **Pros:** Safe assumption. **Cons:** Lowers averages.")
                df[cols_to_clean] = df[cols_to_clean].fillna(0)
                
            # E. DROP ROWS
            elif clean_method == "Drop Rows":
                st.error("⚠️ **Warning:** Removes data.")
                df = df.dropna(subset=cols_to_clean)

            st.success(f"✅ Applied Strategy: {clean_method}")
            st.write("**Note:** Data cleaning is applied to input features only. The **Final Grade** is left untouched to ensure accurate predictions.")
        else:
            st.success("✅ Data is strictly clean! No missing values detected.")

        st.divider()
        # --- FEATURE 2: DATA OVERVIEW ---
        st.write(df.tail(5))

        st.divider()  
        # --- FEATURE 2: DISTRIBUTIONS ---
        if 'final_grade' in df.columns:
            st.subheader("Grade Distribution")
            st.write("How are the grades spread across the class?")
            
            # Simple Streamlit Bar Chart
            # We count how many students got each grade (rounded to nearest 10 for grouping)
            grade_counts = df['final_grade'].value_counts().sort_index()
            st.bar_chart(grade_counts)
        else:
            st.info("ℹ️ Upload a file with 'final_grade' to see the distribution.")

        st.divider()

        # --- FEATURE 3: CORRELATIONS ---
        # We can only show correlations if the file actually has 'final_grade'
        if 'final_grade' in df.columns:
            st.subheader("Correlation Analysis")
            st.write("Which factors actually affect the final grade?")
            
            # Select only numbers (ignore Names/IDs if they exist)
            numeric_df = df.select_dtypes(include=['float64', 'int64'])
            
            # Calculate Correlation Matrix
            corr_matrix = numeric_df.corr()
            
            # DISPLAY HEATMAP (Using Pandas Styling - No heavy Seaborn needed!)
            st.dataframe(corr_matrix.style.background_gradient(cmap="coolwarm"), width="stretch")
            
            # INTELLIGENT INSIGHT
            # Find the factor with the highest correlation to 'final_grade' (ignoring the grade itself)
            top_factor = corr_matrix['final_grade'].drop('final_grade').idxmax()
            top_value = corr_matrix['final_grade'][top_factor]
            
            st.info(f"💡 **Discovery:** The strongest predictor of success in this dataset is **'{top_factor}'** (Correlation: {top_value:.2f}).")
            
        else:
            st.info("ℹ️ Upload a file with 'final_grade' to see correlations.")

        st.divider()

        # --- FEATURE 4: PREDICTIONS ---
        st.subheader("Predictive Modeling")
        if 'final_grade' not in df.columns:
            st.warning("⚠️ New Data Detected (Grades Missing). Running AI Predictions...")
            try:
                X_new = df[['hours_studied', 'attendance_rate', 'previous_score', 'tutoring_sessions']]
                df['predicted_grade'] = model.predict(X_new).clip(0, 100).round(1)
                st.dataframe(df)
                
                # Download
                csv = df.to_csv(index=False).encode('utf-8')
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                st.download_button("⬇️ Download Predictions", csv, f"predictions_{timestamp}.csv", "text/csv")
            except Exception as e:
                st.error(f"❌ Error: CSV must have columns: hours_studied, attendance_rate, previous_score, tutoring_sessions")
        else:
            # If grades exist, just show the stats
            col1, col2, col3 = st.columns(3)
            col1.metric("Class Average", f"{df['final_grade'].mean():.1f}%")
            col2.metric("Highest Score", f"{df['final_grade'].max():.1f}%")
            col3.metric("Lowest Score", f"{df['final_grade'].min():.1f}%")

    else:
        st.info("👆 Please upload a CSV file to begin analysis.")

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
