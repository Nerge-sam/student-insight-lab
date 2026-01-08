# 🎓 Student Performance Insight Lab

A Machine Learning powered dashboard that helps educators analyze student data, predict performance, and receive actionable, AI-driven advice to improve grades.

🔗 **[Live Demo App](https://student-insight-lab-xyz.streamlit.app)**

---

## 🚀 Key Features

### 1. 📊 Advanced Batch Analysis
Upload a CSV of student data to instantly generate a comprehensive report:
* **Auto-Cleaning:** Automatically detects missing values and fills them using mean imputation.
* **Grade Distribution:** Visualizes how grades are spread across the class (bar charts).
* **Correlation Heatmap:** Identifies which factors (e.g., Study Hours, Attendance) have the strongest impact on final grades.

### 2. 🔮 Grade Predictor
Uses a **Linear Regression** model to predict a specific student's final grade based on their study habits and previous scores.

### 3. 🤖 AI Advisor
Integrated **Llama 3 (via Groq API)** to act as a virtual academic advisor. It analyzes the prediction and provides specific, human-like strategies for the student to improve.

### 4. ⚙️ Admin Dashboard
* **Drift Detection:** Compare "Predicted vs. Actual" grades to see if the model's accuracy is dropping.
* **One-Click Retraining:** Administrators can upload fresh data to retrain and update the model without writing code.

---

## 🛠️ Tech Stack

* **Python 3.9+**
* **Streamlit** (Frontend Interface)
* **Scikit-Learn** (Machine Learning Model)
* **Pandas** (Data Manipulation & Cleaning)
* **Matplotlib** (Data Visualization)
* **Groq API** (Generative AI - Llama 3)

---

## 💻 How to Run Locally

1.  **Clone the repository**
    ```bash
    git clone [https://github.com/Nerge-sam/student-insight-lab.git](https://github.com/Nerge-sam/student-insight-lab.git)
    cd student-insight-lab
    ```

2.  **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Set up API Keys**
    * Create a `.env` file in the main folder (do not commit this file).
    * Add your Groq API key:
        ```text
        GROQ_API_KEY=gsk_your_key_here
        ```

4.  **Run the App**
    ```bash
    streamlit run student_app.py
    ```

## 📁 Project Structure

```text
├── student_app.py             # Main application file
├── student_grade_predictor.pkl # Trained ML Model (Linear Regression)
├── requirements.txt           # Python dependencies
├── student_scores.csv         # Sample dataset for testing/retraining
└── .env                       # API Keys (Local only - Not uploaded to GitHub)