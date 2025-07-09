import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 🎨 Konsisten styling & palette
sns.set_style("whitegrid")
sns.set_palette("Set2")
st.set_page_config(page_title="Employee Satisfaction", layout="wide")

# --- Load Dataset ---
@st.cache_data
def load_data():
    return pd.read_csv("cleaned_employeesurvey.csv")

df = load_data()

# --- Sidebar Navigation ---
st.sidebar.title("🔧 Navigation")
page = st.sidebar.radio("Go to", ["📊 Dashboard", "📝 Predict Satisfaction"])

# ==========================================
# =============== DASHBOARD ===============
# ==========================================
if page == "📊 Dashboard":
    st.title("Welcome to Employee Job Satisfaction Dashboard 📊")

    st.markdown("""
    ### About This Dashboard

    This interactive dashboard presents an in-depth analysis of employee job satisfaction, based on survey data covering various work-related and lifestyle attributes.

    It aims to help **HR teams, analysts, and management** to understand what factors most influence employee satisfaction — such as:
    - Work-Life Balance (WLB)
    - Work Environment
    - Stress & Sleep Patterns
    - Employment Type & Experience

    📈 You can explore key trends, correlations, and department-level differences.  
    📝 You can also try the prediction tool to simulate your own work conditions and get an estimated satisfaction score.
    """)

    st.markdown("Explore satisfaction level, workload, stress, and lifestyle impact across departments.")

    # Dataset Preview
    st.subheader("📂 Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🔥 Correlation Heatmap")
        fig1, ax1 = plt.subplots()
        corr = df[['job_satisfaction', 'wlb', 'work_env', 'workload', 'stress', 'sleep_hours']].corr()
        sns.heatmap(corr, annot=True, cmap="YlGnBu", fmt=".2f", ax=ax1)
        st.pyplot(fig1)

    with col2:
        st.subheader("🏢 Avg. Satisfaction by Department")
        dept_avg = df.groupby("dept")["job_satisfaction"].mean().sort_values()
        fig2, ax2 = plt.subplots()
        dept_avg.plot(kind='barh', color=sns.color_palette()[0], ax=ax2)
        ax2.set_xlabel("Avg Satisfaction")
        st.pyplot(fig2)

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("🧠 Stress vs Sleep Hours")
        fig3, ax3 = plt.subplots()
        sns.scatterplot(data=df, x="sleep_hours", y="stress", hue="job_satisfaction", palette="Set2", ax=ax3)
        ax3.set_title("Sleep vs Stress (colored by Satisfaction)")
        st.pyplot(fig3)

    with col4:
        st.subheader("⏳ Satisfaction by Experience")
        fig4, ax4 = plt.subplots()
        sns.boxplot(x="experience", y="job_satisfaction", data=df, palette="Set2", ax=ax4)
        ax4.set_title("Experience vs Satisfaction")
        st.pyplot(fig4)

    # Tambahan chart dari Power BI (satu per baris)
    st.subheader("📈 Additional Visualizations")

    st.markdown("#### 📌 Satisfaction by Experience")
    fig5, ax5 = plt.subplots(figsize=(10, 3))
    df_exp = df.groupby("experience")["job_satisfaction"].mean().reset_index()
    sns.lineplot(data=df_exp, x="experience", y="job_satisfaction", marker="o", color=sns.color_palette()[0], ax=ax5)
    ax5.set_ylabel("Avg Satisfaction")
    ax5.set_xlabel("Years of Experience")
    st.pyplot(fig5)
    st.markdown("""
    <p style='font-weight:normal; color:black;'>Employees with 10–15 years of experience tend to report the highest satisfaction.</p>
    """, unsafe_allow_html=True)

    st.markdown("#### 😫 Stress Level by Age")
    fig6, ax6 = plt.subplots(figsize=(10, 3))
    df_age = df.groupby("age")["stress"].mean().reset_index()
    sns.lineplot(data=df_age, x="age", y="stress", marker="o", color=sns.color_palette()[1], ax=ax6)
    ax6.set_ylabel("Avg Stress")
    ax6.set_xlabel("Age")
    st.pyplot(fig6)
    st.markdown("""
    <p style='font-weight:normal; color:black;'>Stress levels fluctuate across age groups, with some spikes around late 30s and 50s.</p>
    """, unsafe_allow_html=True)

    col7, col8 = st.columns(2)

    with col7:
        st.markdown("#### 👨‍💼 Satisfaction by Employee Type")
        emp_type_avg = df.groupby("emp_type")["job_satisfaction"].mean().sort_values()
        fig7, ax7 = plt.subplots(figsize=(5, 3))
        emp_type_avg.plot(kind='barh', color=sns.color_palette()[2], ax=ax7)
        ax7.set_xlabel("Avg Satisfaction")
        st.pyplot(fig7)
        st.markdown("""
        <p style='font-weight:normal; color:black; text-align:left;'>Full-Time employees tend to have slightly higher satisfaction than Contract and Part-Time.</p>
        """, unsafe_allow_html=True)

    with col8:
        st.markdown("#### 🛌 Sleep Hours per Department")
        dept_sleep = df.groupby("dept")["sleep_hours"].mean().sort_values(ascending=False)
        fig8, ax8 = plt.subplots(figsize=(5, 3))
        dept_sleep.plot(kind="line", marker="o", color=sns.color_palette()[3], ax=ax8)
        ax8.set_ylabel("Avg Sleep Hours")
        ax8.set_xlabel("Department")
        fig8.autofmt_xdate(rotation=30)
        st.pyplot(fig8)
        st.markdown("""
        <p style='font-weight:normal; color:black; text-align:left;'>Customer Service and IT departments report the highest average sleep hours.</p>
        """, unsafe_allow_html=True)

    st.subheader("💡 Insights")
    st.markdown("""
    <ul style="color:black;">
        <li><strong>Work-Life Balance (WLB)</strong> and <strong>Work Environment</strong> are strong positive contributors to satisfaction.</li>
        <li>High <strong>workload</strong> and <strong>stress</strong> reduce satisfaction levels.</li>
        <li>Mid-career employees (10–15 years) tend to report the highest satisfaction.</li>
        <li>Full-Time workers are generally more satisfied than Part-Time or Contract workers.</li>
    </ul>
    """, unsafe_allow_html=True)

# ==========================================
# ========== INPUT & PREDICTION ============
# ==========================================
elif page == "📝 Predict Satisfaction":
    st.title("📝 Predict Your Job Satisfaction Score")
    st.markdown("Input your work conditions to see a predicted satisfaction score and feedback.")

    with st.form("form_satisfaction"):
        st.subheader("🧪 Personal Information ")
        emp_type = st.selectbox("Employment Type", ["Full-Time", "Part-Time", "Contract"])
        dept = st.selectbox("Department", sorted(df["dept"].unique()))
        experience = st.slider("Years of Experience", 0, 40, 5)

        st.subheader("💼 Work Conditions")
        wlb = st.slider("Work-Life Balance (1 = Poor, 5 = Excellent)", 1, 5, 3)
        work_env = st.slider("Work Environment (1 = Toxic, 5 = Great)", 1, 5, 3)
        workload = st.slider("Workload (1 = Light, 10 = Heavy)", 1, 10, 5)
        stress = st.slider("Stress Level (1 = Calm, 10 = Overwhelming)", 1, 10, 5)
        sleep_hours = st.slider("Sleep Hours per Day", 0.0, 12.0, 7.0, step=0.5)

        submitted = st.form_submit_button("**PREDICT**")

    if submitted:
        # Simple heuristic-based prediction
        score = (
            wlb * 0.25 +
            work_env * 0.25 +
            (10 - workload) * 0.2 +
            (10 - stress) * 0.15 +
            sleep_hours * 0.15
        )
        score = round(min(score, 5.0), 2)

        st.markdown(f"### 🎯 Predicted Satisfaction Score: **{score} / 5.0**")

        if score >= 4.2:
            st.success("🌟 Excellent! You're likely very satisfied with your work.")
            st.balloons()
        elif score >= 3.2:
            st.info("🙂 Moderate satisfaction. Still room for improvement.")
        else:
            st.warning("⚠️ Consider reviewing your workload or stress levels.")

        st.markdown("---")
        st.markdown("📌 *Note: This is a heuristic-based model. You can improve this with real ML models in the future.*")
