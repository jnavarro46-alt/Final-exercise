import pandas as pd
import streamlit as st
from pathlib import Path
from modules import data_processing as dp, exploratory_analysis as eda, prediction, dashboard

st.set_page_config(page_title="SAD Predictive System", layout="wide")
PAGES = ["Home", "Midterm Exam", "Dataset", "Data Analysis", "Prediction", "Dashboard", "System Design"]
menu = st.sidebar.selectbox("Module", PAGES)
st.sidebar.caption("Predictions are decision support only. Use synthetic or authorized data.")


@st.cache_data
def load_sample():
    return pd.read_csv("data/facility_reservations.csv")


def current_df():
    return st.session_state.get("df", load_sample())


st.title("Campus Facility Reservation Monitoring")

if menu == "Home":
    st.write("Predicts **cancellation / no-show** risk for facility reservations, with a risk queue for "
             "**facility administrator review**. Built by deploying the Midterm Laboratory Examination model.")
    st.markdown("- **Midterm Exam**: original code, notebook output logic and results\n- **Dataset**: upload and validate CSV\n"
                "- **Data Analysis**: EDA\n- **Prediction**: saved pipeline inference\n- **Dashboard**: KPIs, review queue\n- **System Design**: SAD artifacts")
    st.info("Differences from the midterm: preprocessing is now one saved scikit-learn pipeline (joblib), and "
            "Reservation_Date is converted to Day_Of_Week before modelling.")

elif menu == "Midterm Exam":
    st.header("Midterm Laboratory Examination (Parts I–IV)")
    meta = prediction.load_model()["meta"]
    st.write("Target: **Reservation_Outcome**. Features: Facility_Type, User_Category, Lead_Time_Days, Participants, Reservation_Date. "
             "Model: Decision Tree, 80/20 split, random_state=42.")
    cols = st.columns(4)
    for c, (k, v) in zip(cols, meta["metrics"].items()):
        c.metric(k, f"{v:.3f}")
    code = Path("midterm/midterm_code.py").read_text(encoding="utf-8")
    nb = Path("midterm/Midterm_Notebook.ipynb")
    st.download_button("Download midterm code (.py)", code, "midterm_code.py")
    if nb.exists():
        st.download_button("Download original notebook (.ipynb)", nb.read_bytes(), nb.name)
    with st.expander("View original midterm code", expanded=True):
        st.code(code, language="python")
    st.subheader("Run midterm preprocessing on current dataset")
    errs = dp.validate(current_df())
    if errs: st.error(errs[0])
    else:
        d = dp.engineer(current_df())
        st.dataframe(d.head(10)); st.write(d["Reservation_Outcome"].value_counts())

elif menu == "Dataset":
    up = st.file_uploader("Upload CSV", type="csv")
    if up is not None:
        try:
            new = pd.read_csv(up)
            errs = dp.validate(new)
            if errs: st.error(errs[0])
            else: st.session_state["df"] = new; st.success("File accepted and now used by all modules.")
        except Exception as e:
            st.error(f"Could not read the file as CSV: {e}")
    else:
        st.info("No upload yet: showing the bundled synthetic sample dataset.")
    df = current_df(); r = dp.quality_report(df)
    st.write(f"Rows: **{r['rows']}** | Columns: **{r['columns']}** | Duplicates: **{r['duplicates']}**")
    st.write("Required columns:", dp.REQUIRED)
    st.dataframe(df.head(20), use_container_width=True)
    st.dataframe(r["nulls"])

elif menu == "Data Analysis":
    errs = dp.validate(current_df())
    st.error(errs[0]) if errs else eda.render(current_df())

elif menu == "Prediction":
    prediction.render()

elif menu == "Dashboard":
    errs = dp.validate(current_df())
    st.error(errs[0]) if errs else dashboard.render(current_df())

else:
    st.header("System Design (SAD Integration)")
    st.subheader("Functional requirements")
    st.markdown("1. FR1: Upload a reservation CSV and validate its schema, nulls and duplicates.\n"
                "2. FR2: Show descriptive statistics, class distribution and interactive charts.\n"
                "3. FR3: Predict cancellation/no-show risk and probability for a new reservation.\n"
                "4. FR4: Show KPIs and a ranked risk/review queue with recommended follow-up.\n"
                "5. FR5: Let the Facility Administrator record a final decision with justification.")
    st.subheader("Non-functional requirements")
    st.markdown("- NFR1 **Performance**: the model loads once (cached); no retraining per refresh.\n"
                "- NFR2 **Privacy**: only synthetic/authorized data; no passwords or student records in the repo.\n"
                "- NFR3 **Transparency**: model version, timestamp, limitations and human review are always visible.")
    st.subheader("Updated use case diagram")
    st.image("assets/use_case_diagram.png")
    st.write("**Actors:** Data Analyst (upload, EDA, predict), Facility Administrator (view results, record final decisions), "
             "Department Head (view dashboard only).")
    st.subheader("Decision workflow")
    st.graphviz_chart("""digraph { rankdir=LR; node [shape=box];
      "Upload CSV" -> "Validate" -> "Feature engineering" -> "Saved model" -> "Risk score" -> "Review queue" -> "Administrator decision" -> "Decision log" }""")
