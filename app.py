import pandas as pd
import streamlit as st
from inference import CreditScoreInference

SCORE_COLOR = {"Good": "green", "Standard": "orange", "Poor": "red"}
MONTHS = ["January", "February", "March", "April", "May", "June",
          "July", "August", "September", "October", "November", "December"]
CREDIT_MIX = ["Good", "Standard", "Bad", "Unknown"]
MIN_AMOUNT = ["Yes", "No", "NM"]
PAYMENT_BEHAVIOUR = [
    "High_spent_Large_value_payments", "High_spent_Medium_value_payments",
    "High_spent_Small_value_payments", "Low_spent_Large_value_payments",
    "Low_spent_Medium_value_payments", "Low_spent_Small_value_payments", "Unknown",
]
LOAN_TYPES = ["Auto Loan", "Credit-Builder Loan", "Debt Consolidation Loan", "Home Equity Loan",
              "Mortgage Loan", "Payday Loan", "Personal Loan", "Student Loan"]
CLASSES = ["Poor", "Standard", "Good"]

@st.cache_resource
def load_model():
    return CreditScoreInference()

inferencer = load_model()

def main():
    st.set_page_config(page_title="Credit Score Classifier", layout="wide", initial_sidebar_state="expanded")
    st.title("Credit Score Classification")
    st.markdown("***Frederick Allensius - 2802405030 - Dataset C - UAS Model Deployment 2025/2026***")
    st.caption("Assess an applicant's credit score (Good / Standard / Poor). Fill the profile and predict.")

    # applicant identity
    with st.sidebar:
        st.header("Applicant identity")
        st.caption("Identifier fields - dropped before modelling, so they do **not** affect the prediction.")
        name = st.text_input("Name", "Frederick Allensius")
        customer_id = st.text_input("Customer ID", "CUS_0x1234")
        ssn = st.text_input("SSN", "123-45-6789")

    with st.form("applicant"):
        with st.expander("Income & balance", expanded=True):
            c1, c2, c3 = st.columns(3)
            annual_income = c1.number_input("Annual Income", 0.0, 1e8, 35000.0, step=1000.0)
            monthly_salary = c2.number_input("Monthly Inhand Salary", 0.0, 1e8, 3000.0, step=100.0)
            monthly_balance = c3.number_input("Monthly Balance", 0.0, 1e8, 350.0, step=10.0)
            outstanding_debt = c1.number_input("Outstanding Debt", 0.0, 1e8, 1000.0, step=50.0)
            total_emi = c2.number_input("Total EMI per month", 0.0, 1e8, 100.0, step=10.0)
            amount_invested = c3.number_input("Amount Invested Monthly", 0.0, 1e8, 80.0, step=10.0)

        with st.expander("Credit profile", expanded=True):
            c1, c2, c3 = st.columns(3)
            age = c1.number_input("Age", 18, 140, 33)
            num_bank = c2.number_input("Num Bank Accounts", 1, 20, 5)
            num_card = c3.number_input("Num Credit Cards", 0, 20, 5)
            interest = c1.number_input("Interest Rate (%)", 1, 100, 13)
            num_loan = c2.number_input("Num of Loans", 0, 20, 3)
            credit_mix = c3.selectbox("Credit Mix", CREDIT_MIX)
            util = c1.slider("Credit Utilization Ratio (%)", 0.0, 100.0, 32.0)
            ch_years = c2.number_input("Credit History (years)", 0, 100, 15)
            ch_months = c3.number_input("Credit History (extra months)", 0, 11, 6)

        with st.expander("Payment behaviour", expanded=True):
            c1, c2, c3 = st.columns(3)
            delay_due = c1.number_input("Delay from due date", 0, 80, 15)
            num_delayed = c2.number_input("Num of Delayed Payments", 0, 30, 12)
            changed_limit = c3.number_input("Changed Credit Limit", -10.0, 40.0, 10.0)
            num_inquiries = c1.number_input("Num Credit Inquiries", 0, 40, 6)
            min_amount = c2.selectbox("Payment of Min Amount", MIN_AMOUNT)
            behaviour = c3.selectbox("Payment Behaviour", PAYMENT_BEHAVIOUR)

        with st.expander("Loans & schedule", expanded=True):
            c1, c2 = st.columns(2)
            month = c1.selectbox("Month", MONTHS)
            occupation = c2.text_input("Occupation", "Engineer")   # free text; non-matching -> all-zeros
            loans = st.multiselect("Type of Loan", LOAN_TYPES, default=["Personal Loan"])

        submitted = st.form_submit_button("Predict credit score", type="primary", use_container_width=True)

    features = {
        "Month": month, "Age": age, "Occupation": occupation,
        "Annual_Income": annual_income, "Monthly_Inhand_Salary": monthly_salary,
        "Num_Bank_Accounts": num_bank, "Num_Credit_Card": num_card, "Interest_Rate": interest,
        "Num_of_Loan": num_loan, "Type_of_Loan": ", ".join(loans) if loans else "Not Specified",
        "Delay_from_due_date": delay_due, "Num_of_Delayed_Payment": num_delayed,
        "Changed_Credit_Limit": changed_limit, "Num_Credit_Inquiries": num_inquiries,
        "Credit_Mix": credit_mix, "Outstanding_Debt": outstanding_debt,
        "Credit_Utilization_Ratio": util,
        "Credit_History_Age": f"{ch_years} Years and {ch_months} Months",
        "Payment_of_Min_Amount": min_amount, "Total_EMI_per_month": total_emi,
        "Amount_invested_monthly": amount_invested, "Payment_Behaviour": behaviour,
        "Monthly_Balance": monthly_balance,
    }
    record = {"Name": name, "Customer_ID": customer_id, "SSN": ssn, **features}

    st.divider()
    if submitted:
        result = inferencer.predict([record])
        label = result["labels"][0]
        probs = {CLASSES[i]: float(result["probabilities"][0][i]) for i in range(len(CLASSES))}
        st.markdown(f"## Predicted Credit Score: :{SCORE_COLOR.get(label, 'blue')}[{label}]")
        cols = st.columns(len(CLASSES))
        for col, c in zip(cols, CLASSES):
            col.metric(c, f"{probs[c]:.1%}")
        st.bar_chart(pd.DataFrame({"Probability": [probs[c] for c in CLASSES]}, index=CLASSES), height=220)
        with st.expander("Features used for this prediction"):
            st.caption("Only these features are used by the model - the identity fields above are excluded.")
            st.dataframe(pd.Series(features, name="Value").to_frame(), use_container_width=True)
    else:
        st.info("Fill in the applicant profile above and click **Predict credit score**.")

if __name__ == "__main__":
    main()
