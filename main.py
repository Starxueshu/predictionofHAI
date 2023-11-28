# This is a sample Python script.
#import pickle
import joblib as jl
import pandas as pd
import streamlit as st

st.header("An interactive artificial intelligence application to stratify the risk of hospital-acquired infection in geriatric hip fracture: a national cohort study")
st.sidebar.title("Parameters Selection Panel")
st.sidebar.markdown("Picking up parameters")

Age = st.sidebar.selectbox("Age", ("50-59 years", "60-69 years", "70-79 years", "80-89 years", "90-100 years", ">100 years"))
Sex = st.sidebar.selectbox("Sex", ("Male", "Female"))
Fracture = st.sidebar.selectbox("Fracture type ", ("Femoral neck fracture", "Intertrochanteric fracture"))
Operation = st.sidebar.selectbox("Operation", ("Hip joint replacement", "Internal fixation", "Others"))
Comorbidities = st.sidebar.selectbox("Number of comorbidities", ("0", "1", "2", "3"))
Anemia = st.sidebar.selectbox("Anemia", ("No", "Yes"))
Hypertension = st.sidebar.selectbox("Hypertension", ("No", "Yes"))
Coronarydisease = st.sidebar.selectbox("Coronary disease", ("No", "Yes"))
Cerebrovasculardisease = st.sidebar.selectbox("Cerebrovascular disease", ("No", "Yes"))
Heartfailure = st.sidebar.selectbox("Heart failure", ("No", "Yes"))
Atherosclerosis = st.sidebar.selectbox("Atherosclerosis", ("No", "Yes"))
Renalfailure = st.sidebar.selectbox("Renal failure", ("No", "Yes"))
Nephroticsyndrome = st.sidebar.selectbox("Nephrotic syndrome", ("No", "Yes"))
Respiratorysystemdisease = st.sidebar.selectbox("Respiratory system disease", ("No", "Yes"))
Gastrointestinalbleeding = st.sidebar.selectbox("Gastrointestinal bleeding", ("No", "Yes"))
Liverfailure = st.sidebar.selectbox("Liver failure", ("No", "Yes"))
Cirrhosis = st.sidebar.selectbox("Cirrhosis", ("No", "Yes"))
Diabetes = st.sidebar.selectbox("Diabetes", ("No", "Yes"))
Dementia = st.sidebar.selectbox("Dementia", ("No", "Yes"))
Cancer = st.sidebar.selectbox("Cancer", ("No", "Yes"))

if st.button("Submit"):
    rf_clf = jl.load("Xgbc_clf_final_round.pkl")
    x = pd.DataFrame([[Age, Sex, Fracture, Operation, Comorbidities, Anemia, Hypertension, Coronarydisease, Cerebrovasculardisease, Heartfailure, Atherosclerosis, Renalfailure, Nephroticsyndrome, Respiratorysystemdisease, Gastrointestinalbleeding, Liverfailure, Cirrhosis, Diabetes, Dementia, Cancer]],
                     columns=["Age", "Sex", "Fracture", "Operation", "Comorbidities", "Anemia", "Hypertension", "Coronarydisease", "Cerebrovasculardisease", "Heartfailure", "Atherosclerosis", "Renalfailure", "Nephroticsyndrome", "Respiratorysystemdisease", "Gastrointestinalbleeding", "Liverfailure", "Cirrhosis", "Diabetes", "Dementia", "Cancer"])
    x = x.replace(["Male", "Female"], [1, 2])
    x = x.replace(["50-59 years", "60-69 years", "70-79 years", "80-89 years", "90-100 years", ">100 years"], [5, 6, 7, 8, 9, 10])
    x = x.replace(["Femoral neck fracture", "Intertrochanteric fracture"], [1, 2])
    x = x.replace(["Hip joint replacement", "Internal fixation", "Others"], [1, 2, 3])
    x = x.replace(["0", "1", "2", "3"], [0, 1, 2, 3])
    x = x.replace(["No", "Yes"], [0, 1])


    # Get prediction
    prediction = rf_clf.predict_proba(x)[0, 1]
        # Output prediction
    st.text(f"Predicted probability of HAI: {'{:.2%}'.format(round(prediction, 5))}")
    if prediction < 0.526:
        st.markdown('Risk stratification:')
        st.success(f"low-risk group")
    else:
        st.markdown('Risk stratification:')
        st.error(f"High-risk group")
    if prediction < 0.526:
        st.success('Therapeutic recommendation:')
        st.markdown(f"For low-risk individuals, the AI tool’s predictions can provide reassurance and guide clinical management accordingly. While preventive measures should still be in place for all patients, the focus may be more on general infection control practices rather than targeted interventions. This could include routine hand hygiene, appropriate antimicrobial use, and adherence to standard infection prevention protocols.")
    else:
        st.error('Therapeutic recommendation:')
        st.markdown(f"For high-risk individuals identified by the AI tool, healthcare providers can implement targeted preventive measures to reduce the risk of HAI. This may involve avoiding invasive procedures like urinary catheterization unless absolutely necessary, as well as implementing strict infection control practices and removing catheters as soon as possible, if appropriate. Additionally, healthcare providers can prioritize hand hygiene, antimicrobial stewardship, and surveillance programs to minimize the risk of infections.")

st.subheader('AI information')
st.markdown('Among all the models tested in the study, the XGBM model demonstrated the best performance, achieving an impressive area under the curve (AUC) of 0.817. The developed AI tool, accessible through an online application, provides personalized predictions of HAI risk, risk classification, and therapeutic recommendations for geriatric hip fracture patients. Nonetheless, it is important to note that while the AI application provides valuable risk estimates and recommendations, clinical decision-making should always incorporate the expertise of healthcare providers and take into account the individual patient context. The AI tool serves as a helpful aid in predicting HAI risk and guiding preventive strategies, but it should not replace the clinical judgment and experience of healthcare professionals.')
