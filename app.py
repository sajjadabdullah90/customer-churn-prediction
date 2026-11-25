from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="ChurnIQ | Retention Intelligence", page_icon="📈", layout="wide", initial_sidebar_state="expanded")

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "dataset" / "Telco-Customer-Churn.csv"
ARTIFACTS = ROOT / "artifacts"
MODELS = ARTIFACTS / "models"

# ---------- Visual system ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
:root{--bg:#0b1020;--panel:#121b2e;--panel2:#17243a;--line:#273752;--muted:#91a3bf;--text:#eef4ff;--cyan:#61d5ed;--green:#55d6a4;--pink:#ff8298}
html,body,[class*="css"]{font-family:'DM Sans',sans-serif}
.stApp{background:radial-gradient(ellipse at 8% 0%,#172a45 0%,transparent 34%),var(--bg);color:var(--text)}
[data-testid="stSidebar"]{background:#0e1729;border-right:1px solid var(--line)}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h1{font-family:Manrope,sans-serif}
.block-container{padding-top:2rem;padding-bottom:3rem;max-width:1500px}
.hero{padding:1.7rem 1.9rem;border:1px solid #304763;border-radius:22px;background:linear-gradient(115deg,rgba(31,58,88,.92),rgba(18,29,49,.96) 65%,rgba(20,43,58,.9));margin-bottom:1.3rem;box-shadow:0 14px 40px rgba(0,0,0,.16)}
.hero .eyebrow{font-size:.68rem;letter-spacing:.17em;color:#80d9e9;font-weight:800}
.hero h1{font-family:Manrope,sans-serif;font-size:2rem;letter-spacing:-.04em;color:#f5f8ff;margin:.45rem 0}
.hero p{color:#b4c5dc;margin:0;font-size:.96rem}
.metric-card{background:linear-gradient(145deg,#17253b,#111b2d);border:1px solid #2b405c;border-radius:17px;padding:1.05rem 1.15rem;min-height:126px;box-shadow:0 8px 24px rgba(0,0,0,.12)}
.metric-label{color:#9eb1cc;font-size:.69rem;font-weight:700;letter-spacing:.09em}
.metric-value{color:#f5f8ff;font:800 1.75rem Manrope,sans-serif;margin-top:.45rem;letter-spacing:-.04em}
.metric-note{color:#8094b2;font-size:.74rem;margin-top:.2rem}
[data-testid="stMetric"]{background:linear-gradient(145deg,#17253b,#111b2d);border:1px solid #2b405c;padding:15px 17px;border-radius:16px}
[data-testid="stMetricLabel"]{color:#a8bad3}
[data-testid="stMetricValue"]{color:#f5f8ff}
[data-testid="stTabs"] button{font-weight:650}
[data-testid="stDataFrame"]{border:1px solid #2b405c;border-radius:12px;overflow:hidden}
div.stButton>button,div.stDownloadButton>button{border-radius:11px;font-weight:700;border:1px solid #42617f}
div.stButton>button[kind="primary"],div.stFormSubmitButton>button{background:linear-gradient(100deg,#64d8e8,#69dfb4);color:#071522;border:0}
div[data-testid="stForm"]{background:rgba(18,29,46,.62);border:1px solid #2a3e59;border-radius:17px;padding:1rem}
div[data-testid="stAlert"]{border-radius:13px}
.section-note{color:#91a3bf;font-size:.84rem}
div[data-testid="stExpander"]{border:1px solid #2b405c;border-radius:13px}
footer{visibility:hidden}
</style>
""", unsafe_allow_html=True)

def hero(title, subtitle, eyebrow="CUSTOMER CHURN INTELLIGENCE"):
    st.markdown(f'<div class="hero"><div class="eyebrow">{eyebrow}</div><h1>{title}</h1><p>{subtitle}</p></div>', unsafe_allow_html=True)

def metric(label, value, note):
    st.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div><div class="metric-note">{note}</div></div>', unsafe_allow_html=True)

def polish(fig, height=350):
    fig.update_layout(height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#dce8f8",family="DM Sans"), margin=dict(l=18,r=18,t=52,b=18),
        legend=dict(bgcolor="rgba(0,0,0,0)"))
    fig.update_xaxes(gridcolor="#263752",zerolinecolor="#263752")
    fig.update_yaxes(gridcolor="#263752",zerolinecolor="#263752")
    return fig

def pct(x):
    return f"{float(x):.2%}"

@st.cache_data
def load_data(path):
    frame = pd.read_csv(path)
    frame["TotalCharges"] = pd.to_numeric(frame["TotalCharges"], errors="coerce")
    return frame

@st.cache_data
def load_table(path):
    return pd.read_csv(path)

@st.cache_data
def load_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)

@st.cache_resource
def load_models():
    return {
        "Logistic Regression": joblib.load(MODELS / "logistic_regression.joblib"),
        "Random Forest": joblib.load(MODELS / "random_forest.joblib")
    }

required = [
    DATA, ARTIFACTS / "test_metrics.csv", ARTIFACTS / "cross_validation_metrics.csv",
    ARTIFACTS / "test_predictions.csv", ARTIFACTS / "classification_reports.json",
    ARTIFACTS / "confusion_matrices.json", ARTIFACTS / "error_analysis.csv",
    ARTIFACTS / "metadata.json", MODELS / "logistic_regression.joblib",
    MODELS / "random_forest.joblib"
]
missing = [p.relative_to(ROOT).as_posix() for p in required if not p.exists()]
if missing:
    st.error("Dashboard artifacts are missing. Run the notebook from top to bottom, including Section 20 (Exporting Results for the Interactive Dashboard), then restart Streamlit.")
    st.markdown("**Missing files:**")
    for item in missing:
        st.code(item)
    st.stop()

try:
    raw = load_data(str(DATA))
    metrics_df = load_table(str(ARTIFACTS / "test_metrics.csv"))
    cv_df = load_table(str(ARTIFACTS / "cross_validation_metrics.csv"))
    predictions = load_table(str(ARTIFACTS / "test_predictions.csv"))
    errors = load_table(str(ARTIFACTS / "error_analysis.csv"))
    reports = load_json(str(ARTIFACTS / "classification_reports.json"))
    matrices = load_json(str(ARTIFACTS / "confusion_matrices.json"))
    metadata = load_json(str(ARTIFACTS / "metadata.json"))
    models = load_models()
except Exception as exc:
    st.error(f"Could not load the notebook exports: {exc}")
    st.stop()

selected = metrics_df.loc[metrics_df["Model"] == metadata["selected_model"]].iloc[0]
test_size = len(predictions)
churn_rate = (raw["Churn"] == "Yes").mean()

st.sidebar.markdown("# 📈 ChurnIQ")
st.sidebar.caption("Retention intelligence workspace")
st.sidebar.markdown("---")
page = st.sidebar.radio("WORKSPACE", [
    "Executive Overview", "Customer Insights", "Churn Predictor",
    "Model Performance", "Error Analysis", "Data Explorer"
], label_visibility="collapsed")
st.sidebar.markdown("---")
st.sidebar.markdown("**MODEL STATUS**")
st.sidebar.success("Notebook artifacts loaded")
st.sidebar.caption(f"Selected model · {metadata['selected_model']}")
st.sidebar.caption(f"Test records · {test_size:,}")
st.sidebar.markdown("---")
st.sidebar.caption("IBM Telco Customer Churn · AI Engineering Bootcamp")

# ---------- Executive overview ----------
if page == "Executive Overview":
    hero("Understand churn. Protect customer value.", "A clear view of customer retention patterns, model performance, and the customers most likely to leave.", "RETENTION PERFORMANCE / OVERVIEW")
    cols = st.columns(4)
    cards = [
        ("CUSTOMER BASE", f"{len(raw):,}", "Historical customer records"),
        ("OBSERVED CHURN", pct(churn_rate), "Share of customers who left"),
        ("MODEL ACCURACY", pct(selected["Accuracy"]), metadata["selected_model"]),
        ("CHURN RECALL", pct(selected["Churn Recall"]), "Share of churners identified")
    ]
    for col, item in zip(cols, cards):
        with col: metric(*item)
    left, right = st.columns([1,1])
    with left:
        counts = raw["Churn"].value_counts().rename_axis("Status").reset_index(name="Customers")
        fig = px.pie(counts, names="Status", values="Customers", hole=.72,
            color="Status", color_discrete_map={"No":"#61d5ed","Yes":"#ff8298"}, title="Customer retention mix")
        fig.update_traces(textposition="outside",textinfo="percent+label",marker=dict(line=dict(color="#101a2b",width=3)))
        fig.add_annotation(text=f"{len(raw):,}<br><span style='font-size:11px'>customers</span>",showarrow=False,font=dict(size=22,color="#f3f7ff"))
        st.plotly_chart(polish(fig,360),width="stretch")
    with right:
        contract = raw.groupby("Contract",observed=False)["Churn"].apply(lambda s:(s=="Yes").mean()*100).reset_index(name="Churn rate")
        fig = px.bar(contract,x="Contract",y="Churn rate",color="Churn rate",text=contract["Churn rate"].map(lambda v:f"{v:.1f}%"),
            color_continuous_scale=["#59d6a9","#f5c66b","#ff8298"],title="Churn rate by contract")
        fig.update_traces(textposition="outside",cliponaxis=False)
        st.plotly_chart(polish(fig,360),width="stretch")
    st.subheader("Model scorecard")
    display = metrics_df.copy()
    for col in display.columns:
        if col != "Model": display[col] = display[col].map(lambda v:pct(v) if pd.notna(v) else "—")
    st.dataframe(display,hide_index=True,width="stretch")
    st.info("Interpretation: churn recall measures how many customers who actually left were identified. Precision measures how often a churn alert was correct. Review both alongside accuracy.")

# ---------- Customer insights ----------
elif page == "Customer Insights":
    hero("Explore the signals behind churn", "Compare customer segments and discover where retention risk is concentrated.", "CUSTOMER INTELLIGENCE / SEGMENTS")
    a,b,c = st.columns(3)
    with a: metric("AVERAGE TENURE",f"{raw.tenure.mean():.1f} mo","Across all customers")
    with b: metric("MONTHLY CHARGES",f"${raw.MonthlyCharges.mean():.2f}","Average per customer")
    with c:
            service_cols=["PhoneService","MultipleLines","OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport","StreamingTV","StreamingMovies"]
            avg_services=raw[service_cols].eq("Yes").sum(axis=1).mean()
            metric("AVG SERVICES",f"{avg_services:.1f}","Per customer")
    t1,t2,t3 = st.tabs(["Churn drivers","Customer behaviour","Feature relationships"])
    with t1:
        feature = st.selectbox("Choose a customer attribute",["Contract","InternetService","PaymentMethod","OnlineSecurity","TechSupport","OnlineBackup","DeviceProtection","PaperlessBilling","Partner","Dependents","MultipleLines","StreamingTV","StreamingMovies"])
        rates = raw.groupby(feature,observed=False)["Churn"].apply(lambda s:(s=="Yes").mean()*100).reset_index(name="Churn rate").sort_values("Churn rate")
        fig = px.bar(rates,x="Churn rate",y=feature,orientation="h",text=rates["Churn rate"].map(lambda v:f"{v:.1f}%"),color="Churn rate",
            color_continuous_scale=["#59d6a9","#f5c66b","#ff8298"],title=f"Churn rate across {feature}")
        fig.update_traces(textposition="outside",cliponaxis=False)
        st.plotly_chart(polish(fig,max(350,45*len(rates))),width="stretch")
    with t2:
        left,right = st.columns(2)
        with left:
            fig=px.histogram(raw,x="tenure",color="Churn",nbins=30,barmode="overlay",opacity=.8,
                color_discrete_map={"No":"#61d5ed","Yes":"#ff8298"},title="Customer tenure distribution")
            st.plotly_chart(polish(fig),width="stretch")
        with right:
            fig=px.box(raw,x="Churn",y="MonthlyCharges",color="Churn",
                color_discrete_map={"No":"#61d5ed","Yes":"#ff8298"},title="Monthly charges by churn status")
            st.plotly_chart(polish(fig),width="stretch")
        fig=px.box(raw,x="Churn",y="tenure",color="Churn",
            color_discrete_map={"No":"#61d5ed","Yes":"#ff8298"},title="Tenure by churn status")
        st.plotly_chart(polish(fig),width="stretch")
    with t3:
        numeric = raw.select_dtypes(include="number")
        corr=numeric.corr(numeric_only=True)
        fig=px.imshow(corr,text_auto=".2f",aspect="auto",color_continuous_scale="RdBu_r",zmin=-1,zmax=1,title="Numerical feature correlation")
        st.plotly_chart(polish(fig,560),width="stretch")
        st.caption("Correlation describes association, not causation.")

# ---------- Predictor ----------
elif page == "Churn Predictor":
    hero("Assess an individual customer", "Use the exact fitted notebook pipeline to estimate churn probability from a customer profile.", "PREDICTIVE WORKSPACE / INDIVIDUAL RISK")
    st.caption("Prediction output is an estimate from historical data, not a guarantee or an automated retention decision.")
    with st.form("customer_prediction"):
        st.subheader("Customer profile")
        c1,c2,c3=st.columns(3)
        with c1:
            senior=st.selectbox("Senior citizen",["No","Yes"])
            partner=st.selectbox("Has partner",["Yes","No"])
            dependents=st.selectbox("Has dependents",["Yes","No"])
            tenure=st.slider("Tenure (months)",0,72,12)
            phone=st.selectbox("Phone service",["Yes","No"])
            lines=st.selectbox("Multiple lines",["No","Yes","No phone service"])
        with c2:
            internet=st.selectbox("Internet service",["DSL","Fiber optic","No"])
            security=st.selectbox("Online security",["Yes","No","No internet service"])
            backup=st.selectbox("Online backup",["Yes","No","No internet service"])
            protection=st.selectbox("Device protection",["Yes","No","No internet service"])
            support=st.selectbox("Tech support",["Yes","No","No internet service"])
        with c3:
            tv=st.selectbox("Streaming TV",["Yes","No","No internet service"])
            movies=st.selectbox("Streaming movies",["Yes","No","No internet service"])
            contract=st.selectbox("Contract",["Month-to-month","One year","Two year"])
            paperless=st.selectbox("Paperless billing",["Yes","No"])
            payment=st.selectbox("Payment method",["Electronic check","Mailed check","Bank transfer (automatic)","Credit card (automatic)"])
        st.subheader("Billing information")
        x,y=st.columns(2)
        with x: monthly=st.number_input("Monthly charges (USD)",min_value=0.0,max_value=2000.0,value=70.0,step=1.0)
        with y: total=st.number_input("Total charges (USD)",min_value=0.0,max_value=100000.0,value=float(tenure*monthly),step=10.0)
        model_name=st.selectbox("Prediction model",list(models.keys()),index=0)
        submit=st.form_submit_button("Calculate churn probability",type="primary",width="stretch")
    if submit:
        tenure_group=pd.cut(pd.Series([tenure]),[-1,12,24,48,72],labels=["0-12 months","13-24 months","25-48 months","49-72 months"]).iloc[0]
        total_services=sum(v=="Yes" for v in [phone,lines,security,backup,protection,support,tv,movies])
        profile={
            "SeniorCitizen":int(senior=="Yes"),"Partner":partner,"Dependents":dependents,"tenure":tenure,
            "MultipleLines":lines,"InternetService":internet,"OnlineSecurity":security,"OnlineBackup":backup,
            "DeviceProtection":protection,"TechSupport":support,"StreamingTV":tv,"StreamingMovies":movies,
            "Contract":contract,"PaperlessBilling":paperless,"PaymentMethod":payment,
            "MonthlyCharges":monthly,"TotalCharges":total,"TotalServices":total_services,"TenureGroup":tenure_group
        }
        row=pd.DataFrame([{col:profile[col] for col in metadata["feature_columns"]}])
        fitted=models[model_name]
        probability=float(fitted.predict_proba(row)[0][list(fitted.classes_).index("Yes")])
        prediction=fitted.predict(row)[0]
        band="High" if probability>=.60 else "Moderate" if probability>=.35 else "Low"
        color={"High":"#ff8298","Moderate":"#f5c66b","Low":"#55d6a4"}[band]
        st.markdown("---")
        st.subheader("Prediction result")
        r1,r2,r3=st.columns(3)
        with r1: metric("PREDICTED OUTCOME","Likely to churn" if prediction=="Yes" else "Likely to stay",model_name)
        with r2: metric("CHURN PROBABILITY",f"{probability:.1%}","Model-estimated probability")
        with r3: metric("RISK BAND",band,"Low <35% · Moderate 35–60% · High ≥60%")
        st.progress(probability,text=f"Estimated churn probability · {probability:.1%}")
        if prediction=="Yes": st.warning("This customer is classified as likely to churn. Review their contract, billing experience, and service needs before taking action.")
        else: st.success("This customer is classified as likely to stay. Continue regular engagement and monitor future behaviour.")
        with st.expander("Inspect submitted customer features"):
            st.dataframe(row,hide_index=True,width="stretch")

# ---------- Model performance ----------
elif page == "Model Performance":
    hero("Evaluate model quality beyond accuracy", "Review the notebook's held-out test metrics, cross-validation results, confusion matrices, and classification reports.", "MODEL GOVERNANCE / EVALUATION")
    st.caption("All evaluation values shown here are loaded from the notebook export. No model training or metric recalculation occurs in this app.")
    st.subheader("Held-out test performance")
    shown=metrics_df.copy()
    for col in shown.columns:
        if col!="Model": shown[col]=shown[col].map(lambda v:pct(v) if pd.notna(v) else "—")
    st.dataframe(shown,hide_index=True,width="stretch")
    st.markdown("**Accuracy** · overall correctness  |  **Precision** · reliability of churn alerts  |  **Recall** · actual churners identified  |  **F1** · precision/recall balance  |  **ROC-AUC** · ranking separation")
    st.subheader("Five-fold stratified cross-validation")
    cv=cv_df.copy()
    for col in cv.columns:
        if col!="Model": cv[col]=cv[col].map(lambda v:pct(v) if pd.notna(v) else "—")
    st.dataframe(cv,hide_index=True,width="stretch")
    st.caption("Cross-validation was performed on training data only; the final test set remained held out.")
    name=st.selectbox("Inspect model diagnostics",list(matrices.keys()))
    cm=np.asarray(matrices[name])
    left,right=st.columns(2)
    with left:
        fig=px.imshow(cm,text_auto=True,x=["Predicted No","Predicted Yes"],y=["Actual No","Actual Yes"],
            color_continuous_scale="Blues",aspect="auto",title=f"{name} · confusion matrix")
        st.plotly_chart(polish(fig,370),width="stretch")
        tn,fp,fn,tp=cm.ravel()
        st.markdown(f"**TN** {tn:,}  ·  **FP** {fp:,}  ·  **FN** {fn:,}  ·  **TP** {tp:,}")
        st.metric("Specificity",pct(tn/(tn+fp)) if tn+fp else "N/A")
    with right:
        st.markdown("**Classification report**")
        st.dataframe(pd.DataFrame(reports[name]).T.style.format("{:.3f}"),width="stretch")
    st.subheader("ROC curves")
    fig=go.Figure()
    actual=(predictions["Actual Churn"]=="Yes").astype(int)
    for label,col in [("Logistic Regression","Logistic Regression Churn Probability"),("Random Forest","Random Forest Churn Probability")]:
        p=predictions[col].to_numpy()
        # The notebook has already exported probability scores; plot them without fitting or evaluating another model.
        from sklearn.metrics import roc_curve
        fpr,tpr,_=roc_curve(actual,p)
        auc_value=metrics_df.loc[metrics_df.Model==label,"ROC-AUC"].iloc[0]
        fig.add_trace(go.Scatter(x=fpr,y=tpr,mode="lines",name=f"{label} · AUC {auc_value:.3f}"))
    fig.add_trace(go.Scatter(x=[0,1],y=[0,1],mode="lines",name="Random baseline",line=dict(dash="dash",color="#718096")))
    fig.update_layout(title="Receiver operating characteristic",xaxis_title="False positive rate",yaxis_title="True positive rate")
    st.plotly_chart(polish(fig,400),width="stretch")

# ---------- Error analysis ----------
elif page == "Error Analysis":
    hero("Understand where predictions go wrong", "Inspect false positives and false negatives from the notebook's selected Logistic Regression model.", "MODEL GOVERNANCE / ERROR REVIEW")
    fp=int((errors["Actual Churn"]=="No").sum() and ((errors["Actual Churn"]=="No") & (errors["Predicted Churn"]=="Yes")).sum())
    fn=int(((errors["Actual Churn"]=="Yes") & (errors["Predicted Churn"]=="No")).sum())
    cols=st.columns(3)
    for col,item in zip(cols,[("TOTAL INCORRECT",f"{len(errors):,}","Held-out test errors"),("FALSE POSITIVES",f"{fp:,}","Predicted churn, actually stayed"),("FALSE NEGATIVES",f"{fn:,}","Predicted stay, actually churned")]):
        with col: metric(*item)
    errors=errors.copy()
    errors["Error Type"]=np.where((errors["Actual Churn"]=="No")&(errors["Predicted Churn"]=="Yes"),"False Positive","False Negative")
    counts=errors["Error Type"].value_counts().rename_axis("Error type").reset_index(name="Count")
    fig=px.bar(counts,x="Error type",y="Count",color="Error type",text="Count",
        color_discrete_map={"False Positive":"#f5c66b","False Negative":"#ff8298"},title="Incorrect prediction breakdown")
    fig.update_traces(textposition="outside")
    st.plotly_chart(polish(fig,330),width="stretch")
    st.warning("False negatives are missed churners: customers who left but were predicted to stay. These cases may represent missed retention opportunities.")
    choice=st.radio("Show records",["All errors","False positives","False negatives"],horizontal=True)
    subset=errors if choice=="All errors" else errors[errors["Error Type"]==("False Positive" if choice=="False positives" else "False Negative")]
    st.dataframe(subset.head(250),hide_index=True,width="stretch")
    st.download_button("Download selected error records",subset.to_csv(index=False).encode("utf-8"),"churn_error_analysis.csv","text/csv")

# ---------- Data explorer ----------
elif page == "Data Explorer":
    hero("Explore the underlying customer records", "Filter the source dataset, inspect customer profiles, and export a segment for further analysis.", "DATA WORKSPACE / EXPLORATION")
    c1,c2,c3=st.columns(3)
    with c1: churn=st.multiselect("Churn status",["Yes","No"],default=["Yes","No"])
    with c2:
        contracts=sorted(raw["Contract"].dropna().unique().tolist())
        contract=st.multiselect("Contract type",contracts,default=contracts)
    with c3: tenure=st.slider("Tenure range (months)",0,72,(0,72))
    subset=raw[raw["Churn"].isin(churn)&raw["Contract"].isin(contract)&raw["tenure"].between(*tenure)]
    st.markdown(f"### {len(subset):,} <span class='section-note'>matching customer records</span>",unsafe_allow_html=True)
    st.dataframe(subset,hide_index=True,width="stretch",height=480)
    st.download_button("Export filtered records",subset.to_csv(index=False).encode("utf-8"),"filtered_customer_records.csv","text/csv")

st.markdown("---")
st.caption("ChurnIQ · IBM Telco Customer Churn · Educational analytics project · Predictions are estimates, not guarantees.")
