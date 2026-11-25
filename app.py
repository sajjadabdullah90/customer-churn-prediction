from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report,
 confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

st.set_page_config(page_title="ChurnIQ | Customer Analytics",page_icon="📊",layout="wide")
ROOT=Path(__file__).resolve().parent
DATA=ROOT/"dataset"/"Telco-Customer-Churn.csv"

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@600;700;800&display=swap');
html,body,[class*="css"]{font-family:'DM Sans',sans-serif}.stApp{background:#0b1020;color:#e8edf7}
[data-testid="stSidebar"]{background:#10182b;border-right:1px solid #26324a}
.hero{padding:1.5rem;border:1px solid #263652;border-radius:20px;background:linear-gradient(120deg,#172846,#111b31 55%,#18253d);margin-bottom:1rem}
.hero h1{font-family:Manrope,sans-serif;color:#f5f8ff;margin:.3rem 0}.hero p{color:#b6c4dc;margin:0}
.eyebrow{color:#8ea9d4;font-size:.72rem;font-weight:700;letter-spacing:.13em}
.metric-card{background:#121d32;border:1px solid #283955;border-radius:15px;padding:1rem;min-height:105px}
.metric-label{color:#9bacc7;font-size:.76rem;font-weight:600}.metric-value{color:#f5f8ff;font:800 1.8rem Manrope;margin-top:.3rem}.metric-note{color:#8295b4;font-size:.72rem}
[data-testid="stMetric"]{background:#121d32;border:1px solid #283955;padding:14px;border-radius:14px}
</style>""",unsafe_allow_html=True)

def hero(title,sub):
 st.markdown(f'<div class="hero"><div class="eyebrow">CUSTOMER CHURN INTELLIGENCE</div><h1>{title}</h1><p>{sub}</p></div>',unsafe_allow_html=True)
def card(label,value,note):
 st.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div><div class="metric-note">{note}</div></div>',unsafe_allow_html=True)
def pct(x): return f"{x:.2%}"
def style(fig,h=350):
 fig.update_layout(height=h,paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font_color="#dce6f7",margin=dict(l=15,r=15,t=45,b=15))
 fig.update_xaxes(gridcolor="#26344d");fig.update_yaxes(gridcolor="#26344d")
 return fig

@st.cache_data
def load(path):
 d=pd.read_csv(path);d["TotalCharges"]=pd.to_numeric(d["TotalCharges"],errors="coerce");return d
def engineer(d):
 d=d.copy()
 services=["PhoneService","MultipleLines","OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport","StreamingTV","StreamingMovies"]
 d["TotalServices"]=d[services].eq("Yes").sum(axis=1)
 d["TenureGroup"]=pd.cut(d.tenure,[-1,12,24,48,72],labels=["0-12 months","13-24 months","25-48 months","49-72 months"])
 return d

@st.cache_resource(show_spinner="Training models and calculating metrics...")
def train(path):
 raw=load(path);df=engineer(raw)
 X=df.drop(columns=["customerID","gender","PhoneService","Churn"]);y=df.Churn
 Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
 # Same dtype selection and preprocessing approach as the notebook.
 nums=Xtr.select_dtypes(include=["int64","float64"]).columns.tolist()
 cats=Xtr.select_dtypes(include=["object"]).columns.tolist()
 pre=ColumnTransformer([("num",Pipeline([("imputer",SimpleImputer(strategy="median")),("scale",StandardScaler())]),nums),
  ("cat",Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),("encode",OneHotEncoder(handle_unknown="ignore"))]),cats)])
 models={"Baseline":DummyClassifier(strategy="most_frequent"),
  "Logistic Regression":Pipeline([("preprocessor",pre),("classifier",LogisticRegression(max_iter=1000,random_state=42))]),
  "Random Forest":Pipeline([("preprocessor",pre),("classifier",RandomForestClassifier(n_estimators=200,random_state=42))])}
 cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=42);cvrows=[]
 for name,m in models.items():
  scores=cross_val_score(m,Xtr,ytr,cv=cv,scoring="accuracy")
  cvrows.append({"Model":name,"Mean CV Accuracy":scores.mean(),"CV Std. Dev.":scores.std()})
 preds={};probs={};reports={};cms={};rows=[]
 for name,m in models.items():
  m.fit(Xtr,ytr);p=m.predict(Xte);preds[name]=p
  reports[name]=classification_report(yte,p,labels=["No","Yes"],output_dict=True,zero_division=0)
  cms[name]=confusion_matrix(yte,p,labels=["No","Yes"])
  row={"Model":name,"Accuracy":accuracy_score(yte,p),"Balanced Accuracy":balanced_accuracy_score(yte,p),
   "Churn Precision":precision_score(yte,p,pos_label="Yes",zero_division=0),
   "Churn Recall":recall_score(yte,p,pos_label="Yes",zero_division=0),
   "Churn F1-score":f1_score(yte,p,pos_label="Yes",zero_division=0)}
  if hasattr(m,"predict_proba"):
   prob=m.predict_proba(Xte)[:,list(m.classes_).index("Yes")];probs[name]=prob
   row["ROC-AUC"]=roc_auc_score((yte=="Yes").astype(int),prob)
  else: row["ROC-AUC"]=np.nan
  rows.append(row)
 err=Xte.copy();err.insert(0,"customerID",raw.loc[Xte.index,"customerID"])
 err["Actual Churn"]=yte;err["Predicted Churn"]=preds["Logistic Regression"]
 err["Error Type"]=np.where(err["Actual Churn"]==err["Predicted Churn"],"Correct",np.where(err["Predicted Churn"]=="Yes","False Positive","False Negative"))
 err=err[err["Error Type"]!="Correct"]
 return dict(raw=raw,df=df,X=X,y=y,Xte=Xte,yte=yte,models=models,preds=preds,probs=probs,
  cv=pd.DataFrame(cvrows),metrics=pd.DataFrame(rows),reports=reports,cms=cms,errors=err)

if not DATA.exists():
 st.error("Dataset missing: dataset/Telco-Customer-Churn.csv. Run Streamlit from the repository root.");st.stop()
try: d=train(str(DATA))
except Exception as e: st.exception(e);st.stop()
raw=d["raw"];df=d["df"];m=d["metrics"]

st.sidebar.markdown("## 📊 ChurnIQ")
st.sidebar.caption("Customer retention intelligence")
page=st.sidebar.radio("NAVIGATION",["Overview","Customer Analytics","Predict Churn","Model Evaluation","Error Analysis","Data Explorer"])
st.sidebar.divider();st.sidebar.caption("IBM Telco Customer Churn · AI Engineering Bootcamp")

if page=="Overview":
 hero("Customer Churn Analytics","Explore customer behaviour, identify churn patterns, compare ML models, and estimate individual customer risk.")
 lr=m[m.Model=="Logistic Regression"].iloc[0]
 cols=st.columns(4)
 for c,(a,b,n) in zip(cols,[("TOTAL CUSTOMERS",f"{len(raw):,}","Dataset records"),("OBSERVED CHURN",pct((raw.Churn=="Yes").mean()),"Historical churn rate"),("TEST ACCURACY",pct(lr.Accuracy),"Logistic Regression"),("CHURN RECALL",pct(lr["Churn Recall"]),"Actual churners identified")]):
  with c: card(a,b,n)
 x,y=st.columns(2)
 with x:
  cnt=raw.Churn.value_counts().rename_axis("Churn").reset_index(name="Customers")
  fig=px.pie(cnt,names="Churn",values="Customers",hole=.65,color="Churn",color_discrete_map={"No":"#38bdf8","Yes":"#fb7185"},title="Churn distribution")
  st.plotly_chart(style(fig),use_container_width=True)
 with y:
  con=raw.groupby("Contract",observed=False).Churn.apply(lambda s:(s=="Yes").mean()*100).reset_index(name="Churn rate")
  fig=px.bar(con,x="Contract",y="Churn rate",color="Contract",title="Churn rate by contract")
  st.plotly_chart(style(fig),use_container_width=True)
 st.subheader("Model comparison")
 v=m.copy()
 for col in v.columns[1:]: v[col]=v[col].map(lambda z:pct(z) if pd.notna(z) else "—")
 st.dataframe(v,use_container_width=True,hide_index=True)
 st.info("Accuracy alone is not enough for an imbalanced dataset. Precision and recall explain the quality of churn alerts.")

elif page=="Customer Analytics":
 hero("Customer Analytics","Explore churn patterns across contracts, services, tenure, and billing behaviour.")
 cols=st.columns(3)
 with cols[0]: card("AVG TENURE",f"{raw.tenure.mean():.1f} months","All customers")
 with cols[1]: card("AVG MONTHLY CHARGES",f"USD {raw.MonthlyCharges.mean():.2f}","Per customer")
 with cols[2]: card("AVG TOTAL CHARGES",f"USD {raw.TotalCharges.mean():,.2f}","Available values")
 t1,t2,t3=st.tabs(["Churn drivers","Customer behaviour","Correlations"])
 with t1:
  feature=st.selectbox("Analyse churn by",["Contract","InternetService","PaymentMethod","OnlineSecurity","TechSupport","OnlineBackup","DeviceProtection","PaperlessBilling","Partner","Dependents","MultipleLines","StreamingTV","StreamingMovies"])
  rate=df.groupby(feature,observed=False).Churn.apply(lambda s:(s=="Yes").mean()*100).reset_index(name="Churn Rate").sort_values("Churn Rate")
  fig=px.bar(rate,x="Churn Rate",y=feature,orientation="h",color="Churn Rate",color_continuous_scale=["#34d399","#fbbf24","#fb7185"],text=rate["Churn Rate"].map(lambda z:f"{z:.1f}%"))
  st.plotly_chart(style(fig,max(350,45*len(rate))),use_container_width=True)
 with t2:
  a,b=st.columns(2)
  with a:
   fig=px.histogram(df,x="tenure",color="Churn",nbins=30,barmode="overlay",opacity=.8,title="Tenure distribution",color_discrete_map={"No":"#38bdf8","Yes":"#fb7185"})
   st.plotly_chart(style(fig),use_container_width=True)
  with b:
   fig=px.box(df,x="Churn",y="MonthlyCharges",color="Churn",title="Monthly charges by churn")
   st.plotly_chart(style(fig),use_container_width=True)
  fig=px.box(df,x="Churn",y="tenure",color="Churn",title="Tenure by churn status")
  st.plotly_chart(style(fig),use_container_width=True)
 with t3:
  corr=df.select_dtypes(include=["number"]).corr(numeric_only=True)
  fig=px.imshow(corr,text_auto=".2f",aspect="auto",color_continuous_scale="RdBu_r",zmin=-1,zmax=1,title="Numerical feature correlation")
  st.plotly_chart(style(fig,550),use_container_width=True)
  st.caption("Correlation shows association, not causation.")

elif page=="Predict Churn":
 hero("Individual Churn Prediction","Enter a customer's details to estimate churn probability using the trained model pipeline.")
 st.caption("Predictions are estimates based on historical patterns, not guarantees.")
 with st.form("predict"):
  a,b,c=st.columns(3)
  with a:
   senior=st.selectbox("Senior citizen",[0,1],format_func=lambda z:"Yes" if z else "No")
   partner=st.selectbox("Partner",["Yes","No"]);depend=st.selectbox("Dependents",["Yes","No"])
   tenure=st.slider("Tenure (months)",0,72,12);phone=st.selectbox("Phone service",["Yes","No"])
   lines=st.selectbox("Multiple lines",["No","Yes","No phone service"])
  with b:
   internet=st.selectbox("Internet service",["DSL","Fiber optic","No"])
   security=st.selectbox("Online security",["Yes","No","No internet service"])
   backup=st.selectbox("Online backup",["Yes","No","No internet service"])
   protection=st.selectbox("Device protection",["Yes","No","No internet service"])
   support=st.selectbox("Tech support",["Yes","No","No internet service"])
  with c:
   tv=st.selectbox("Streaming TV",["Yes","No","No internet service"])
   movies=st.selectbox("Streaming movies",["Yes","No","No internet service"])
   contract=st.selectbox("Contract",["Month-to-month","One year","Two year"])
   paperless=st.selectbox("Paperless billing",["Yes","No"])
   payment=st.selectbox("Payment method",["Electronic check","Mailed check","Bank transfer (automatic)","Credit card (automatic)"])
  st.subheader("Billing")
  x,y=st.columns(2)
  with x: monthly=st.number_input("Monthly charges (USD)",0.0,2000.0,70.0,1.0)
  with y: total=st.number_input("Total charges (USD)",0.0,100000.0,float(tenure*monthly),10.0)
  modelname=st.selectbox("Model",["Logistic Regression","Random Forest"])
  submit=st.form_submit_button("Estimate churn risk",type="primary",use_container_width=True)
 if submit:
  tg=pd.cut(pd.Series([tenure]),[-1,12,24,48,72],labels=["0-12 months","13-24 months","25-48 months","49-72 months"]).iloc[0]
  services=sum(z=="Yes" for z in [phone,lines,security,backup,protection,support,tv,movies])
  row=pd.DataFrame([{"SeniorCitizen":senior,"Partner":partner,"Dependents":depend,"tenure":tenure,"MultipleLines":lines,"InternetService":internet,
   "OnlineSecurity":security,"OnlineBackup":backup,"DeviceProtection":protection,"TechSupport":support,"StreamingTV":tv,"StreamingMovies":movies,
   "Contract":contract,"PaperlessBilling":paperless,"PaymentMethod":payment,"MonthlyCharges":monthly,"TotalCharges":total,"TotalServices":services,"TenureGroup":tg}])
  model=d["models"][modelname];pred=model.predict(row)[0];prob=model.predict_proba(row)[0][list(model.classes_).index("Yes")]
  risk="High" if prob>=.60 else "Moderate" if prob>=.35 else "Low"
  col={"High":"#fb7185","Moderate":"#fbbf24","Low":"#34d399"}[risk]
  st.divider();st.subheader("Prediction result")
  x,y,z=st.columns(3)
  with x: card("PREDICTED CLASS","Likely to churn" if pred=="Yes" else "Likely to stay",modelname)
  with y: card("CHURN PROBABILITY",f"{prob:.1%}","Estimated model probability")
  with z: card("RISK BAND",risk,"Low <35% · Moderate 35–60% · High ≥60%")
  st.markdown(f'<div style="padding:14px;border:1px solid {col};border-radius:12px;background:#121d32"><b style="color:{col}">{risk} risk</b> · Predicted class: {pred}</div>',unsafe_allow_html=True)
  st.progress(float(prob),text=f"Estimated churn probability: {prob:.1%}")
  st.warning("Review contract, billing, and service experience before considering retention action.") if pred=="Yes" else st.success("Continue regular engagement and monitor customer behaviour.")
  with st.expander("View model input features"): st.dataframe(row,use_container_width=True,hide_index=True)

elif page=="Model Evaluation":
 hero("Model Evaluation & Decision Metrics","Held-out test results, stratified cross-validation, confusion matrices, and classification reports. Churn is the positive class.")
 st.subheader("Test-set metrics")
 v=m.copy()
 for col in v.columns[1:]: v[col]=v[col].map(lambda z:pct(z) if pd.notna(z) else "—")
 st.dataframe(v,use_container_width=True,hide_index=True)
 st.markdown("**Accuracy:** overall correct predictions. **Precision:** reliability of churn alerts. **Recall:** actual churners detected. **F1-score:** balance of precision and recall. **Specificity:** non-churners correctly identified. **ROC-AUC:** probability ranking quality.")
 st.subheader("Five-fold stratified cross-validation")
 cv=d["cv"].copy()
 for col in ["Mean CV Accuracy","CV Std. Dev."]: cv[col]=cv[col].map(pct)
 st.dataframe(cv,use_container_width=True,hide_index=True)
 st.caption("Cross-validation uses training data only; the test set remains held out.")
 name=st.selectbox("Inspect model",["Logistic Regression","Random Forest","Baseline"]);cm=d["cms"][name]
 x,y=st.columns(2)
 with x:
  fig=px.imshow(cm,text_auto=True,x=["Predicted No","Predicted Yes"],y=["Actual No","Actual Yes"],color_continuous_scale="Blues",title=f"{name} confusion matrix",aspect="auto")
  st.plotly_chart(style(fig,370),use_container_width=True)
  tn,fp,fn,tp=cm.ravel();st.write(f"**Specificity:** {tn/(tn+fp):.2%}" if tn+fp else "**Specificity:** N/A")
  st.write(f"TN {tn:,} · FP {fp:,} · FN {fn:,} · TP {tp:,}")
 with y:
  st.markdown("**Classification report**")
  st.dataframe(pd.DataFrame(d["reports"][name]).T.style.format("{:.3f}"),use_container_width=True)
 st.subheader("ROC curve")
 fig=go.Figure();yb=(d["yte"]=="Yes").astype(int)
 for n,p in d["probs"].items():
  fpr,tpr,_=roc_curve(yb,p);fig.add_trace(go.Scatter(x=fpr,y=tpr,mode="lines",name=f"{n} (AUC {roc_auc_score(yb,p):.3f})"))
 fig.add_trace(go.Scatter(x=[0,1],y=[0,1],mode="lines",name="Random baseline",line=dict(dash="dash")))
 fig.update_layout(xaxis_title="False Positive Rate",yaxis_title="True Positive Rate",title="Receiver Operating Characteristic")
 st.plotly_chart(style(fig,400),use_container_width=True)

elif page=="Error Analysis":
 hero("Prediction Error Analysis","Inspect incorrect Logistic Regression predictions and distinguish false positives from false negatives.")
 e=d["errors"];fp=int((e["Error Type"]=="False Positive").sum());fn=int((e["Error Type"]=="False Negative").sum())
 cols=st.columns(3)
 for c,(a,b,n) in zip(cols,[("TOTAL INCORRECT",f"{len(e):,}","Test records"),("FALSE POSITIVES",f"{fp:,}","Predicted churn, stayed"),("FALSE NEGATIVES",f"{fn:,}","Predicted stay, churned")]):
  with c: card(a,b,n)
 counts=e["Error Type"].value_counts().rename_axis("Error Type").reset_index(name="Count")
 fig=px.bar(counts,x="Error Type",y="Count",color="Error Type",text="Count",color_discrete_map={"False Positive":"#fbbf24","False Negative":"#fb7185"},title="Incorrect prediction types")
 st.plotly_chart(style(fig),use_container_width=True)
 st.markdown("**False negatives matter:** these are customers who churned but were predicted to stay, potentially representing missed retention opportunities.")
 choice=st.selectbox("Filter error records",["All errors","False Positives","False Negatives"])
 filtered=e if choice=="All errors" else e[e["Error Type"]==("False Positive" if choice=="False Positives" else "False Negative")]
 st.dataframe(filtered.head(150),use_container_width=True,hide_index=True)
 st.download_button("Download error records",filtered.to_csv(index=False).encode("utf-8"),"churn_prediction_errors.csv","text/csv")

elif page=="Data Explorer":
 hero("Customer Data Explorer","Filter source records, inspect customer segments, and export selected rows.")
 a,b,c=st.columns(3)
 with a: churn=st.multiselect("Churn status",["Yes","No"],default=["Yes","No"])
 with b:
  contracts=sorted(raw.Contract.dropna().unique().tolist());contract=st.multiselect("Contract type",contracts,default=contracts)
 with c: tenure=st.slider("Tenure range (months)",0,72,(0,72))
 subset=raw[raw.Churn.isin(churn)&raw.Contract.isin(contract)&raw.tenure.between(*tenure)]
 st.write(f"**{len(subset):,}** matching customers")
 st.dataframe(subset,use_container_width=True,hide_index=True,height=470)
 st.download_button("Download filtered customers",subset.to_csv(index=False).encode("utf-8"),"filtered_customers.csv","text/csv")

st.divider()
st.caption("ChurnIQ · IBM Telco Customer Churn · Educational ML project · Predictions are estimates, not guarantees.")
