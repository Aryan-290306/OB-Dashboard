"""
StaySignal: full Python version (model + dashboard)

Setup:  pip install streamlit pandas numpy scikit-learn
Run:    streamlit run staysignal_app.py

Data:   IBM HR Analytics Employee Attrition dataset (1,470 employees, fictional public benchmark).
        Downloaded automatically. If you have the Kaggle CSV, save it next to this file as
        WA_Fn-UseC_-HR-Employee-Attrition.csv and it will be used instead.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

URL = "https://raw.githubusercontent.com/IBM/employee-attrition-aif360/master/data/emp_attrition.csv"
LOCAL = Path(__file__).with_name("WA_Fn-UseC_-HR-Employee-Attrition.csv")
LOW, HIGH = 0.12, 0.30                      # risk-band cut-offs
BANDS = ["Low", "Medium", "High"]

# key, label, category, how it would be captured, actionable?
FACTORS = [
    ("OverTime", "Overtime", "Workload & engagement", "System data: timesheets", True),
    ("BusinessTravel", "Business travel", "Workload & engagement", "System data: travel records", True),
    ("JobInvolvement", "Job involvement", "Workload & engagement", "Manager rating (1-4)", True),
    ("MonthlyIncome", "Monthly income", "Compensation & growth", "System data: payroll", True),
    ("PercentSalaryHike", "Last salary hike", "Compensation & growth", "System data: payroll", True),
    ("StockOptionLevel", "Stock option level", "Compensation & growth", "System data: benefits records", True),
    ("YearsSinceLastPromotion", "Years since last promotion", "Compensation & growth", "System data: HRIS", True),
    ("TrainingTimesLastYear", "Trainings last year", "Compensation & growth", "System data: learning system", True),
    ("JobSatisfaction", "Job satisfaction", "Satisfaction & environment", "Employee pulse survey (1-4)", True),
    ("EnvironmentSatisfaction", "Environment satisfaction", "Satisfaction & environment", "Employee pulse survey (1-4)", True),
    ("RelationshipSatisfaction", "Relationship satisfaction", "Satisfaction & environment", "Employee pulse survey (1-4)", True),
    ("WorkLifeBalance", "Work-life balance", "Satisfaction & environment", "Employee pulse survey (1-4)", True),
    ("TotalWorkingYears", "Total working years", "Career trajectory", "System data: HRIS", False),
    ("YearsInCurrentRole", "Years in current role", "Career trajectory", "System data: HRIS", True),
    ("YearsWithCurrManager", "Years with current manager", "Career trajectory", "System data: HRIS", True),
    ("NumCompaniesWorked", "Companies worked at before", "Career trajectory", "System data: application records", False),
    ("DistanceFromHome", "Distance from home", "Logistics", "System data: HRIS address", True),
]
KEYS = [f[0] for f in FACTORS]
LABEL = {f[0]: f[1] for f in FACTORS}
ACTIONABLE = {f[0]: f[4] for f in FACTORS}

# Rule-based actions linked to OB theory (the model predicts risk; it does not learn which actions work)
ACTIONS = {
    "workload": ("Rebalance workload", "Find which tasks drive the overtime, redistribute them and agree a cap.", "JD-R model: lowers job demands", ["OverTime"]),
    "travel": ("Rotate or reduce travel", "Share travel across the team or swap some trips for remote meetings.", "JD-R model: lowers job demands", ["BusinessTravel"]),
    "role": ("Make the role more meaningful", "Offer a stretch assignment or more ownership.", "Herzberg: motivator (the work itself)", ["JobInvolvement"]),
    "pay": ("Review pay against peers", "Compare pay and last raise with similar roles; close unjustified gaps.", "Herzberg: hygiene factor; equity theory", ["MonthlyIncome", "PercentSalaryHike"]),
    "incentive": ("Offer a longer-term incentive", "Consider stock options or a dated retention bonus.", "Expectancy theory: a reward worth staying for", ["StockOptionLevel"]),
    "career": ("Hold a career-path conversation", "Agree the next role, what it takes and a realistic date.", "Herzberg: motivator (advancement); expectancy theory", ["YearsSinceLastPromotion", "YearsInCurrentRole"]),
    "develop": ("Plan development", "Agree a training or skill-building project.", "Herzberg: motivator (growth)", ["TrainingTimesLastYear"]),
    "stay": ("Run a stay interview", "Ask what keeps them, what could push them out, and what one thing you could change.", "Herzberg: hygiene factors and motivators", ["JobSatisfaction", "EnvironmentSatisfaction", "RelationshipSatisfaction"]),
    "flex": ("Offer flexibility", "Discuss flexible hours, protected time off or hybrid work.", "JD-R model: adds job resources", ["WorkLifeBalance", "DistanceFromHome"]),
    "manager": ("Check the manager relationship", "Arrange a neutral or skip-level conversation.", "Herzberg: hygiene factor (supervision)", ["YearsWithCurrManager"]),
}
ACTION_OF = {k: a for a, (_, _, _, ks) in ACTIONS.items() for k in ks}


# ---------------------------------------------------------------- model
def to_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Raw HR columns -> numeric model inputs. Age, gender and marital status are excluded on purpose."""
    X = raw[KEYS].copy()
    X["OverTime"] = X["OverTime"].map({"Yes": 1, "No": 0}).fillna(X["OverTime"]).astype(float)
    X["BusinessTravel"] = X["BusinessTravel"].map({"Non-Travel": 0, "Travel_Rarely": 1, "Travel_Frequently": 2}).fillna(X["BusinessTravel"]).astype(float)
    X["MonthlyIncome"] = np.log(X["MonthlyIncome"].astype(float))
    return X.astype(float)


@st.cache_resource(show_spinner="Training the model...")
def build():
    df = pd.read_csv(LOCAL if LOCAL.exists() else URL)
    y = (df["Attrition"] == "Yes").astype(int).values
    X = to_features(df)
    scaler = StandardScaler().fit(X)
    Z = scaler.transform(X)
    model = LogisticRegression(C=0.3, max_iter=5000)      # unweighted, so scores are real probabilities
    oof = cross_val_predict(model, Z, y, cv=StratifiedKFold(5, shuffle=True, random_state=42), method="predict_proba")[:, 1]
    model.fit(Z, y)
    df = df.copy()
    df["risk"] = model.predict_proba(Z)[:, 1]
    df["band"] = pd.cut(df["risk"], [-1, LOW, HIGH, 2], labels=BANDS)
    df["oof_band"] = pd.cut(oof, [-1, LOW, HIGH, 2], labels=BANDS)
    df["left"] = y
    return df, scaler, model


df, scaler, model = build()


def score(row: dict) -> float:
    X = to_features(pd.DataFrame([row]))
    return float(model.predict_proba(scaler.transform(X))[0, 1])


def contributions(row: dict) -> pd.Series:
    """Each factor's push on the log-odds of leaving (positive = raises risk)."""
    z = scaler.transform(to_features(pd.DataFrame([row])))[0]
    return pd.Series(model.coef_[0] * z, index=KEYS)


def reason(key: str, v) -> str:
    r4, wlb = ["low", "medium", "high", "very high"], ["bad", "good", "better", "best"]
    if key == "OverTime": return "Works overtime" if v in (1, "Yes") else "Does not work overtime"
    if key == "BusinessTravel":
        v = {"Non-Travel": 0, "Travel_Rarely": 1, "Travel_Frequently": 2}.get(v, v)
        return {0: "Does not travel for work", 1: "Travels occasionally", 2: "Travels frequently"}[int(v)]
    if key != "MonthlyIncome": v = int(v)
    if key == "JobInvolvement": return f"Job involvement is {r4[v-1]} ({v} of 4)"
    if key == "MonthlyIncome":
        share_higher = (df["MonthlyIncome"] > v).mean()
        return (f"Monthly income of {v:,.0f} is lower than {share_higher:.0%} of colleagues" if share_higher >= .5
                else f"Monthly income of {v:,.0f} is higher than {(df['MonthlyIncome'] < v).mean():.0%} of colleagues")
    if key == "PercentSalaryHike": return f"Last salary hike was {v}%"
    if key == "StockOptionLevel": return f"Has stock options (level {v})" if v else "Has no stock options"
    if key == "YearsSinceLastPromotion": return "Promoted within the last year" if v == 0 else f"{v} years since last promotion"
    if key == "TrainingTimesLastYear": return "No trainings last year" if v == 0 else f"{v} trainings last year"
    if key == "JobSatisfaction": return f"Job satisfaction is {r4[v-1]} ({v} of 4)"
    if key == "EnvironmentSatisfaction": return f"Satisfaction with the work environment is {r4[v-1]} ({v} of 4)"
    if key == "RelationshipSatisfaction": return f"Satisfaction with colleagues is {r4[v-1]} ({v} of 4)"
    if key == "WorkLifeBalance": return f"Work-life balance is {wlb[v-1]} ({v} of 4)"
    if key == "TotalWorkingYears": return f"{v} years of total experience"
    if key == "YearsInCurrentRole": return f"{v} years in the current role"
    if key == "YearsWithCurrManager": return f"{v} years with the current manager"
    if key == "NumCompaniesWorked": return "No earlier employers" if v == 0 else f"Has worked at {v} other companies"
    if key == "DistanceFromHome": return f"{'Long' if v >= 15 else 'Short' if v <= 5 else 'Moderate'} commute (distance {v})"
    return LABEL[key]


def band_of(p: float) -> str:
    return BANDS[0] if p < LOW else BANDS[1] if p < HIGH else BANDS[2]


# ---------------------------------------------------------------- app
st.set_page_config(page_title="StaySignal", page_icon="📶", layout="wide")
st.title("StaySignal")
st.caption("Predicting who is likely to leave, why, and what a manager could do about it. "
           "Built on the IBM HR Analytics Employee Attrition dataset (fictional, public).")
tab_over, tab_emp, tab_fac, tab_mod = st.tabs(["Overview", "Employees", "Factors", "Model and limits"])

# ---- Overview
with tab_over:
    st.subheader("Does the score tell leavers from stayers?")
    st.caption("Share of each group that actually left. Cross-validated scores: each employee was scored by a model that had not seen them.")
    val = df.groupby("oof_band", observed=True)["left"].agg(["count", "mean"]).reindex(BANDS)
    c = st.columns(3)
    for col, b in zip(c, BANDS):
        col.metric(f"{b} band ({int(val.loc[b, 'count'])} employees)", f"{val.loc[b, 'mean']:.0%} left")
    st.bar_chart(val["mean"].rename("Share that left"))
    st.caption(f"Across all {len(df):,} employees, {df['left'].mean():.0%} left.")

    st.subheader("Where risk concentrates")
    for col, title in (("Department", "By department"), ("JobRole", "By job role")):
        g = df.groupby(col).agg(Staff=("left", "size"), High_band=("band", lambda s: (s == "High").mean()),
                                Average_risk=("risk", "mean"), Actually_left=("left", "mean")).sort_values("High_band", ascending=False)
        st.markdown(f"**{title}**")
        st.dataframe(g.style.format({"High_band": "{:.0%}", "Average_risk": "{:.0%}", "Actually_left": "{:.0%}"}))

# ---- Employees
with tab_emp:
    f1, f2, f3 = st.columns(3)
    dept = f1.selectbox("Department", ["All"] + sorted(df["Department"].unique()))
    role = f2.selectbox("Job role", ["All"] + sorted(df["JobRole"].unique()))
    band = f3.selectbox("Risk band", ["All", "High", "Medium", "Low"])
    view = df[(dept == "All") | (df["Department"] == dept)]
    view = view[(role == "All") | (view["JobRole"] == role)]
    view = view[(band == "All") | (view["band"] == band)].sort_values("risk", ascending=False)
    if view.empty:
        st.info("No employees match these filters.")
        st.stop()
    emp_id = st.selectbox("Employee", view["EmployeeNumber"].tolist(),
                          format_func=lambda i: f"Employee {i}: {view.set_index('EmployeeNumber').loc[i, 'risk']:.0%} ({view.set_index('EmployeeNumber').loc[i, 'band']})")
    row = df[df["EmployeeNumber"] == emp_id].iloc[0]
    raw = {k: row[k] for k in KEYS}
    p0 = float(row["risk"])

    st.header(f"Employee {emp_id}")
    st.write(f"{row['JobRole']}, {row['Department']}")
    m1, m2 = st.columns(2)
    m1.metric("Chance of leaving", f"{p0:.0%}", f"{band_of(p0)} risk", delta_color="off")
    m2.metric("Average across all staff", f"{df['left'].mean():.0%}")
    st.progress(min(p0, 1.0))

    contrib = contributions(raw)
    up = contrib[contrib > 0.1].sort_values(ascending=False)
    down = contrib[contrib < -0.1].sort_values()
    left, right = st.columns(2)
    left.subheader("What is raising the risk")
    for k in up.index[:5]:
        left.write(f"- {reason(k, raw[k])}")
    if up.empty:
        left.write("No factor is pushing the risk up in a meaningful way.")
    right.subheader("What is working in their favour")
    for k in down.index[:4]:
        right.write(f"- {reason(k, raw[k])}")
    if down.empty:
        right.write("No factor stands out as protective.")

    st.subheader("Suggested actions")
    if band_of(p0) == "Low":
        st.write("This employee is in the low band, so no action is suggested. A routine check-in is enough.")
    else:
        grouped = {}
        for k in up.index:
            if ACTIONABLE[k] and k in ACTION_OF:
                a = grouped.setdefault(ACTION_OF[k], {"c": 0.0, "why": []})
                a["c"] += up[k]; a["why"].append(reason(k, raw[k]))
        for a_id, info in sorted(grouped.items(), key=lambda kv: -kv[1]["c"])[: 4 if band_of(p0) == "High" else 3]:
            title, text, theory, _ = ACTIONS[a_id]
            with st.container(border=True):
                st.markdown(f"**{title}**  \n{text}")
                st.caption(f"Because: {'; '.join(info['why'])}.  \n{theory}")
    st.caption("Actions come from simple rules linked to organisational behaviour theory. The model predicts risk; it does not learn which actions work.")

    # ---- What-if simulator
    st.subheader("What if their situation changed?")
    k = f"e{emp_id}_"          # per-employee widget keys, so changing employee resets the sliders
    s = dict(raw)
    w1, w2, w3 = st.columns(3)
    with w1:
        st.markdown("**Workload and engagement**")
        s["OverTime"] = 1 if st.radio("Overtime", ["No", "Yes"], index=int(raw["OverTime"] in (1, "Yes")), horizontal=True, key=k + "ot") == "Yes" else 0
        trav = {"Non-Travel": 0, "Travel_Rarely": 1, "Travel_Frequently": 2}.get(raw["BusinessTravel"], raw["BusinessTravel"])
        s["BusinessTravel"] = st.select_slider("Business travel (0 none, 1 rarely, 2 often)", [0, 1, 2], value=int(trav), key=k + "bt")
        s["JobInvolvement"] = st.select_slider("Job involvement (1 low, 4 very high)", [1, 2, 3, 4], value=int(raw["JobInvolvement"]), key=k + "ji")
        st.markdown("**Satisfaction**")
        for key in ("JobSatisfaction", "EnvironmentSatisfaction", "RelationshipSatisfaction", "WorkLifeBalance"):
            s[key] = st.select_slider(LABEL[key] + " (1 low, 4 high)", [1, 2, 3, 4], value=int(raw[key]), key=k + key)
    with w2:
        st.markdown("**Pay and growth**")
        pay = st.slider("Change in monthly income (%)", -20, 50, 0, 5, key=k + "pay")
        s["MonthlyIncome"] = raw["MonthlyIncome"] * (1 + pay / 100)
        s["StockOptionLevel"] = st.select_slider("Stock option level", [0, 1, 2, 3], value=int(raw["StockOptionLevel"]), key=k + "so")
        s["YearsSinceLastPromotion"] = st.slider("Years since last promotion", 0, max(15, int(raw["YearsSinceLastPromotion"])), int(raw["YearsSinceLastPromotion"]), key=k + "yp")
        s["TrainingTimesLastYear"] = st.slider("Trainings last year", 0, max(6, int(raw["TrainingTimesLastYear"])), int(raw["TrainingTimesLastYear"]), key=k + "tr")
        st.markdown("**Logistics**")
        s["DistanceFromHome"] = st.slider("Distance from home", 1, max(29, int(raw["DistanceFromHome"])), int(raw["DistanceFromHome"]), key=k + "dist")
    p1 = score(s)
    with w3:
        st.metric("Risk now", f"{p0:.0%}", band_of(p0), delta_color="off")
        st.metric("What-if risk", f"{p1:.0%}", f"{(p1 - p0) * 100:+.0f} points", delta_color="inverse")
        st.write(f"Band: **{band_of(p0)}** to **{band_of(p1)}**")
        st.progress(min(p1, 1.0))
    st.caption("This shows how the model's score responds to a change. It does not prove that making the change would keep this person.")

# ---- Factors
with tab_fac:
    st.subheader("Factors and how to measure them")
    st.write("The model uses 17 factors across 5 groups. System data is objective but only as good as record keeping; "
             "survey data captures feelings but depends on honest answers and response rates; manager ratings are open to rater bias.")
    coef = dict(zip(KEYS, model.coef_[0]))
    table = pd.DataFrame([{"Category": f[2], "Factor": f[1], "How it would be captured": f[3],
                           "Effect on risk": "raises" if coef[f[0]] > 0 else "lowers", "Strength": abs(coef[f[0]])} for f in FACTORS])
    st.dataframe(table, hide_index=True,
                 column_config={"Strength": st.column_config.ProgressColumn("Strength", min_value=0, max_value=float(table["Strength"].max()), format=" ")})
    st.info("Age, gender and marital status are excluded on purpose to avoid unfair flagging. Job level and years at the company "
            "were dropped because they overlap heavily with pay and other tenure measures.")

# ---- Model and limits
with tab_mod:
    y = df["left"].values
    hi = (df["oof_band"] == "High").values
    tp = int((hi & (y == 1)).sum())
    st.subheader("How well it works")
    st.write(f"- Logistic regression on {len(df):,} employees, {int(y.sum())} of whom left ({y.mean():.0%}).")
    st.write(f"- The high band flags {int(hi.sum())} employees ({hi.mean():.0%} of staff), catches {tp / y.sum():.0%} of all leavers "
             f"and is right {tp / hi.sum():.0%} of the time.")
    st.write(f"- A model that predicted nobody leaves would be {1 - y.mean():.0%} accurate, so accuracy is not reported.")
    st.subheader("What it cannot do")
    st.markdown("- The data is a fictional benchmark dataset, not records from a real company.\n"
                "- It is a single snapshot, not employees tracked over time.\n"
                "- Factors are associations; the what-if tool does not prove cause and effect.\n"
                "- Suggested actions are rules from OB theory; the data has no record of which actions worked.\n"
                "- No team or manager IDs, so risk is grouped by department and job role only.\n"
                "- A real rollout needs employee consent, clear rules on who sees scores, and fairness checks.")
