import os
import sys
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
  sys.path.append(BASE_DIR)

from src.pre_screening import apply_hard_filters

# 1. ตั้งค่าหน้าเว็บหลัก
st.set_page_config(
    page_title="ระบบประเมินความเสี่ยงสินเชื่อ | สหกรณ์อิสลามอัศศิดดีก จำกัด",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 2. Custom CSS (โทนสีน้ำเงินเข้ม-เขียวน้ำทะเล ตามอัตลักษณ์ AIC)
st.markdown(
    """
<style>
    /* พื้นหลังหลักและฟอนต์ */
    .stApp {
        background-color: #F4F7F9;
        color: #1E293B;
    }
    
    /* แถบด้านข้าง (Sidebar) */
    [data-testid="stSidebar"] {
        background-color: #0B2545;
        color: #FFFFFF;
    }
    [data-testid="stSidebar"] * {
        color: #F8FAFC !important;
    }
    [data-testid="stSidebar"] hr {
        border-color: rgba(255, 255, 255, 0.15);
    }

    /* ส่วนหัวองค์กร (Corporate Header) */
    .aic-header {
        background: linear-gradient(135deg, #0B2545 0%, #134E4A 100%);
        padding: 1.5rem 2rem;
        border-radius: 8px;
        color: #FFFFFF;
        margin-bottom: 1.5rem;
        border-left: 6px solid #14B8A6;
        box-shadow: 0 2px 6px rgba(11, 37, 69, 0.12);
    }
    .aic-header h1 {
        color: #FFFFFF !important;
        font-size: 1.65rem;
        font-weight: 700;
        margin: 0 0 0.3rem 0;
        letter-spacing: 0.2px;
    }
    .aic-header p {
        color: #CCFBF1 !important;
        font-size: 0.95rem;
        margin: 0;
    }

    /* กล่องการ์ดเนื้อหา (Cards) */
    .aic-card {
        background-color: #FFFFFF;
        padding: 1.25rem 1.5rem;
        border-radius: 8px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
        margin-bottom: 1rem;
    }
    .aic-card-title {
        font-size: 1.05rem;
        font-weight: 600;
        color: #0B2545;
        margin-bottom: 0.75rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #F1F5F9;
    }

    /* ป้ายสถานะผลลัพธ์ (Status Badges) */
    .status-pass {
        background-color: #F0FDF4;
        border: 1px solid #BBF7D0;
        border-left: 5px solid #0D9488;
        color: #115E59;
        padding: 1rem 1.25rem;
        border-radius: 6px;
        margin-bottom: 1rem;
    }
    .status-risk {
        background-color: #FEF2F2;
        border: 1px solid #FECACA;
        border-left: 5px solid #DC2626;
        color: #991B1B;
        padding: 1rem 1.25rem;
        border-radius: 6px;
        margin-bottom: 1rem;
    }
    .status-warn {
        background-color: #FFFBEB;
        border: 1px solid #FDE68A;
        border-left: 5px solid #D97706;
        color: #92400E;
        padding: 0.85rem 1.25rem;
        border-radius: 6px;
        margin-bottom: 1rem;
    }

    /* แถบวัดระดับภาระหนี้ (Net DTI Progress Bar) */
    .dti-bar-bg {
        width: 100%;
        height: 12px;
        background-color: #E2E8F0;
        border-radius: 6px;
        overflow: hidden;
        margin: 0.5rem 0;
    }
    .dti-bar-fill {
        height: 100%;
        border-radius: 6px;
        transition: width 0.4s ease;
    }

    /* ปรับแต่งปุ่มกดหลัก */
    .stButton > button, [data-testid="stFormSubmitButton"] > button {
        background-color: #0D9488 !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 6px !important;
        padding: 0.5rem 1.5rem !important;
        font-weight: 600 !important;
    }
    .stButton > button:hover, [data-testid="stFormSubmitButton"] > button:hover {
        background-color: #0F766E !important;
    }

    /* ปรับแต่ง Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #E2E8F0;
        padding: 4px;
        border-radius: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px;
        padding: 8px 16px;
        font-weight: 600;
        color: #475569;
    }
    .stTabs [aria-selected="true"] {
        background-color: #FFFFFF !important;
        color: #0B2545 !important;
        box-shadow: 0 1px 2px rgba(0,0,0,0.08);
    }
</style>
""",
    unsafe_allow_html=True,
)


# 3. โหลดโมเดลและฐานข้อมูล
@st.cache_resource
def load_model():
  model_path = os.path.join(BASE_DIR, "models", "best_credit_model.pkl")
  return joblib.load(model_path)


@st.cache_data
def load_data():
  data_path = os.path.join(
      BASE_DIR, "data", "raw", "islamic_coop_synthetic_data.csv"
  )
  return pd.read_csv(data_path)


model_package = load_model()
pipeline = model_package["pipeline"]
optimal_threshold = model_package["optimal_threshold"]
df = load_data()

# จัดการ Session State สำหรับเก็บผลการประเมินและบันทึกเจ้าหน้าที่
if "assessment_result" not in st.session_state:
  st.session_state.assessment_result = None
if "officer_record" not in st.session_state:
  st.session_state.officer_record = {
      "decision": "สอดคล้องตามระบบประเมิน",
      "notes": "",
  }

# 4. แถบเมนูด้านข้าง (Sidebar)
with st.sidebar:
  st.markdown("### สหกรณ์อิสลามอัศศิดดีก จำกัด")
  st.caption("As-Siddiq Islamic Cooperative (AIC)")
  st.divider()
  menu = st.radio(
      "เมนูระบบงานสินเชื่อ",
      ["ประเมินสินเชื่อรายบุคคล", "ภาพรวมพอร์ตสินเชื่อสหกรณ์"],
      label_visibility="collapsed",
  )
  st.divider()
  st.caption(f"เกณฑ์เฝ้าระวังความเสี่ยง: >= {optimal_threshold*100:.2f}%")
  st.caption("เวอร์ชันระบบ: 2.0 (Production)")

# 5. ส่วนหัวของระบบ (Corporate Header)
st.markdown(
    """
<div class="aic-header">
    <h1>สหกรณ์อิสลามอัศศิดดีก จำกัด (AIC)</h1>
    <p>ระบบสนับสนุนการตัดสินใจและประเมินความเสี่ยงสินเชื่อ (Islamic Credit Scoring System)</p>
</div>
""",
    unsafe_allow_html=True,
)

# ==========================================
# หน้าต่างที่ 1: ระบบประเมินสินเชื่อรายบุคคล
# ==========================================
if menu == "ประเมินสินเชื่อรายบุคคล":
  st.markdown("#### ข้อมูลผู้ขอสินเชื่อและรายละเอียดคำขอ")

  with st.form("credit_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
      st.markdown("**1. ข้อมูลส่วนบุคคลและรายได้**")
      age = st.number_input("อายุผู้ขอสินเชื่อ (ปี)", 18, 75, 35)
      income = st.number_input("รายได้รวมต่อเดือน (บาท)", 7000, 300000, 30000)
      family_members = st.number_input("จำนวนสมาชิกในครอบครัว (คน)", 1, 15, 3)
      occupation = st.selectbox(
          "กลุ่มอาชีพ",
          [
              "ข้าราชการ/รัฐวิสาหกิจ",
              "พนักงานเอกชน",
              "เจ้าของธุรกิจ/ค้าขาย",
              "อาชีพอิสระ/เกษตรกร",
          ],
      )
      guarantor = st.selectbox(
          "ประเภทหลักประกัน/ผู้ค้ำ",
          [
              "ข้าราชการ",
              "เงินฝาก 100%",
              "โฉนดที่ดิน",
              "พนักงานเอกชน",
              "บุคคลทั่วไป",
          ],
      )

    with col2:
      st.markdown("**2. รายละเอียดวงเงินสินเชื่อ**")
      fin_amount = st.number_input(
          "วงเงินสินเชื่อที่ขอ (บาท)", 10000, 3000000, 150000
      )
      fin_tenure = st.selectbox(
          "ระยะเวลาผ่อนชำระ (งวด/เดือน)", [12, 24, 36, 48, 60, 84, 120, 240]
      )
      fin_type = st.selectbox(
          "ประเภทสัญญา/สินเชื่อ", ["Personal", "Business"]
      )
      collateral_val = st.number_input(
          "มูลค่าหลักประกันประเมิน (บาท)", 0, 5000000, 200000
      )
      distance = st.number_input("ระยะห่างที่อยู่/กิจการจากสาขา (กม.)", 1, 50, 10)

    with col3:
      st.markdown("**3. ประวัติสมาชิกและวินัยทางการเงิน**")
      member_months = st.number_input("อายุการเป็นสมาชิก (เดือน)", 1, 240, 24)
      share_cap = st.number_input("ทุนเรือนหุ้นสะสม (บาท)", 0, 1000000, 30000)
      consec_saving = st.number_input("ระยะเวลาออมต่อเนื่อง (เดือน)", 0, 120, 12)
      own_cap_ratio = (
          st.slider("สัดส่วนทุนส่วนตัวต่อโครงการ (%)", 0, 50, 15) / 100.0
      )
      hist_late = st.number_input("ประวัติค้างชำระในอดีต (ครั้ง)", 0, 10, 0)

    submitted = st.form_submit_button("ประมวลผลการประเมินสินเชื่อ")

  if submitted:
    # คำนวณตัวแปรเชิงลึกตามเกณฑ์สหกรณ์
    annual_expenses = (family_members * 32000) + 60000
    net_monthly_income = income - (annual_expenses / 12)
    ltv_ratio = (
        round(fin_amount / collateral_val, 4) if collateral_val > 0 else 2.0
    )
    guarantor_quality = (
        "A"
        if guarantor in ["ข้าราชการ", "เงินฝาก 100%"]
        else "B"
        if guarantor in ["โฉนดที่ดิน", "พนักงานเอกชน"]
        else "C"
    )

    credit_health_score = (
        100
        - (hist_late * 15)
        + (consec_saving * 0.5)
        + ((share_cap / fin_amount) * 20)
    )
    credit_health_score = float(np.clip(credit_health_score, 0, 100))

    profit_rate = 0.05
    monthly_installment = (
        fin_amount + (fin_amount * profit_rate * (fin_tenure / 12))
    ) / fin_tenure
    net_dti_ratio = (
        round(monthly_installment / net_monthly_income, 4)
        if net_monthly_income > 0
        else 1.5
    )

    input_data = pd.DataFrame(
        [[
            age,
            income,
            family_members,
            annual_expenses,
            net_monthly_income,
            occupation,
            fin_amount,
            fin_tenure,
            fin_type,
            distance,
            own_cap_ratio,
            collateral_val,
            ltv_ratio,
            guarantor,
            guarantor_quality,
            member_months,
            share_cap,
            hist_late,
            consec_saving,
            credit_health_score,
            net_dti_ratio,
        ]],
        columns=[
            "Age",
            "Monthly_Income",
            "Family_Members",
            "Annual_Expenses",
            "Net_Monthly_Income",
            "Occupation",
            "Financing_Amount",
            "Financing_Tenure_Months",
            "Financing_Type",
            "Business_Distance_km",
            "Own_Capital_Ratio",
            "Collateral_Value",
            "LTV_Ratio",
            "Guarantor_Type",
            "Guarantor_Quality",
            "Membership_Months",
            "Share_Capital",
            "Historical_Late_Payments",
            "Consecutive_Saving_Months",
            "Credit_Health_Score",
            "Net_DTI_Ratio",
        ],
    )

    df_passed, df_rejected = apply_hard_filters(input_data)

    if len(df_passed) == 0:
      reasons = [
          r for r in df_rejected.iloc[0]["Reject_Reasons"].split(" | ") if r
      ]
      st.session_state.assessment_result = {
          "status": "REJECTED_RULE",
          "reasons": reasons,
      }
    else:
      flags_str = df_passed.iloc[0]["Flags"]
      model_input = df_passed.drop(
          columns=["Is_Passed_Filter", "Reject_Reasons", "Flags"]
      )
      prob = float(pipeline.predict_proba(model_input)[0][1])

      st.session_state.assessment_result = {
          "status": "EVALUATED",
          "prob": prob,
          "flags": flags_str,
          "model_input": model_input,
          "metrics": {
              "age": age,
              "income": income,
              "family_members": family_members,
              "net_monthly_income": net_monthly_income,
              "occupation": occupation,
              "fin_amount": fin_amount,
              "fin_tenure": fin_tenure,
              "fin_type": fin_type,
              "guarantor": guarantor,
              "guarantor_quality": guarantor_quality,
              "share_cap": share_cap,
              "ltv_ratio": ltv_ratio,
              "net_dti_ratio": net_dti_ratio,
              "credit_health_score": credit_health_score,
              "monthly_installment": monthly_installment,
          },
      }
      # รีเซ็ตบันทึกเจ้าหน้าที่เมื่อประเมินเคสใหม่
      st.session_state.officer_record = {
          "decision": "สอดคล้องตามระบบประเมิน",
          "notes": "",
      }

  # แสดงผลลัพธ์จาก Session State
  res = st.session_state.assessment_result
  if res is not None:
    st.markdown("---")
    if res["status"] == "REJECTED_RULE":
      reasons_html = "".join([f"<li>{r}</li>" for r in res["reasons"]])
      st.markdown(
          f"""
            <div class="status-risk">
                <h4 style="margin:0 0 0.5rem 0; color:#991B1B;">ไม่ผ่านเกณฑ์คุณสมบัติเบื้องต้นตามระเบียบสหกรณ์</h4>
                <p style="margin:0 0 0.5rem 0;">ระบบยุติการส่งประเมินความเสี่ยงเนื่องจากไม่เข้าเงื่อนไขขั้นต่ำดังนี้:</p>
                <ul style="margin:0; padding-left:1.25rem;">{reasons_html}</ul>
            </div>
            """,
          unsafe_allow_html=True,
      )

    elif res["status"] == "EVALUATED":
      prob = res["prob"]
      m = res["metrics"]
      is_high_risk = prob >= optimal_threshold
      ai_decision_text = (
          "เสี่ยงสูง - ควรทบทวนเงื่อนไขหรือเพิ่มหลักประกัน"
          if is_high_risk
          else "ผ่านเกณฑ์มาตรฐาน - อนุมัติเบื้องต้น"
      )

      if res["flags"]:
        st.markdown(
            f"""
                <div class="status-warn">
                    <strong>ข้อสังเกตพิเศษประกอบการพิจารณา:</strong> {res['flags']}
                </div>
                """,
            unsafe_allow_html=True,
        )

      # แบ่งการแสดงผลออกเป็น 3 แท็บ
      tab1, tab2, tab3 = st.tabs([
          "สรุปผลการประเมินและเกณฑ์ภาระหนี้",
          "ปัจจัยประกอบการพิจารณาเชิงลึก",
          "บันทึกความเห็นเจ้าหน้าที่และรายงาน",
      ])

      # ------------------------------------------
      # แท็บที่ 1: สรุปผลและเกณฑ์ภาระหนี้
      # ------------------------------------------
      with tab1:
        if is_high_risk:
          st.markdown(
              f"""
                    <div class="status-risk">
                        <h4 style="margin:0 0 0.3rem 0; color:#991B1B;">ผลการประเมินความเสี่ยง: เสี่ยงสูง ({prob*100:.2f}%)</h4>
                        <p style="margin:0;">โอกาสเกิดหนี้เสียสูงกว่าเกณฑ์มาตรฐานสหกรณ์ที่กำหนดไว้ ({optimal_threshold*100:.2f}%) แนะนำให้พิจารณาทบทวนวงเงินหรือหลักประกัน</p>
                    </div>
                    """,
              unsafe_allow_html=True,
          )
        else:
          st.markdown(
              f"""
                    <div class="status-pass">
                        <h4 style="margin:0 0 0.3rem 0; color:#115E59;">ผลการประเมินความเสี่ยง: ผ่านเกณฑ์มาตรฐาน ({prob*100:.2f}%)</h4>
                        <p style="margin:0;">โอกาสเกิดหนี้เสียอยู่ในระดับที่สหกรณ์ยอมรับได้ (ต่ำกว่าเกณฑ์เฝ้าระวัง {optimal_threshold*100:.2f}%)</p>
                    </div>
                    """,
              unsafe_allow_html=True,
          )

        c1, c2 = st.columns(2)
        with c1:
          dti_pct = m["net_dti_ratio"] * 100
          bar_width = min(dti_pct, 100)
          if dti_pct <= 40:
            bar_color = "#0D9488"
            dti_label = "อยู่ในเกณฑ์ปลอดภัย (ไม่เกิน 40%)"
          elif dti_pct <= 70:
            bar_color = "#D97706"
            dti_label = "อยู่ในเกณฑ์เฝ้าระวัง (41% - 70%)"
          else:
            bar_color = "#DC2626"
            dti_label = "ภาระหนี้ตึงตัวสูง (เกิน 70%)"

          st.markdown(
              f"""
                    <div class="aic-card">
                        <div class="aic-card-title">สัดส่วนภาระหนี้ต่อรายได้สุทธิ (Net DTI)</div>
                        <div style="font-size: 1.75rem; font-weight: 700; color: {bar_color};">{dti_pct:.1f}%</div>
                        <div class="dti-bar-bg">
                            <div class="dti-bar-fill" style="width: {bar_width}%; background-color: {bar_color};"></div>
                        </div>
                        <div style="font-size: 0.9rem; color: #475569; margin-top: 0.4rem;">สถานะ: <strong>{dti_label}</strong></div>
                        <div style="font-size: 0.85rem; color: #64748B; margin-top: 0.2rem;">ค่างวดผ่อนชำระประมาณการ: {m['monthly_installment']:,.2f} บาท/เดือน</div>
                    </div>
                    """,
              unsafe_allow_html=True,
          )

        with c2:
          st.markdown(
              f"""
                    <div class="aic-card">
                        <div class="aic-card-title">ดัชนีความมั่นคงทางการเงินและหลักประกัน</div>
                        <table style="width:100%; font-size:0.92rem; border-collapse: collapse;">
                            <tr style="border-bottom: 1px solid #F1F5F9;">
                                <td style="padding: 6px 0; color:#64748B;">รายได้สุทธิหลังหักค่าใช้จ่ายครอบครัว</td>
                                <td style="padding: 6px 0; text-align:right; font-weight:600;">{m['net_monthly_income']:,.2f} บาท/เดือน</td>
                            </tr>
                            <tr style="border-bottom: 1px solid #F1F5F9;">
                                <td style="padding: 6px 0; color:#64748B;">สัดส่วนสินเชื่อต่อหลักประกัน (LTV)</td>
                                <td style="padding: 6px 0; text-align:right; font-weight:600;">{m['ltv_ratio']*100:.1f}% (ผู้ค้ำระดับ {m['guarantor_quality']})</td>
                            </tr>
                            <tr>
                                <td style="padding: 6px 0; color:#64748B;">คะแนนสุขภาพทางการเงิน</td>
                                <td style="padding: 6px 0; text-align:right; font-weight:600;">{m['credit_health_score']:.1f} / 100 คะแนน</td>
                            </tr>
                        </table>
                    </div>
                    """,
              unsafe_allow_html=True,
          )

      # ------------------------------------------
      # แท็บที่ 2: ปัจจัยประกอบการพิจารณาเชิงลึก (SHAP)
      # ------------------------------------------
      with tab2:
        st.markdown("##### ปัจจัยหลักที่ส่งผลต่อระดับความเสี่ยงของผู้ขอสินเชื่อรายนี้")
        try:
          preprocessor = pipeline.named_steps["preprocessor"]
          classifier = pipeline.named_steps["classifier"]

          input_transformed = preprocessor.transform(res["model_input"])
          if hasattr(input_transformed, "toarray"):
            input_transformed = input_transformed.toarray()

          num_cols = preprocessor.named_transformers_[
              "num"
          ].get_feature_names_out()
          cat_cols = preprocessor.named_transformers_[
              "cat"
          ].get_feature_names_out()
          feature_names = list(num_cols) + list(cat_cols)

          X_bg = df.drop("Target_NPF", axis=1).sample(50, random_state=42)
          bg_transformed = preprocessor.transform(X_bg)
          if hasattr(bg_transformed, "toarray"):
            bg_transformed = bg_transformed.toarray()

          explainer = shap.TreeExplainer(classifier, bg_transformed)
          shap_values = explainer(input_transformed)

          shap_to_plot = (
              shap_values[0, :, 1]
              if len(shap_values.shape) == 3
              else shap_values[0]
          )

          shap_df = pd.DataFrame(
              {"Feature": feature_names, "SHAP_Value": shap_to_plot.values}
          )

          thai_feature_names = {
              "num__Net_DTI_Ratio": "ภาระหนี้ต่อรายได้สุทธิ (Net DTI)",
              "num__Net_Monthly_Income": (
                  "รายได้สุทธิหลังหักค่าใช้จ่ายครอบครัว"
              ),
              "num__Credit_Health_Score": "คะแนนสุขภาพทางการเงิน",
              "num__LTV_Ratio": "สัดส่วนวงเงินต่อหลักประกัน (LTV)",
              "num__Historical_Late_Payments": "ประวัติค้างชำระในอดีต",
              "num__Consecutive_Saving_Months": "วินัยการออมต่อเนื่อง",
              "num__Share_Capital": "ทุนเรือนหุ้นสะสม",
              "num__Own_Capital_Ratio": "สัดส่วนทุนส่วนตัว",
              "num__Financing_Amount": "วงเงินสินเชื่อที่ขอ",
              "num__Financing_Tenure_Months": "ระยะเวลาผ่อนชำระ",
              "num__Monthly_Income": "รายได้รวมต่อเดือน",
              "num__Family_Members": "จำนวนสมาชิกในครอบครัว",
              "num__Age": "อายุผู้กู้",
              "num__Membership_Months": "อายุการเป็นสมาชิก",
          }

          shap_df["Feature_TH"] = shap_df["Feature"].apply(
              lambda x: thai_feature_names.get(
                  x, x.replace("cat__", "").replace("num__", "")
              )
          )

          risk_factors = (
              shap_df[shap_df["SHAP_Value"] > 0]
              .sort_values(by="SHAP_Value", ascending=False)
              .head(3)
          )
          good_factors = (
              shap_df[shap_df["SHAP_Value"] < 0]
              .sort_values(by="SHAP_Value", ascending=True)
              .head(3)
          )

          col_f1, col_f2 = st.columns(2)
          with col_f1:
            risk_items = "".join([
                f"<li><strong>{r['Feature_TH']}</strong> (เพิ่มน้ำหนักความเสี่ยง"
                f" {abs(r['SHAP_Value'])*100:.1f}%)</li>"
                for _, r in risk_factors.iterrows()
            ])
            if not risk_items:
              risk_items = "<li>ไม่พบปัจจัยที่เพิ่มความเสี่ยงอย่างมีนัยสำคัญ</li>"
            st.markdown(
                f"""
                        <div class="aic-card" style="border-top: 4px solid #DC2626;">
                            <div class="aic-card-title" style="color:#991B1B;">ปัจจัยที่เพิ่มความเสี่ยง</div>
                            <ul style="margin:0; padding-left:1.25rem; line-height:1.7;">{risk_items}</ul>
                        </div>
                        """,
                unsafe_allow_html=True,
            )

          with col_f2:
            good_items = "".join([
                f"<li><strong>{r['Feature_TH']}</strong> (ลดน้ำหนักความเสี่ยง"
                f" {abs(r['SHAP_Value'])*100:.1f}%)</li>"
                for _, r in good_factors.iterrows()
            ])
            if not good_items:
              good_items = "<li>ไม่พบปัจจัยสนับสนุนที่โดดเด่น</li>"
            st.markdown(
                f"""
                        <div class="aic-card" style="border-top: 4px solid #0D9488;">
                            <div class="aic-card-title" style="color:#115E59;">ปัจจัยที่ช่วยลดความเสี่ยง</div>
                            <ul style="margin:0; padding-left:1.25rem; line-height:1.7;">{good_items}</ul>
                        </div>
                        """,
                unsafe_allow_html=True,
            )

          with st.expander("ดูกราฟวิเคราะห์เชิงลึก (SHAP Waterfall Plot)"):
            shap_to_plot.feature_names = feature_names
            fig, ax = plt.subplots(figsize=(8, 4))
            shap.plots.waterfall(shap_to_plot, max_display=10, show=False)
            st.pyplot(fig)

        except Exception as e:
          st.error(f"ไม่สามารถแสดงผลการวิเคราะห์ปัจจัยได้: {e}")

      # ------------------------------------------
      # แท็บที่ 3: บันทึกความเห็นเจ้าหน้าที่และรายงาน
      # ------------------------------------------
      with tab3:
        st.markdown("##### การพิจารณาและบันทึกความเห็นเพิ่มเติมของเจ้าหน้าที่สินเชื่อ")
        with st.form("officer_form"):
          officer_decision = st.selectbox(
              "ความเห็นสรุปของเจ้าหน้าที่สินเชื่อ",
              [
                  "สอดคล้องตามระบบประเมิน",
                  "อนุมัติภายใต้เงื่อนไขพิเศษ (เจ้าหน้าที่รับรองข้อมูลเพิ่มเติม)",
                  "ปฏิเสธคำขอสินเชื่อ (พบความเสี่ยงเพิ่มเติมจากการตรวจสอบหน้างาน)",
              ],
          )
          officer_notes = st.text_area(
              "บันทึกเหตุผลและข้อมูลประกอบการตรวจสอบหน้างาน",
              value=st.session_state.officer_record["notes"],
              placeholder=(
                  "ระบุรายละเอียดเพิ่มเติม เช่น รายได้เสริม ทรัพย์สินค้ำประกัน"
                  " หรือวัตถุประสงค์การใช้เงินตามหลักชะรีอะฮ์..."
              ),
          )
          save_btn = st.form_submit_button("บันทึกความเห็นเจ้าหน้าที่")
          if save_btn:
            st.session_state.officer_record = {
                "decision": officer_decision,
                "notes": officer_notes,
            }
            st.success("บันทึกข้อมูลความเห็นเจ้าหน้าที่เรียบร้อยแล้ว")

        st.markdown("##### เอกสารสรุปผลการพิจารณาสินเชื่อ")
        saved_dec = st.session_state.officer_record["decision"]
        saved_notes = (
            st.session_state.officer_record["notes"]
            if st.session_state.officer_record["notes"]
            else "ไม่มีหมายเหตุเพิ่มเติม"
        )

        report_content = f"""====================================================================
เอกสารสรุปผลการประเมินความเสี่ยงสินเชื่อ (CREDIT ASSESSMENT REPORT)
สหกรณ์อิสลามอัศศิดดีก จำกัด (AIC)
====================================================================
[1. ข้อมูลผู้ขอสินเชื่อ]
- อายุ: {m['age']} ปี | อาชีพ: {m['occupation']} | สมาชิกครอบครัว: {m['family_members']} คน
- รายได้รวม: {m['income']:,.2f} บาท/เดือน | รายได้สุทธิ: {m['net_monthly_income']:,.2f} บาท/เดือน
- วงเงินที่ขอ: {m['fin_amount']:,.2f} บาท (ระยะเวลา {m['fin_tenure']} งวด)
- ประเภทสินเชื่อ: {m['fin_type']} | หลักประกัน: {m['guarantor']} (เกรด {m['guarantor_quality']})
- ทุนเรือนหุ้นสะสม: {m['share_cap']:,.2f} บาท
- สัดส่วน LTV: {m['ltv_ratio']*100:.1f}% | ภาระหนี้สุทธิ Net DTI: {m['net_dti_ratio']*100:.1f}%

[2. ผลการวิเคราะห์จากระบบประเมินความเสี่ยง]
- เกณฑ์คุณสมบัติเบื้องต้น 4 ข้อ: ผ่านเกณฑ์ตามระเบียบสหกรณ์
- โอกาสเกิดหนี้เสีย (NPF Probability): {prob*100:.2f}%
- เกณฑ์มาตรฐานเฝ้าระวังของสหกรณ์: {optimal_threshold*100:.2f}%
- คำแนะนำจากระบบ: {ai_decision_text}

[3. ความเห็นเพิ่มเติมของเจ้าหน้าที่สินเชื่อ]
- ผลการพิจารณา: {saved_dec}
- บันทึกเพิ่มเติม: {saved_notes}
===================================================================="""

        st.text_area(
            "ตัวอย่างเอกสารรายงาน",
            value=report_content,
            height=260,
            label_visibility="collapsed",
        )
        st.download_button(
            label="ดาวน์โหลดรายงานสรุปผลการพิจารณา (.txt)",
            data=report_content,
            file_name=f"AIC_Credit_Report_{m['income']}_{m['fin_amount']}.txt",
            mime="text/plain",
        )

# ==========================================
# หน้าต่างที่ 2: ภาพรวมพอร์ตสินเชื่อสหกรณ์
# ==========================================
elif menu == "ภาพรวมพอร์ตสินเชื่อสหกรณ์":
  st.markdown("#### สถิติภาพรวมพอร์ตสินเชื่อและอัตราหนี้เสีย")

  c1, c2, c3 = st.columns(3)
  with c1:
    st.markdown(
        f"""
        <div class="aic-card">
            <div style="color:#64748B; font-size:0.9rem;">จำนวนบัญชีสินเชื่อในฐานข้อมูล</div>
            <div style="color:#0B2545; font-size:1.8rem; font-weight:700; margin-top:0.25rem;">{len(df):,} ราย</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
  with c2:
    st.markdown(
        f"""
        <div class="aic-card">
            <div style="color:#64748B; font-size:0.9rem;">อัตราส่วนหนี้ที่ไม่ก่อให้เกิดรายได้ (NPF)</div>
            <div style="color:#DC2626; font-size:1.8rem; font-weight:700; margin-top:0.25rem;">{(df['Target_NPF'].mean() * 100):.1f}%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
  with c3:
    st.markdown(
        f"""
        <div class="aic-card">
            <div style="color:#64748B; font-size:0.9rem;">มูลค่าสินเชื่อรวมทั้งพอร์ต</div>
            <div style="color:#0D9488; font-size:1.8rem; font-weight:700; margin-top:0.25rem;">{df['Financing_Amount'].sum():,.0f} บาท</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

  col_chart1, col_chart2 = st.columns(2)
  with col_chart1:
    st.markdown('<div class="aic-card">', unsafe_allow_html=True)
    st.markdown(
        '<div class="aic-card-title">จำนวนบัญชีหนี้เสียจำแนกตามกลุ่มอาชีพ</div>',
        unsafe_allow_html=True,
    )
    npf_by_occ = df[df["Target_NPF"] == 1]["Occupation"].value_counts()
    st.bar_chart(npf_by_occ, color="#0B2545")
    st.markdown("</div>", unsafe_allow_html=True)

  with col_chart2:
    st.markdown('<div class="aic-card">', unsafe_allow_html=True)
    st.markdown(
        '<div class="aic-card-title">การกระจายตัวของภาระหนี้ต่อรายได้สุทธิ (Net'
        " DTI)</div>",
        unsafe_allow_html=True,
    )
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.hist(
        df[df["Target_NPF"] == 0]["Net_DTI_Ratio"].clip(0, 2),
        bins=20,
        alpha=0.65,
        label="Normal",
        color="#0D9488",
    )
    ax.hist(
        df[df["Target_NPF"] == 1]["Net_DTI_Ratio"].clip(0, 2),
        bins=20,
        alpha=0.65,
        label="NPF (Default)",
        color="#DC2626",
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper right")
    st.pyplot(fig)
    st.markdown("</div>", unsafe_allow_html=True)