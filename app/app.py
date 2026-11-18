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

st.set_page_config(
    page_title="Islamic Credit Scoring", layout="wide", page_icon="🏦"
)


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

st.sidebar.title("เมนูการทำงาน")
menu = st.sidebar.radio(
    "เลือกหน้าต่าง:", ["📝 ประเมินสินเชื่อรายบุคคล", "📊 แดชบอร์ดภาพรวม"]
)

# ==========================================
# หน้าต่างที่ 1: ระบบประเมินสินเชื่อ
# ==========================================
if menu == "📝 ประเมินสินเชื่อรายบุคคล":
  st.title("ระบบประเมินความเสี่ยงสินเชื่อ (Islamic Credit Scoring)")
  st.markdown(
      "**เกณฑ์การพิจารณา AI ปัจจุบัน:** เตือนความเสี่ยงเมื่อโอกาสเป็นหนี้เสีย"
      f" ≥ **{optimal_threshold*100:.2f}%** (ผ่านเกณฑ์คัดกรองเบื้องต้น 4"
      " ข้อตามระเบียบสหกรณ์)"
  )

  with st.form("credit_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
      age = st.number_input("อายุ (ปี)", 18, 75, 35)
      income = st.number_input("รายได้ต่อเดือน (บาท)", 7000, 300000, 30000)
      family_members = st.number_input("จำนวนสมาชิกในครอบครัว (คน)", 1, 15, 3)
      occupation = st.selectbox(
          "อาชีพ",
          [
              "ข้าราชการ/รัฐวิสาหกิจ",
              "พนักงานเอกชน",
              "เจ้าของธุรกิจ/ค้าขาย",
              "อาชีพอิสระ/เกษตรกร",
          ],
      )
      guarantor = st.selectbox(
          "ประเภทหลักประกัน",
          [
              "ข้าราชการ",
              "เงินฝาก 100%",
              "โฉนดที่ดิน",
              "พนักงานเอกชน",
              "บุคคลทั่วไป",
          ],
      )

    with col2:
      fin_amount = st.number_input(
          "วงเงินสินเชื่อที่ขอ (บาท)", 10000, 3000000, 150000
      )
      fin_tenure = st.selectbox(
          "ระยะเวลาผ่อน (งวด/เดือน)", [12, 24, 36, 48, 60, 84, 120, 240]
      )
      fin_type = st.selectbox("ประเภทสินเชื่อ", ["Personal", "Business"])
      collateral_val = st.number_input(
          "มูลค่าหลักประกันประเมิน (บาท)", 0, 5000000, 200000
      )
      distance = st.number_input("ระยะห่างจากสหกรณ์ (กม.)", 1, 50, 10)

    with col3:
      member_months = st.number_input("อายุสมาชิก (เดือน)", 1, 240, 24)
      share_cap = st.number_input("ทุนเรือนหุ้นสะสม (บาท)", 0, 1000000, 30000)
      consec_saving = st.number_input("ออมเงินต่อเนื่อง (เดือน)", 0, 120, 12)
      own_cap_ratio = st.slider("สัดส่วนทุนส่วนตัว (%)", 0, 50, 15) / 100.0
      hist_late = st.number_input("ประวัติจ่ายล่าช้า (ครั้ง)", 0, 10, 0)

    submitted = st.form_submit_button("ประเมินความเสี่ยง")

  if submitted:
    # คำนวณตัวแปรเชิงลึกให้ตรงกับโครงสร้างข้อมูล V2
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

    st.divider()

    # ขั้นตอนที่ 1: ตรวจสอบกฎเหล็ก 4 ข้อ (Hard Filter)
    df_passed, df_rejected = apply_hard_filters(input_data)

    if len(df_passed) == 0:
      st.error("🚫 **ไม่ผ่านเกณฑ์เบื้องต้น (Hard Filter Rejected)**")
      st.warning(
          "ผู้ขอสินเชื่อไม่ผ่านข้อกำหนดพื้นฐานของสหกรณ์"
          " ระบบจึงยุติการส่งประเมินต่อให้ AI:"
      )
      reasons = df_rejected.iloc[0]["Reject_Reasons"].split(" | ")
      for r in reasons:
        if r:
          st.write(f"- **{r}**")
    else:
      # แสดง Flag แจ้งเตือน (ถ้ามีประวัติค้างชำระ)
      flags_str = df_passed.iloc[0]["Flags"]
      if flags_str:
        st.warning(f"⚠️ **ข้อสังเกตพิเศษ (Flagged):** {flags_str}")

      # ตัดคอลัมน์ช่วยคัดกรองออกก่อนส่งเข้า AI Pipeline
      model_input = df_passed.drop(
          columns=["Is_Passed_Filter", "Reject_Reasons", "Flags"]
      )
      prob = pipeline.predict_proba(model_input)[0][1]

      st.subheader("ผลการประเมินจากระบบ AI")
      col_res1, col_res2 = st.columns(2)
      with col_res1:
        dti_val = net_dti_ratio * 100
        if dti_val <= 40:
          dti_status = "🟢 ปลอดภัย (Net DTI ต่ำกว่า 40%)"
        elif dti_val <= 70:
          dti_status = "🟡 ปานกลาง / เฝ้าระวัง (Net DTI ระหว่าง 41% - 70%)"
        else:
          dti_status = "🔴 เสี่ยงสูง / ภาระหนี้ตึงตัว (Net DTI สูงกว่า 70%)"

        st.metric("ภาระหนี้ต่อรายได้สุทธิ (Net DTI)", f"{dti_val:.1f} %")
        st.markdown(f"**สถานะเกณฑ์ Net DTI:** {dti_status}")

        if prob >= optimal_threshold:
          st.error(f"⚠️ เสี่ยงสูง: โอกาสเป็นหนี้เสีย {prob*100:.2f}%")
          ai_decision = "ปฏิเสธ / ควรทบทวนความเสี่ยงอย่างเข้มงวด"
        else:
          st.success(f"✅ ผ่านเกณฑ์: โอกาสเป็นหนี้เสีย {prob*100:.2f}%")
          ai_decision = "อนุมัติเบื้องต้นตามเกณฑ์"

      with col_res2:
        st.info("💡 สรุปข้อมูลทางการเงินเชิงลึก:")
        st.write(f"- **รายได้สุทธิหลังหักค่าใช้จ่ายครอบครัว:** {net_monthly_income:,.0f} บาท/เดือน")
        st.write(
            f"- **อัตราส่วนสินเชื่อต่อหลักประกัน (LTV):** {ltv_ratio*100:.1f}%"
            f" (คุณภาพผู้ค้ำระดับ **{guarantor_quality}**)"
        )
        st.write(f"- **คะแนนสุขภาพการเงิน (Credit Health):** {credit_health_score:.1f} / 100")

      # ช่องความเห็นเจ้าหน้าที่ (Human Override)
      st.divider()
      st.subheader("🧑‍💼 การพิจารณาและบันทึกความเห็นของเจ้าหน้าที่สินเชื่อ")

      with st.form("override_form"):
        officer_decision = st.selectbox(
            "ความเห็นสรุปของเจ้าหน้าที่ (ตัดสินใจขั้นสุดท้าย):",
            [
                "สอดคล้องตาม AI (เห็นชอบตามระบบ)",
                "อนุมัติพิเศษ (Override อนุมัติ แม้ AI เตือนเสี่ยงสูง)",
                "ปฏิเสธพิเศษ (Override ปฏิเสธ แม้ AI ผ่านเกณฑ์)",
            ],
        )
        officer_notes = st.text_area(
            "บันทึกเหตุผล / หมายเหตุเพิ่มเติมจากเจ้าหน้าที่หน้างาน:",
            placeholder="เช่น ลูกค้ามีรายได้เสริมจากการทำสวนยางพาราเพิ่มเติม...",
        )
        save_override = st.form_submit_button("บันทึกความเห็น")
        if save_override:
          st.success("บันทึกความเห็นเข้าสู่ระบบเรียบร้อยแล้ว!")

      # การอธิบายผลด้วย SHAP Value (ฉบับแปลภาษาไทย)
      st.divider()
      st.subheader("🔍 วิเคราะห์ปัจจัยที่ส่งผลต่อคะแนน (ฉบับเข้าใจง่าย)")

      try:
        preprocessor = pipeline.named_steps["preprocessor"]
        classifier = pipeline.named_steps["classifier"]

        input_transformed = preprocessor.transform(model_input)
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
            shap_values[0, :, 1] if len(shap_values.shape) == 3 else shap_values[0]
        )

        shap_df = pd.DataFrame(
            {"Feature": feature_names, "SHAP_Value": shap_to_plot.values}
        )

        thai_feature_names = {
            "num__Net_DTI_Ratio": "ภาระหนี้ต่อรายได้สุทธิ (Net DTI)",
            "num__Net_Monthly_Income": "รายได้สุทธิหลังหักค่าใช้จ่ายครอบครัว",
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

        col_red, col_green = st.columns(2)
        with col_red:
          st.error("🔴 **ปัจจัยหลักที่เพิ่มความเสี่ยง**")
          if not risk_factors.empty:
            for _, row in risk_factors.iterrows():
              st.write(
                  f"- **{row['Feature_TH']}** (เพิ่มความเสี่ยง"
                  f" {abs(row['SHAP_Value'])*100:.1f}%)"
              )
          else:
            st.write("- ไม่มีปัจจัยความเสี่ยงที่น่ากังวล")

        with col_green:
          st.success("🟢 **ปัจจัยหลักที่ช่วยลดความเสี่ยง**")
          if not good_factors.empty:
            for _, row in good_factors.iterrows():
              st.write(
                  f"- **{row['Feature_TH']}** (ลดความเสี่ยง"
                  f" {abs(row['SHAP_Value'])*100:.1f}%)"
              )
          else:
            st.write("- ไม่มีปัจจัยสนับสนุนที่โดดเด่น")

        with st.expander("ดูกราฟวิเคราะห์เชิงลึก (SHAP Waterfall Plot)"):
          shap_to_plot.feature_names = feature_names
          fig, ax = plt.subplots(figsize=(8, 4))
          shap.plots.waterfall(shap_to_plot, max_display=10, show=False)
          st.pyplot(fig)

      except Exception as e:
        st.error(f"ระบบวิเคราะห์ปัจจัยขัดข้องชั่วคราว: {e}")

      # รายงานสรุปผลการพิจารณา
      st.divider()
      st.subheader("📄 รายงานสรุปผลการพิจารณา (พร้อมพิมพ์)")

      report_content = f"""
==================================================
รายงานสรุปการพิจารณาสินเชื่อ (CREDIT ASSESSMENT REPORT)
ระบบประเมินความเสี่ยงสินเชื่อสหกรณ์อิสลาม
==================================================
[ข้อมูลผู้ขอสินเชื่อ]
- อายุ: {age} ปี | รายได้รวม: {income:,.2f} บาท | รายได้สุทธิ: {net_monthly_income:,.2f} บาท
- อาชีพ: {occupation} | สมาชิกครอบครัว: {family_members} คน
- วงเงินสินเชื่อที่ขอ: {fin_amount:,.2f} บาท (ระยะเวลา {fin_tenure} งวด)
- ประเภทสินเชื่อ: {fin_type} | หลักประกัน: {guarantor} (เกรด {guarantor_quality})
- ทุนเรือนหุ้น: {share_cap:,.2f} บาท | LTV: {ltv_ratio*100:.1f}% | Net DTI: {net_dti_ratio*100:.1f}%

[ผลการวิเคราะห์จากระบบ AI]
- สถานะกฎเหล็ก 4 ข้อ (Hard Filter): ผ่านเกณฑ์
- โอกาสเกิดหนี้เสีย (NPF Probability): {prob*100:.2f}%
- เกณฑ์ตัดสินความเสี่ยงของสหกรณ์: {optimal_threshold*100:.2f}%
- สถานะ AI: {ai_decision}

[บันทึกความเห็นเจ้าหน้าที่สินเชื่อ]
- ผลการตัดสินใจ: {officer_decision}
- หมายเหตุเพิ่มเติม: {officer_notes if 'officer_notes' in locals() and officer_notes else 'ไม่มีหมายเหตุพิเศษ'}
==================================================
"""
      st.text_area("ตัวอย่างข้อความรายงาน:", value=report_content, height=250)
      st.download_button(
          label="📥 ดาวน์โหลดรายงานสรุปสินเชื่อ (Text Report)",
          data=report_content,
          file_name=f"Credit_Report_{income}_{fin_amount}.txt",
          mime="text/plain",
      )

# ==========================================
# หน้าต่างที่ 2: แดชบอร์ดภาพรวม
# ==========================================
elif menu == "📊 แดชบอร์ดภาพรวม":
  st.title("แดชบอร์ดสถิติภาพรวมสหกรณ์ (Portfolio Dashboard)")

  col1, col2, col3 = st.columns(3)
  col1.metric("จำนวนลูกหนี้ทั้งหมด", f"{len(df):,} ราย")
  col2.metric("สัดส่วนหนี้เสีย (NPF)", f"{(df['Target_NPF'].mean() * 100):.1f} %")
  col3.metric("วงเงินสินเชื่อรวม", f"{df['Financing_Amount'].sum():,.0f} บาท")

  st.divider()

  col_chart1, col_chart2 = st.columns(2)
  with col_chart1:
    st.subheader("จำนวนหนี้เสียจำแนกตามอาชีพ")
    npf_by_occ = df[df["Target_NPF"] == 1]["Occupation"].value_counts()
    st.bar_chart(npf_by_occ)

  with col_chart2:
    st.subheader("การกระจายตัวของภาระหนี้สุทธิ (Net DTI)")
    fig, ax = plt.subplots()
    ax.hist(
        df[df["Target_NPF"] == 0]["Net_DTI_Ratio"].clip(0, 2),
        bins=20,
        alpha=0.5,
        label="Normal",
    )
    ax.hist(
        df[df["Target_NPF"] == 1]["Net_DTI_Ratio"].clip(0, 2),
        bins=20,
        alpha=0.5,
        label="NPF (Default)",
        color="red",
    )
    ax.legend(loc="upper right")
    st.pyplot(fig)