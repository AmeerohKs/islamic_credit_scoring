import sys
import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# เพิ่ม Root Directory เข้าไปใน sys.path เพื่อให้ import โมดูลจาก src/ ได้
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from src.pre_screening import apply_hard_filters

# 1. ตั้งค่าหน้าเว็บ
st.set_page_config(page_title="Islamic Credit Scoring", layout="wide", page_icon="🏦")

# 2. โหลดโมเดลและข้อมูลเวอร์ชันล่าสุด
@st.cache_resource
def load_model():
    model_path = os.path.join(BASE_DIR, 'models', 'best_credit_model.pkl')
    return joblib.load(model_path)

@st.cache_data
def load_data():
    data_path = os.path.join(BASE_DIR, 'data', 'raw', 'islamic_coop_synthetic_data_v2.csv')
    return pd.read_csv(data_path)

model_package = load_model()
pipeline = model_package['pipeline']
optimal_threshold = model_package['optimal_threshold']
df = load_data()

# 3. เมนูด้านข้าง (Sidebar)
st.sidebar.title("เมนูการทำงาน")
menu = st.sidebar.radio("เลือกหน้าต่าง:", ["📝 ประเมินสินเชื่อรายบุคคล", "📊 แดชบอร์ดภาพรวม"])

# ==========================================
# หน้าต่างที่ 1: ระบบประเมินสินเชื่อ
# ==========================================
if menu == "📝 ประเมินสินเชื่อรายบุคคล":
    st.title("ระบบประเมินความเสี่ยงสินเชื่อ (Islamic Credit Scoring)")
    st.markdown(f"**เกณฑ์การพิจารณา AI ปัจจุบัน:** เตือนความเสี่ยงเมื่อโอกาสเป็นหนี้เสีย ≥ **{optimal_threshold*100:.2f}%** (ผ่านการคัดกรองเบื้องต้นตามระเบียบสหกรณ์)")

    with st.form("credit_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = st.number_input("อายุ (ปี)", 18, 75, 35)
            income = st.number_input("รายได้ต่อเดือน (บาท)", 7000, 200000, 25000)
            occupation = st.selectbox("อาชีพ", ['ข้าราชการ/รัฐวิสาหกิจ', 'พนักงานเอกชน', 'เจ้าของธุรกิจ/ค้าขาย', 'อาชีพอิสระ/เกษตรกร'])
            guarantor = st.selectbox("ประเภทหลักประกัน", ['บุคคลทั่วไป', 'ข้าราชการ/รัฐวิสาหกิจ', 'โฉนดที่ดิน', 'เงินฝากค้ำประกัน 100%'])
            
        with col2:
            fin_amount = st.number_input("วงเงินสินเชื่อที่ขอ (บาท)", 10000, 500000, 100000)
            fin_tenure = st.selectbox("ระยะเวลาผ่อน (งวด/เดือน)", [12, 24, 36, 48, 60, 72, 84, 96, 108, 120])
            fin_type = st.selectbox("ประเภทสินเชื่อ", ['Personal', 'Business'])
            distance = st.number_input("ระยะห่างจากสหกรณ์ (กม.)", 1, 30, 10)
            
        with col3:
            member_months = st.number_input("อายุสมาชิก (เดือน)", 1, 120, 24)
            share_cap = st.number_input("ทุนเรือนหุ้น (บาท)", 1000, 200000, 10000)
            own_cap_ratio = st.slider("สัดส่วนทุนส่วนตัว (%)", 0, 50, 10) / 100.0
            hist_late = st.number_input("ประวัติจ่ายล่าช้า (ครั้ง)", 0, 5, 0)
            
        submitted = st.form_submit_button("ประเมินความเสี่ยง")

    if submitted:
        profit_rate = 0.05
        total_payment = fin_amount + (fin_amount * profit_rate * (fin_tenure / 12))
        monthly_installment = total_payment / fin_tenure
        dti_ratio = round(monthly_installment / income, 2)
        
        input_data = pd.DataFrame([[
            age, income, occupation, member_months, share_cap, fin_amount, 
            fin_tenure, fin_type, distance, own_cap_ratio, guarantor, 
            hist_late, dti_ratio
        ]], columns=[
            'Age', 'Monthly_Income', 'Occupation', 'Membership_Months', 
            'Share_Capital', 'Financing_Amount', 'Financing_Tenure_Months', 
            'Financing_Type', 'Business_Distance_km', 'Own_Capital_Ratio', 
            'Guarantor_Type', 'Historical_Late_Payments', 'Debt_to_Income_Ratio'
        ])
        
        st.divider()
        
        # ขั้นตอนที่ 1: ตรวจสอบด้วยกฎเหล็ก (Hard Filter) จาก src/pre_screening.py
        df_passed, _ = apply_hard_filters(input_data)
        
        if len(df_passed) == 0:
            st.error("🚫 **ไม่ผ่านเกณฑ์เบื้องต้น (Hard Filter Rejected)**")
            st.warning("ผู้ขอสินเชื่อไม่ผ่านข้อกำหนดพื้นฐานของสหกรณ์ ระบบจึงยุติการส่งประเมินต่อให้ AI:")
            if age < 20 or age > 60:
                st.write(f"- **อายุ ({age} ปี):** อยู่นอกเกณฑ์ที่สหกรณ์กำหนด (ต้องมีอายุ 20 - 60 ปี)")
            if dti_ratio > 0.70:
                st.write(f"- **ภาระหนี้ต่อรายได้ DTI ({dti_ratio*100:.1f}%):** สูงเกินเพดานความเสี่ยงสูงสุดที่สหกรณ์รับได้ (ไม่เกิน 70%)")
        else:
            # ขั้นตอนที่ 2: ผ่านเกณฑ์เบื้องต้น ส่งให้ AI ประเมินความเสี่ยง
            prob = pipeline.predict_proba(input_data)[0][1]
            
            st.subheader("ผลการประเมินจากระบบ AI")
            col_res1, col_res2 = st.columns(2)
            with col_res1:
                st.metric("ภาระหนี้ต่อรายได้ (DTI)", f"{dti_ratio*100:.1f} %")
                
                if prob >= optimal_threshold:
                    st.error(f"⚠️ เสี่ยงสูง: โอกาสเป็นหนี้เสีย {prob*100:.2f}%")
                    ai_decision = "ปฏิเสธ / ควรทบทวนความเสี่ยงอย่างเข้มงวด"
                else:
                    st.success(f"✅ ผ่านเกณฑ์: โอกาสเป็นหนี้เสีย {prob*100:.2f}%")
                    ai_decision = "อนุมัติเบื้องต้นตามเกณฑ์"
                    
            with col_res2:
                st.info("💡 ข้อแนะนำจาก AI:")
                has_advice = False
                if dti_ratio > 0.4:
                    st.write("- ภาระหนี้ (DTI) สูงเกิน 40% แนะนำให้ลดยอดกู้หรือเพิ่มงวดผ่อน")
                    has_advice = True
                if hist_late > 0:
                    st.write("- มีประวัติชำระล่าช้า ควรตรวจสอบวินัยการเงินเพิ่มเติม")
                    has_advice = True
                if share_cap / fin_amount < 0.1:
                    st.write("- ทุนเรือนหุ้นต่ำกว่า 10% ของวงเงินกู้ แนะนำให้ซื้อหุ้นเพิ่ม")
                    has_advice = True
                if not has_advice:
                    st.write("- คุณสมบัติทางการเงินอยู่ในเกณฑ์ดี ไม่มีข้อกังวลเพิ่มเติม")

            # ช่องความเห็นเจ้าหน้าที่ (Human Override)
            st.divider()
            st.subheader("🧑‍💼 การพิจารณาและบันทึกความเห็นของเจ้าหน้าที่สินเชื่อ")
            
            with st.form("override_form"):
                officer_decision = st.selectbox(
                    "ความเห็นสรุปของเจ้าหน้าที่ (ตัดสินใจขั้นสุดท้าย):",
                    ["สอดคล้องตาม AI (เห็นชอบตามระบบ)", "อนุมัติพิเศษ (Override อนุมัติ แม้ AI เตือนเสี่ยงสูง)", "ปฏิเสธพิเศษ (Override ปฏิเสธ แม้ AI ผ่านเกณฑ์)"]
                )
                officer_notes = st.text_area(
                    "บันทึกเหตุผล / หมายเหตุเพิ่มเติมจากเจ้าหน้าที่หน้างาน:",
                    placeholder="เช่น ลูกค้ามีรายได้เสริมจากการทำสวนยางพาราเพิ่มเติม..."
                )
                save_override = st.form_submit_button("บันทึกความเห็น")
                if save_override:
                    st.success("บันทึกความเห็นเข้าสู่ระบบเรียบร้อยแล้ว!")

            # การอธิบายผลด้วย SHAP Value
            st.divider()
            st.subheader("🔍 วิเคราะห์ปัจจัยที่ส่งผลต่อคะแนน (SHAP Value)")
            
            try:
                preprocessor = pipeline.named_steps['preprocessor']
                classifier = pipeline.named_steps['classifier']
                
                input_transformed = preprocessor.transform(input_data)
                if hasattr(input_transformed, 'toarray'):
                    input_transformed = input_transformed.toarray()
                    
                num_cols = preprocessor.named_transformers_['num'].get_feature_names_out()
                cat_cols = preprocessor.named_transformers_['cat'].get_feature_names_out()
                feature_names = list(num_cols) + list(cat_cols)
                
                X_bg = df.drop('Target_NPF', axis=1).sample(50, random_state=42)
                bg_transformed = preprocessor.transform(X_bg)
                if hasattr(bg_transformed, 'toarray'):
                    bg_transformed = bg_transformed.toarray()
                    
                explainer = shap.TreeExplainer(classifier, bg_transformed)
                shap_values = explainer(input_transformed)
                
                if len(shap_values.shape) == 3:
                    shap_to_plot = shap_values[0, :, 1]
                else:
                    shap_to_plot = shap_values[0]

                shap_to_plot.feature_names = feature_names
                
                fig, ax = plt.subplots(figsize=(8, 4))
                shap.plots.waterfall(shap_to_plot, max_display=10, show=False)
                st.pyplot(fig)

            except Exception as e:
                st.error(f"ระบบ SHAP ขัดข้องชั่วคราว: {e}")

            # ปุ่มพิมพ์รายงาน (Print Report / Export)
            st.divider()
            st.subheader("📄 รายงานสรุปผลการพิจารณา (พร้อมพิมพ์)")
            
            report_content = f"""
==================================================
รายงานสรุปการพิจารณาสินเชื่อ (CREDIT ASSESSMENT REPORT)
ระบบประเมินความเสี่ยงสินเชื่อสหกรณ์อิสลาม
==================================================
[ข้อมูลผู้ขอสินเชื่อ]
- อายุ: {age} ปี | รายได้: {income:,.2f} บาท/เดือน
- อาชีพ: {occupation}
- วงเงินสินเชื่อที่ขอ: {fin_amount:,.2f} บาท (ระยะเวลา {fin_tenure} งวด)
- ประเภทสินเชื่อ: {fin_type} | หลักประกัน: {guarantor}
- ทุนเรือนหุ้น: {share_cap:,.2f} บาท | ภาระหนี้ DTI: {dti_ratio*100:.1f}%

[ผลการวิเคราะห์จากระบบ AI]
- สถานะกฎเหล็กเบื้องต้น (Hard Filter): ผ่านเกณฑ์
- โอกาสเกิดหนี้เสีย (NPF Probability): {prob*100:.2f}%
- เกณฑ์มาตรฐานของสหกรณ์: ตัดสินความเสี่ยงที่ {optimal_threshold*100:.2f}%
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
                mime="text/plain"
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
        npf_by_occ = df[df['Target_NPF'] == 1]['Occupation'].value_counts()
        st.bar_chart(npf_by_occ)
        
    with col_chart2:
        st.subheader("การกระจายตัวของภาระหนี้ (DTI)")
        fig, ax = plt.subplots()
        ax.hist(df[df['Target_NPF']==0]['Debt_to_Income_Ratio'], bins=20, alpha=0.5, label='Normal')
        ax.hist(df[df['Target_NPF']==1]['Debt_to_Income_Ratio'], bins=20, alpha=0.5, label='NPF (Default)', color='red')
        ax.legend(loc='upper right')
        st.pyplot(fig)