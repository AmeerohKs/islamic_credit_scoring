import pandas as pd
import numpy as np
import random

# กำหนด Seed เพื่อให้สุ่มออกมากี่ครั้งก็ได้ผลเหมือนเดิม (Reproducibility)
np.random.seed(42)
random.seed(42)

# 1. กำหนดขนาดข้อมูล
TOTAL_ROWS = 10000
NEW_CUST_ROWS = int(TOTAL_ROWS * 0.3) # 30% ลูกค้าใหม่
OLD_CUST_ROWS = TOTAL_ROWS - NEW_CUST_ROWS # 70% ลูกค้าเก่า

# 2. สร้างโครงสร้างข้อมูล
data = []

# ฟังก์ชันสุ่มอาชีพและรายได้ให้สมเหตุสมผล
def generate_income_by_occupation():
    occ = random.choice(['ข้าราชการ/รัฐวิสาหกิจ', 'พนักงานเอกชน', 'เจ้าของธุรกิจ/ค้าขาย', 'อาชีพอิสระ/เกษตรกร'])
    if occ == 'ข้าราชการ/รัฐวิสาหกิจ':
        income = np.random.randint(15000, 80000)
    elif occ == 'พนักงานเอกชน':
        income = np.random.randint(12000, 100000)
    elif occ == 'เจ้าของธุรกิจ/ค้าขาย':
        income = np.random.randint(15000, 150000)
    else:
        income = np.random.randint(7000, 40000)
    return occ, income

# สร้างข้อมูลลูกค้า (รวมเก่าและใหม่)
for i in range(TOTAL_ROWS):
    age = np.random.randint(20, 66)
    occ, income = generate_income_by_occupation()
    fin_amount = np.random.randint(10000, 500001)
    tenure_months = random.choice([12, 24, 36, 48, 60, 72, 84, 96, 108, 120])
    fin_type = random.choice(['Personal', 'Business'])
    distance = np.random.randint(1, 31)
    own_cap = np.random.randint(0, 51) / 100.0 # 0.00 ถึง 0.50
    guarantor = random.choice(['บุคคลทั่วไป', 'ข้าราชการ/รัฐวิสาหกิจ', 'โฉนดที่ดิน', 'เงินฝากค้ำประกัน 100%'])
    
    # เงื่อนไขลูกค้าใหม่ vs ลูกค้าเก่า
    if i < NEW_CUST_ROWS: # ลูกค้าใหม่
        hist_late = 0
        member_months = np.random.randint(1, 13) # เพิ่งเป็นสมาชิกไม่เกิน 1 ปี
        share_cap = np.random.randint(1000, 10000) # หุ้นยังน้อย
    else: # ลูกค้าเก่า
        hist_late = np.random.randint(0, 6)
        member_months = np.random.randint(13, 121)
        share_cap = np.random.randint(5000, 200001)

    # 3. Feature Engineering: คำนวณ DTI
    # สมมติการคิดยอดผ่อนแบบง่าย: (เงินต้น + กำไรคงที่สมมติ 5% ต่อปี) / จำนวนงวด
    profit_rate = 0.05
    total_payment = fin_amount + (fin_amount * profit_rate * (tenure_months / 12))
    monthly_installment = total_payment / tenure_months
    dti_ratio = round(monthly_installment / income, 2)

    # 4. คำนวณความน่าจะเป็นของการเป็นหนี้เสีย (NPF Probability) จาก 4 ปัจจัย
    npf_prob = 0.05 # ฐานความเสี่ยงตั้งต้น 5%
    
    # ปัจจัย 1: DTI สูงเกิน 40% (0.4)
    if dti_ratio > 0.4:
        npf_prob += 0.25
        
    # ปัจจัย 2: ประวัติจ่ายช้า
    if hist_late >= 2:
        npf_prob += 0.30
        
    # ปัจจัย 3: ทุนเรือนหุ้นต่ำมากเมื่อเทียบกับวงเงินกู้ (< 10%)
    if share_cap / fin_amount < 0.10:
        npf_prob += 0.15
        
    # ปัจจัย 4: ความเสี่ยงทางธุรกิจ (ทุนตัวเองน้อย + อยู่ไกล)
    if fin_type == 'Business' and own_cap < 0.30 and distance > 15:
        npf_prob += 0.20

    # สุ่มผลลัพธ์ Target NPF จากความน่าจะเป็น
    # (เพิ่มเงื่อนไขควบคุมให้ภาพรวมหนี้เสียอยู่ที่ราวๆ 10% ตามที่กำหนด)
    target_npf = np.random.choice([0, 1], p=[max(0, 1-npf_prob), min(1, npf_prob)])

    # เก็บข้อมูลลง List
    data.append([
        age, income, occ, member_months, share_cap, fin_amount, 
        tenure_months, fin_type, distance, own_cap, guarantor, 
        hist_late, dti_ratio, target_npf
    ])

# 5. สร้าง DataFrame และ บันทึกเป็น CSV
columns = [
    'Age', 'Monthly_Income', 'Occupation', 'Membership_Months', 
    'Share_Capital', 'Financing_Amount', 'Financing_Tenure_Months', 
    'Financing_Type', 'Business_Distance_km', 'Own_Capital_Ratio', 
    'Guarantor_Type', 'Historical_Late_Payments', 'Debt_to_Income_Ratio', 
    'Target_NPF'
]

df = pd.DataFrame(data, columns=columns)

# ตรวจสอบสัดส่วน NPF ที่สร้างได้
print("--- สรุปชุดข้อมูลสังเคราะห์ ---")
print(f"จำนวนข้อมูลทั้งหมด: {len(df)} แถว")
print(df['Target_NPF'].value_counts(normalize=True) * 100)

# เซฟไฟล์
df.to_csv('islamic_coop_synthetic_data.csv', index=False)
print("\nบันทึกไฟล์ 'islamic_coop_synthetic_data.csv' สำเร็จแล้ว!")