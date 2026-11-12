import pandas as pd
import numpy as np
import random

def generate_data(total_rows=10000, save_path=None):
    print(f"กำลังสร้างข้อมูลสังเคราะห์จำนวน {total_rows} แถว...")
    np.random.seed(42)
    random.seed(42)

    NEW_CUST_ROWS = int(total_rows * 0.3)
    data = []

    def generate_income_by_occupation():
        occ = random.choice(['ข้าราชการ/รัฐวิสาหกิจ', 'พนักงานเอกชน', 'เจ้าของธุรกิจ/ค้าขาย', 'อาชีพอิสระ/เกษตรกร'])
        if occ == 'ข้าราชการ/รัฐวิสาหกิจ': return occ, np.random.randint(15000, 80000)
        elif occ == 'พนักงานเอกชน': return occ, np.random.randint(12000, 100000)
        elif occ == 'เจ้าของธุรกิจ/ค้าขาย': return occ, np.random.randint(15000, 150000)
        else: return occ, np.random.randint(7000, 40000)

    for i in range(total_rows):
        age = np.random.randint(20, 66)
        occ, income = generate_income_by_occupation()
        fin_amount = np.random.randint(10000, 500001)
        tenure_months = random.choice([12, 24, 36, 48, 60, 72, 84, 96, 108, 120])
        fin_type = random.choice(['Personal', 'Business'])
        distance = np.random.randint(1, 31)
        own_cap = np.random.randint(0, 51) / 100.0
        guarantor = random.choice(['บุคคลทั่วไป', 'ข้าราชการ/รัฐวิสาหกิจ', 'โฉนดที่ดิน', 'เงินฝากค้ำประกัน 100%'])
        
        if i < NEW_CUST_ROWS:
            hist_late, member_months, share_cap = 0, np.random.randint(1, 13), np.random.randint(1000, 10000)
        else:
            hist_late, member_months, share_cap = np.random.randint(0, 6), np.random.randint(13, 121), np.random.randint(5000, 200001)

        profit_rate = 0.05
        monthly_installment = (fin_amount + (fin_amount * profit_rate * (tenure_months / 12))) / tenure_months
        dti_ratio = round(monthly_installment / income, 2)

        npf_prob = 0.05
        if dti_ratio > 0.4: npf_prob += 0.25
        if hist_late >= 2: npf_prob += 0.30
        if share_cap / fin_amount < 0.10: npf_prob += 0.15
        if fin_type == 'Business' and own_cap < 0.30 and distance > 15: npf_prob += 0.20

        target_npf = np.random.choice([0, 1], p=[max(0, 1-npf_prob), min(1, npf_prob)])
        data.append([age, income, occ, member_months, share_cap, fin_amount, tenure_months, fin_type, distance, own_cap, guarantor, hist_late, dti_ratio, target_npf])

    cols = ['Age', 'Monthly_Income', 'Occupation', 'Membership_Months', 'Share_Capital', 'Financing_Amount', 'Financing_Tenure_Months', 'Financing_Type', 'Business_Distance_km', 'Own_Capital_Ratio', 'Guarantor_Type', 'Historical_Late_Payments', 'Debt_to_Income_Ratio', 'Target_NPF']
    df = pd.DataFrame(data, columns=cols)
    
    if save_path:
        df.to_csv(save_path, index=False)
        
    return df