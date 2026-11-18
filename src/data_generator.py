import os
import numpy as np
import pandas as pd


def generate_data(total_rows=10000, save_path=None):
  print(f"⚙️ กำลังสร้างชุดข้อมูลสังเคราะห์เชิงลึกจำนวน {total_rows:,} แถว...")
  np.random.seed(42)

  # 1. ข้อมูลพื้นฐานและสินเชื่อ
  age = np.random.randint(20, 65, total_rows)
  monthly_income = np.random.randint(9000, 150000, total_rows)
  financing_amount = np.random.randint(10000, 1000000, total_rows)
  financing_tenure_months = np.random.choice(
      [12, 24, 36, 48, 60, 84, 120, 240], total_rows
  )
  financing_type = np.random.choice(["Personal", "Business"], total_rows)
  business_distance_km = np.random.randint(1, 30, total_rows)
  own_capital_ratio = np.random.uniform(0.0, 0.5, total_rows)
  occupation = np.random.choice(
      [
          "ข้าราชการ/รัฐวิสาหกิจ",
          "พนักงานเอกชน",
          "เจ้าของธุรกิจ/ค้าขาย",
          "อาชีพอิสระ/เกษตรกร",
      ],
      total_rows,
  )

  # 2. ปัจจัยครอบครัวและรายได้สุทธิ (Net Monthly Income)
  family_members = np.random.randint(1, 16, total_rows)
  annual_expenses = (
      family_members * np.random.randint(25000, 40000, total_rows)
  ) + np.random.randint(40000, 80000, total_rows)
  net_monthly_income = monthly_income - (annual_expenses / 12)

  # 3. ปัจจัยหลักประกัน (LTV & Guarantor Quality)
  collateral_value = financing_amount * np.random.uniform(0.5, 2.5, total_rows)
  ltv_ratio = financing_amount / collateral_value
  guarantor_type = np.random.choice(
      [
          "ข้าราชการ",
          "เงินฝาก 100%",
          "โฉนดที่ดิน",
          "พนักงานเอกชน",
          "บุคคลทั่วไป",
      ],
      total_rows,
  )
  guarantor_quality = [
      "A"
      if g in ["ข้าราชการ", "เงินฝาก 100%"]
      else "B"
      if g in ["โฉนดที่ดิน", "พนักงานเอกชน"]
      else "C"
      for g in guarantor_type
  ]

  # 4. ปัจจัยสุขภาพการเงินและวินัยการออม (Credit Health & Shariah Cooperative Principle)
  membership_months = np.random.randint(1, 120, total_rows)
  share_capital = financing_amount * np.random.uniform(0.05, 0.3, total_rows)
  historical_late_payments = np.random.randint(0, 6, total_rows)
  consecutive_saving_months = np.random.randint(0, 60, total_rows)

  credit_health_score = (
      100
      - (historical_late_payments * 15)
      + (consecutive_saving_months * 0.5)
      + ((share_capital / financing_amount) * 20)
  )
  credit_health_score = np.clip(credit_health_score, 0, 100)

  # 5. ภาระหนี้ต่อรายได้สุทธิ (Net DTI Ratio)
  profit_rate = 0.05
  monthly_installment = (
      financing_amount
      + (financing_amount * profit_rate * (financing_tenure_months / 12))
  ) / financing_tenure_months
  net_dti_ratio = np.where(
      net_monthly_income > 0, monthly_installment / net_monthly_income, 1.5
  )

  # 6. คำนวณความน่าจะเป็นหนี้เสีย (Target NPF)
  npf_prob = np.zeros(total_rows)
  npf_prob += np.where(net_dti_ratio > 0.4, 0.35, 0.0)
  npf_prob += np.where(net_monthly_income <= 0, 0.35, 0.0)
  npf_prob += np.where(ltv_ratio > 1.0, 0.20, 0.0)
  npf_prob += np.array(
      [
          0.15 if q == "C" else (0.05 if q == "B" else 0.0)
          for q in guarantor_quality
      ]
  )
  npf_prob += np.where(credit_health_score < 50, 0.25, 0.0)

  npf_prob = np.clip(npf_prob + np.random.normal(0, 0.1, total_rows), 0, 1)
  target_npf = np.where(npf_prob > 0.6, 1, 0)

  df = pd.DataFrame({
      "Age": age,
      "Monthly_Income": monthly_income,
      "Family_Members": family_members,
      "Annual_Expenses": np.round(annual_expenses, 2),
      "Net_Monthly_Income": np.round(net_monthly_income, 2),
      "Occupation": occupation,
      "Financing_Amount": financing_amount,
      "Financing_Tenure_Months": financing_tenure_months,
      "Financing_Type": financing_type,
      "Business_Distance_km": business_distance_km,
      "Own_Capital_Ratio": np.round(own_capital_ratio, 2),
      "Collateral_Value": np.round(collateral_value, 2),
      "LTV_Ratio": np.round(ltv_ratio, 4),
      "Guarantor_Type": guarantor_type,
      "Guarantor_Quality": guarantor_quality,
      "Membership_Months": membership_months,
      "Share_Capital": np.round(share_capital, 2),
      "Historical_Late_Payments": historical_late_payments,
      "Consecutive_Saving_Months": consecutive_saving_months,
      "Credit_Health_Score": np.round(credit_health_score, 2),
      "Net_DTI_Ratio": np.round(net_dti_ratio, 4),
      "Target_NPF": target_npf,
  })

  if save_path:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)

  return df