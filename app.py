# ============================================================
# ICM-AAR Pre-scan Risk Stratification Tool
# 碘造影剂急性不良反应 CT增强扫描前风险分层工具
# 
# Streamlit Cloud Deployment Version
# ============================================================

import streamlit as st
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
import plotly.graph_objects as go
import os

# ==================== Page Config ====================
st.set_page_config(
    page_title="ICM-AAR Risk Tool",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== Load Model ====================
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "Final_CatBoost_Model_Corrected.cbm")
    model = CatBoostClassifier()
    model.load_model(model_path)
    return model

model = load_model()

# Platt scaling parameters (from R glm output)
PLATT_INTERCEPT = -10.011311
PLATT_SLOPE = 9.056946

# Feature order (alphabetical, must match training)
FEATURE_ORDER = [
    "Age", "Alanine_Aminotransferase", "Albumin", "Albumin_Globulin_Ratio",
    "Alkaline_Phosphatase", "Allergy_History", "Alpha_Fetoprotein",
    "Aspartate_Aminotransferase", "Basophil_Count", "Blood_Urea_Nitrogen",
    "Carbon_Dioxide_Combining_Power_tested", "Carcinoembryonic_Antigen",
    "Chemotherapy_History", "Chloride_tested", "Cholinesterase", "Colon_Descending",
    "Comorbidity_Anemia", "Creatinine", "Cystatin_C_tested", "Direct_Bilirubin",
    "Eosinophil_Percentage", "Fibrinogen_tested", "Gamma_Glutamyl_Transferase",
    "Gastric_Cancer_Other", "Globulin", "Glucose_tested", "Hematocrit", "Hemoglobin",
    "Indirect_Bilirubin", "Intervention_Stent", "Lung_Cancer_Unspecified_Lobe",
    "Lymphocyte_Percentage", "Magnesium_tested", "Mean_Platelet_Volume",
    "Metastasis_Lung", "Monocyte_Percentage", "Neutrophil_Percentage",
    "Nutrition_Hypoalbuminemia", "Patient_Source", "Phosphorus_tested",
    "Platelet_Count", "Platelet_Distribution_Width", "Plateletcrit",
    "Potassium_tested", "Prealbumin_tested", "Primary_Esophageal_Cancer",
    "Primary_Head_Neck_Cancer", "Radiotherapy_History", "Red_Blood_Cell_Count",
    "Red_Cell_Distribution_Width", "Sodium_tested", "Stage_Neoadjuvant",
    "Symptom_Pain", "Thrombin_Time_tested", "Total_Bile_Acid", "Total_Bilirubin",
    "Total_Carbon_Dioxide_tested", "Treatment_Radiotherapy", "Treatment_Surgery",
    "Uric_Acid", "White_Blood_Cell_Count"
]

# Categorical feature indices (0-based) matching FEATURE_ORDER
CAT_FEATURES = [
    "Allergy_History", "Chemotherapy_History", "Colon_Descending",
    "Comorbidity_Anemia", "Gastric_Cancer_Other", "Intervention_Stent",
    "Lung_Cancer_Unspecified_Lobe", "Metastasis_Lung", "Nutrition_Hypoalbuminemia",
    "Patient_Source", "Primary_Esophageal_Cancer", "Primary_Head_Neck_Cancer",
    "Radiotherapy_History", "Stage_Neoadjuvant", "Symptom_Pain",
    "Treatment_Radiotherapy", "Treatment_Surgery",
    "Carbon_Dioxide_Combining_Power_tested", "Chloride_tested", "Cystatin_C_tested",
    "Fibrinogen_tested", "Glucose_tested", "Magnesium_tested", "Phosphorus_tested",
    "Potassium_tested", "Prealbumin_tested", "Sodium_tested",
    "Thrombin_Time_tested", "Total_Carbon_Dioxide_tested"
]
CAT_INDICES = [FEATURE_ORDER.index(c) for c in CAT_FEATURES]

# Bilingual feature names for display
FEAT_LABELS = {
    "Radiotherapy_History": "Prior Radiotherapy 既往放疗史",
    "Allergy_History": "Allergy History 过敏史",
    "Age": "Age 年龄",
    "Patient_Source": "Patient Source 患者来源",
    "Carcinoembryonic_Antigen": "CEA",
    "Monocyte_Percentage": "Monocyte% 单核细胞%",
    "Cystatin_C_tested": "Cystatin C (ordered) 胱抑素C(开具)",
    "Creatinine": "Creatinine 肌酐",
    "Lymphocyte_Percentage": "Lymphocyte% 淋巴细胞%",
    "Blood_Urea_Nitrogen": "BUN 尿素氮",
    "Alkaline_Phosphatase": "ALP 碱性磷酸酶",
    "Eosinophil_Percentage": "Eosinophil% 嗜酸性粒细胞%",
    "Glucose_tested": "Glucose (ordered) 血糖(开具)",
    "Prealbumin_tested": "Prealbumin (ordered) 前白蛋白(开具)",
    "Neutrophil_Percentage": "Neutrophil% 中性粒细胞%",
    "Phosphorus_tested": "Phosphorus (ordered) 磷(开具)",
    "Uric_Acid": "Uric Acid 尿酸",
    "Hematocrit": "HCT 红细胞压积",
    "Aspartate_Aminotransferase": "AST 天冬氨酸转氨酶",
    "Direct_Bilirubin": "Direct Bilirubin 直接胆红素",
    "Plateletcrit": "Plateletcrit 血小板压积",
    "Total_Bile_Acid": "Total Bile Acid 总胆汁酸",
    "Albumin": "Albumin 白蛋白",
    "Hemoglobin": "Hemoglobin 血红蛋白",
    "Cholinesterase": "Cholinesterase 胆碱酯酶",
    "Carbon_Dioxide_Combining_Power_tested": "CO₂CP (ordered) CO₂结合力(开具)",
    "Platelet_Distribution_Width": "PDW 血小板分布宽度",
    "White_Blood_Cell_Count": "WBC 白细胞计数",
    "Red_Blood_Cell_Count": "RBC 红细胞计数",
    "Platelet_Count": "PLT 血小板计数",
    "Gamma_Glutamyl_Transferase": "GGT γ-谷氨酰转移酶",
    "Chemotherapy_History": "Prior Chemotherapy 既往化疗史",
    "Mean_Platelet_Volume": "MPV 平均血小板体积",
    "Fibrinogen_tested": "Fibrinogen (ordered) 纤维蛋白原(开具)",
    "Red_Cell_Distribution_Width": "RDW 红细胞分布宽度",
    "Globulin": "Globulin 球蛋白",
    "Alanine_Aminotransferase": "ALT 丙氨酸转氨酶",
    "Alpha_Fetoprotein": "AFP 甲胎蛋白",
    "Total_Bilirubin": "Total Bilirubin 总胆红素",
    "Indirect_Bilirubin": "Indirect Bilirubin 间接胆红素",
    "Thrombin_Time_tested": "Thrombin Time (ordered) 凝血酶时间(开具)",
    "Basophil_Count": "Basophil Count 嗜碱性粒细胞计数",
    "Albumin_Globulin_Ratio": "A/G Ratio 白球比",
    "Magnesium_tested": "Magnesium (ordered) 镁(开具)",
    "Sodium_tested": "Sodium (ordered) 钠(开具)",
    "Potassium_tested": "Potassium (ordered) 钾(开具)",
    "Total_Carbon_Dioxide_tested": "Total CO₂ (ordered) 总CO₂(开具)",
    "Chloride_tested": "Chloride (ordered) 氯(开具)",
    "Colon_Descending": "Descending Colon Ca. 降结肠癌",
    "Lung_Cancer_Unspecified_Lobe": "Lung Ca. (unspecified) 肺癌(未指定叶)",
    "Primary_Head_Neck_Cancer": "Head & Neck Ca. 头颈部癌",
    "Primary_Esophageal_Cancer": "Esophageal Ca. 食管癌",
    "Treatment_Surgery": "Current Surgery 当前手术",
    "Gastric_Cancer_Other": "Gastric Ca. (other) 胃癌(其他)",
    "Intervention_Stent": "Stent Placement 支架置入",
    "Metastasis_Lung": "Lung Metastasis 肺转移",
    "Comorbidity_Anemia": "Anemia 贫血",
    "Treatment_Radiotherapy": "Current Radiotherapy 当前放疗",
    "Stage_Neoadjuvant": "Neoadjuvant Stage 新辅助治疗阶段",
    "Symptom_Pain": "Pain Symptom 疼痛症状",
    "Nutrition_Hypoalbuminemia": "Hypoalbuminemia 低白蛋白血症",
}

def get_label(feat):
    return FEAT_LABELS.get(feat, feat)

# ==================== Custom CSS ====================
st.markdown("""
<style>
    .main-header {
        text-align: center; padding: 10px 0; 
        border-bottom: 3px solid #2c3e50; margin-bottom: 20px;
    }
    .main-header h1 { font-size: 26px; color: #2c3e50; margin-bottom: 2px; }
    .main-header p { color: #7f8c8d; font-size: 14px; margin-top: 0; }
    .risk-box {
        text-align: center; padding: 25px; border-radius: 12px; margin: 10px 0 20px 0;
    }
    .risk-low { background: linear-gradient(135deg, #d4efdf, #a9dfbf); border: 2px solid #27ae60; }
    .risk-med { background: linear-gradient(135deg, #fdebd0, #f9e79f); border: 2px solid #f39c12; }
    .risk-high { background: linear-gradient(135deg, #fadbd8, #f1948a); border: 2px solid #e74c3c; }
    .risk-prob { font-size: 42px; font-weight: 800; margin: 8px 0; }
    .risk-label { font-size: 18px; font-weight: 700; }
    .risk-action { font-size: 13px; margin-top: 8px; font-weight: 500; }
    .metric-card {
        background: white; padding: 15px; border-radius: 8px; text-align: center;
        box-shadow: 0 2px 6px rgba(0,0,0,0.08); margin: 5px 0;
    }
    .metric-val { font-size: 22px; font-weight: 700; color: #2c3e50; }
    .metric-lab { font-size: 11px; color: #95a5a6; }
    .info-box {
        background: #eaf2f8; border-left: 4px solid #3498db; padding: 12px;
        border-radius: 0 6px 6px 0; font-size: 12px; color: #2c3e50; margin: 10px 0;
    }
    .section-head {
        background: #34495e; color: white; padding: 6px 12px; border-radius: 4px;
        font-size: 13px; font-weight: 600; margin: 12px 0 6px 0;
    }
    div[data-testid="stSidebar"] { background-color: #fafbfc; }
</style>
""", unsafe_allow_html=True)

# ==================== Header ====================
st.markdown("""
<div class="main-header">
    <h1>🏥 ICM-AAR Pre-scan Risk Stratification Tool</h1>
    <p>碘造影剂急性不良反应 CT增强扫描前风险分层工具 | CatBoost + Platt Calibration | 61 Features</p>
</div>
""", unsafe_allow_html=True)

# ==================== Sidebar Inputs ====================
with st.sidebar:
    st.markdown("### 📋 Patient Data 患者数据")
    
    # --- Demographics ---
    st.markdown('<div class="section-head">Demographics 人口学信息</div>', unsafe_allow_html=True)
    age = st.number_input("Age 年龄 (years/岁)", min_value=8, max_value=100, value=61, step=1)
    patient_source = st.selectbox("Patient Source 患者来源", 
                                   ["Inpatient 住院", "Outpatient 门诊"], index=0)
    allergy = st.selectbox("Allergy History 过敏史", ["No 无", "Yes 有"], index=0)
    
    # --- Treatment History ---
    st.markdown('<div class="section-head">Treatment History 治疗史</div>', unsafe_allow_html=True)
    radio_hx = st.selectbox("Prior Radiotherapy 既往放疗", ["Yes 有", "No 无"], index=0)
    chemo_hx = st.selectbox("Prior Chemotherapy 既往化疗", ["Yes 有", "No 无"], index=0)
    tx_radio = st.selectbox("Current Radiotherapy 当前放疗", ["No 无", "Yes 有"], index=0)
    tx_surgery = st.selectbox("Current Surgery 当前手术", ["No 无", "Yes 有"], index=0)
    stage_neo = st.selectbox("Neoadjuvant Stage 新辅助阶段", ["No 否", "Yes 是"], index=0)
    
    # --- Tumor ---
    st.markdown('<div class="section-head">Tumor Characteristics 肿瘤特征</div>', unsafe_allow_html=True)
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        esoph = st.selectbox("Esophageal 食管", ["No", "Yes"], index=0, key="eso")
        gastric_o = st.selectbox("Gastric(other) 胃癌", ["No", "Yes"], index=0, key="gas")
        lung_uns = st.selectbox("Lung(unspec) 肺癌", ["No", "Yes"], index=0, key="lun")
        colon_d = st.selectbox("Desc.Colon 降结肠", ["No", "Yes"], index=0, key="col")
    with col_t2:
        head_neck = st.selectbox("Head&Neck 头颈", ["No", "Yes"], index=0, key="hn")
        met_lung = st.selectbox("Lung Meta 肺转移", ["No", "Yes"], index=0, key="met")
        stent = st.selectbox("Stent 支架", ["No", "Yes"], index=0, key="ste")
        anemia = st.selectbox("Anemia 贫血", ["No", "Yes"], index=0, key="ane")
    
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        hypoalb = st.selectbox("Hypoalbuminemia 低白蛋白", ["No", "Yes"], index=0, key="hyp")
    with col_c2:
        pain = st.selectbox("Pain 疼痛", ["No", "Yes"], index=0, key="pain")
    
    # --- Liver Function ---
    st.markdown('<div class="section-head">Liver Function 肝功能</div>', unsafe_allow_html=True)
    col_l1, col_l2 = st.columns(2)
    with col_l1:
        alb = st.number_input("ALB 白蛋白 (g/L)", value=41.5, step=0.1, format="%.1f")
        ag_ratio = st.number_input("A/G 白球比", value=1.4, step=0.1, format="%.1f")
        alt_v = st.number_input("ALT (U/L)", value=19.0, step=1.0, format="%.0f")
        alp_v = st.number_input("ALP (U/L)", value=91.0, step=1.0, format="%.0f")
        che = st.number_input("ChE 胆碱酯酶 (U/L)", value=7687.0, step=10.0, format="%.0f")
        dbil = st.number_input("DBIL 直胆 (μmol/L)", value=2.4, step=0.1, format="%.1f")
    with col_l2:
        glo = st.number_input("GLB 球蛋白 (g/L)", value=28.7, step=0.1, format="%.1f")
        ast_v = st.number_input("AST (U/L)", value=23.0, step=1.0, format="%.0f")
        ggt = st.number_input("GGT (U/L)", value=33.0, step=1.0, format="%.0f")
        tbil = st.number_input("TBIL 总胆 (μmol/L)", value=13.4, step=0.1, format="%.1f")
        ibil = st.number_input("IBIL 间胆 (μmol/L)", value=10.5, step=0.1, format="%.1f")
        tba = st.number_input("TBA 总胆汁酸 (μmol/L)", value=3.6, step=0.1, format="%.1f")
    
    # --- Renal + Markers ---
    st.markdown('<div class="section-head">Renal / Tumor Markers 肾功能/肿瘤标志物</div>', unsafe_allow_html=True)
    col_r1, col_r2 = st.columns(2)
    with col_r1:
        cr = st.number_input("Cr 肌酐 (μmol/L)", value=75.1, step=0.1, format="%.1f")
        ua = st.number_input("UA 尿酸 (μmol/L)", value=306.0, step=1.0, format="%.0f")
    with col_r2:
        bun = st.number_input("BUN 尿素氮 (mmol/L)", value=5.3, step=0.1, format="%.1f")
        cea = st.number_input("CEA (ng/mL)", value=2.44, step=0.01, format="%.2f")
    afp = st.number_input("AFP 甲胎蛋白 (ng/mL) — leave 0 if not tested 未检测填0", 
                           value=0.0, min_value=0.0, step=0.1, format="%.1f")
    
    # --- CBC ---
    st.markdown('<div class="section-head">Complete Blood Count 血常规</div>', unsafe_allow_html=True)
    col_b1, col_b2 = st.columns(2)
    with col_b1:
        wbc = st.number_input("WBC (×10⁹/L)", value=6.3, step=0.01, format="%.2f")
        hb = st.number_input("Hb (g/L)", value=137.0, step=1.0, format="%.0f")
        plt_c = st.number_input("PLT (×10⁹/L)", value=230.0, step=1.0, format="%.0f")
        lym_p = st.number_input("LYM% 淋巴%", value=27.8, step=0.1, format="%.1f")
        eos_p = st.number_input("EOS% 嗜酸%", value=1.6, step=0.1, format="%.1f")
        mpv = st.number_input("MPV (fL)", value=9.8, step=0.1, format="%.1f")
        pct = st.number_input("PCT 血小板压积 (%)", value=0.23, step=0.01, format="%.2f")
    with col_b2:
        rbc = st.number_input("RBC (×10¹²/L)", value=4.48, step=0.01, format="%.2f")
        hct = st.number_input("HCT 红细胞压积 (%)", value=40.8, step=0.1, format="%.1f")
        neu_p = st.number_input("NEU% 中性粒%", value=61.2, step=0.1, format="%.1f")
        mono_p = st.number_input("MONO% 单核%", value=7.2, step=0.1, format="%.1f")
        baso = st.number_input("BASO 嗜碱 (×10⁹/L)", value=0.03, step=0.01, format="%.2f")
        pdw = st.number_input("PDW (fL)", value=11.7, step=0.1, format="%.1f")
        rdw = st.number_input("RDW (%)", value=13.2, step=0.1, format="%.1f")
    
    # --- Lab Test Ordering ---
    st.markdown('<div class="section-head">Lab Test Ordering 实验室检测开具</div>', unsafe_allow_html=True)
    st.markdown('<div class="info-box">Whether each test was ordered before scan (reflects clinical complexity assessment).<br>'
                '以下记录增强CT前是否开具了该检验（反映医生对患者病情复杂度的判断）</div>', unsafe_allow_html=True)
    
    col_o1, col_o2 = st.columns(2)
    with col_o1:
        t_cys = st.selectbox("Cystatin C 胱抑素C", ["Not ordered 未开具", "Ordered 已开具"], index=0)
        t_glu = st.selectbox("Glucose 血糖", ["Not ordered 未开具", "Ordered 已开具"], index=1)
        t_mag = st.selectbox("Magnesium 镁", ["Not ordered 未开具", "Ordered 已开具"], index=1)
        t_co2cp = st.selectbox("CO₂CP CO₂结合力", ["Not ordered 未开具", "Ordered 已开具"], index=0)
        t_fib = st.selectbox("Fibrinogen 纤维蛋白原", ["Not ordered 未开具", "Ordered 已开具"], index=0)
        t_cl = st.selectbox("Chloride 氯", ["Not ordered 未开具", "Ordered 已开具"], index=1)
    with col_o2:
        t_pre = st.selectbox("Prealbumin 前白蛋白", ["Not ordered 未开具", "Ordered 已开具"], index=1)
        t_phos = st.selectbox("Phosphorus 磷", ["Not ordered 未开具", "Ordered 已开具"], index=1)
        t_na = st.selectbox("Sodium 钠", ["Not ordered 未开具", "Ordered 已开具"], index=1)
        t_tco2 = st.selectbox("Total CO₂ 总CO₂", ["Not ordered 未开具", "Ordered 已开具"], index=0)
        t_tt = st.selectbox("Thrombin Time 凝血酶时间", ["Not ordered 未开具", "Ordered 已开具"], index=0)
        t_k = st.selectbox("Potassium 钾", ["Not ordered 未开具", "Ordered 已开具"], index=1)

    predict_btn = st.button("🔍 Calculate Risk 计算风险", use_container_width=True, type="primary")

# ==================== Helper ====================
def yn(val):
    """Convert Yes/No selectbox to int"""
    return 1 if "Yes" in val or "有" in val or "是" in val or "Ordered" in val or "已开具" in val else 0

def platt_calibrate(raw_p):
    logit = PLATT_INTERCEPT + PLATT_SLOPE * raw_p
    return 1.0 / (1.0 + np.exp(-logit))

# ==================== Predict ====================
if predict_btn:
    # Build feature dict
    afp_val = afp if afp > 0 else np.nan  # 0 means not tested
    
    data = {
        "Age": age, "Alanine_Aminotransferase": alt_v, "Albumin": alb,
        "Albumin_Globulin_Ratio": ag_ratio, "Alkaline_Phosphatase": alp_v,
        "Allergy_History": yn(allergy), "Alpha_Fetoprotein": afp_val,
        "Aspartate_Aminotransferase": ast_v, "Basophil_Count": baso,
        "Blood_Urea_Nitrogen": bun,
        "Carbon_Dioxide_Combining_Power_tested": yn(t_co2cp),
        "Carcinoembryonic_Antigen": cea, "Chemotherapy_History": yn(chemo_hx),
        "Chloride_tested": yn(t_cl), "Cholinesterase": che,
        "Colon_Descending": yn(colon_d), "Comorbidity_Anemia": yn(anemia),
        "Creatinine": cr, "Cystatin_C_tested": yn(t_cys),
        "Direct_Bilirubin": dbil, "Eosinophil_Percentage": eos_p,
        "Fibrinogen_tested": yn(t_fib), "Gamma_Glutamyl_Transferase": ggt,
        "Gastric_Cancer_Other": yn(gastric_o), "Globulin": glo,
        "Glucose_tested": yn(t_glu), "Hematocrit": hct, "Hemoglobin": hb,
        "Indirect_Bilirubin": ibil, "Intervention_Stent": yn(stent),
        "Lung_Cancer_Unspecified_Lobe": yn(lung_uns),
        "Lymphocyte_Percentage": lym_p, "Magnesium_tested": yn(t_mag),
        "Mean_Platelet_Volume": mpv, "Metastasis_Lung": yn(met_lung),
        "Monocyte_Percentage": mono_p, "Neutrophil_Percentage": neu_p,
        "Nutrition_Hypoalbuminemia": yn(hypoalb),
        "Patient_Source": 1 if "Outpatient" in patient_source else 0,
        "Phosphorus_tested": yn(t_phos), "Platelet_Count": plt_c,
        "Platelet_Distribution_Width": pdw, "Plateletcrit": pct,
        "Potassium_tested": yn(t_k), "Prealbumin_tested": yn(t_pre),
        "Primary_Esophageal_Cancer": yn(esoph),
        "Primary_Head_Neck_Cancer": yn(head_neck),
        "Radiotherapy_History": yn(radio_hx),
        "Red_Blood_Cell_Count": rbc, "Red_Cell_Distribution_Width": rdw,
        "Sodium_tested": yn(t_na), "Stage_Neoadjuvant": yn(stage_neo),
        "Symptom_Pain": yn(pain), "Thrombin_Time_tested": yn(t_tt),
        "Total_Bile_Acid": tba, "Total_Bilirubin": tbil,
        "Total_Carbon_Dioxide_tested": yn(t_tco2),
        "Treatment_Radiotherapy": yn(tx_radio),
        "Treatment_Surgery": yn(tx_surgery), "Uric_Acid": ua,
        "White_Blood_Cell_Count": wbc,
    }
    
    df = pd.DataFrame([data])[FEATURE_ORDER]
    
    # Convert categorical columns to str for CatBoost
    for c in CAT_FEATURES:
        df[c] = df[c].astype(str)
    
    pool = Pool(data=df, cat_features=CAT_INDICES)
    
    # Predict
    raw_prob = model.predict_proba(pool)[0, 1]
    cal_prob = platt_calibrate(raw_prob)
    
    # Risk level
    if raw_prob >= 0.5:
        risk = "high"
    elif raw_prob >= 0.032:
        risk = "medium"
    else:
        risk = "low"
    
    # SHAP values
    try:
        shap_vals = model.get_feature_importance(pool, type="ShapValues")[0]
        shap_features = shap_vals[:-1]  # last is base value
        has_shap = True
    except:
        has_shap = False
    
    # ========== Display Results ==========
    risk_config = {
        "low":    {"cls": "risk-low",  "label": "✅ LOW RISK 低风险",
                   "action": "Standard protocol recommended. 可按常规流程进行增强CT检查。"},
        "medium": {"cls": "risk-med",  "label": "⚠️ MODERATE RISK 中风险",
                   "action": "Enhanced monitoring recommended. Prepare emergency medications. 建议加强监测，备好急救药品。"},
        "high":   {"cls": "risk-high", "label": "🚨 HIGH RISK 高风险",
                   "action": "Emergency preparation required. Consider alternative imaging. 建议延长观察，备好急救设备，考虑替代检查。"},
    }
    rc = risk_config[risk]
    
    st.markdown(f"""
    <div class="risk-box {rc['cls']}">
        <div class="risk-label">{rc['label']}</div>
        <div class="risk-prob">{cal_prob*100:.4f}%</div>
        <div style="font-size:12px; color:#555;">
            Calibrated probability of moderate-to-severe ICM-AAR<br>
            校准后中重度碘造影剂急性不良反应概率
        </div>
        <div class="risk-action">{rc['action']}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Metric cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="metric-card"><div class="metric-val">{raw_prob*100:.2f}%</div>'
                    f'<div class="metric-lab">Raw Probability<br>原始概率</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card"><div class="metric-val">{cal_prob*100:.4f}%</div>'
                    f'<div class="metric-lab">Calibrated Probability<br>校准概率</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="metric-card"><div class="metric-val">0.905</div>'
                    '<div class="metric-lab">Model AUC-ROC<br>模型判别力</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown('<div class="metric-card"><div class="metric-val">99.91%</div>'
                    '<div class="metric-lab">NPV<br>阴性预测值</div></div>', unsafe_allow_html=True)
    
    # SHAP plot
    st.markdown("---")
    st.subheader("Feature Contribution Analysis 特征贡献分析")
    
    if has_shap:
        # Top 20 by absolute SHAP
        abs_shap = np.abs(shap_features)
        top_idx = np.argsort(abs_shap)[::-1][:20]
        
        top_feats = [FEATURE_ORDER[i] for i in top_idx]
        top_vals = [shap_features[i] for i in top_idx]
        top_labels = [get_label(f) for f in top_feats]
        colors = ["#e74c3c" if v > 0 else "#2980b9" for v in top_vals]
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=top_labels[::-1], x=top_vals[::-1],
            orientation='h',
            marker_color=colors[::-1],
            text=[f"{v:+.3f}" for v in top_vals[::-1]],
            textposition="outside",
            textfont=dict(size=11),
        ))
        fig.update_layout(
            title=dict(text="Individual SHAP Values (Top 20 Features)<br>"
                           "<sub>个体SHAP值 — 红色=增加风险 Red=Risk↑ | 蓝色=降低风险 Blue=Risk↓</sub>",
                       font=dict(size=15)),
            xaxis_title="SHAP Value (impact on log-odds)",
            height=550, margin=dict(l=250, r=80, t=80, b=50),
            plot_bgcolor="white",
            font=dict(size=11),
        )
        fig.update_xaxes(gridcolor="#eee", zeroline=True, zerolinecolor="#333", zerolinewidth=1)
        fig.update_yaxes(gridcolor="#eee")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("SHAP values could not be computed. 无法计算SHAP值。")
    
    # Patient input summary
    with st.expander("📊 Patient Input Summary 患者输入摘要"):
        display_df = pd.DataFrame({
            "Feature 特征": [get_label(f) for f in FEATURE_ORDER],
            "Value 值": [data[f] for f in FEATURE_ORDER]
        })
        st.dataframe(display_df, use_container_width=True, height=400)

else:
    # Welcome screen
    st.markdown("""
    <div class="info-box">
        <b>Instructions 使用说明:</b><br>
        1. Enter patient data in the left sidebar. 在左侧边栏输入患者数据。<br>
        2. Click "Calculate Risk" to generate prediction. 点击"计算风险"生成预测。<br>
        3. Review the calibrated probability and SHAP analysis. 查看校准概率和SHAP特征贡献分析。<br><br>
        <b>Model Information 模型信息:</b><br>
        • Algorithm: CatBoost gradient boosting with Platt scaling calibration<br>
        • Training cohort: 56,392 cancer patients (July 2022 – June 2024)<br>
        • Temporal validation: 19,099 patients (July 2024 – June 2025)<br>
        • AUC-ROC = 0.905 (95% CI: 0.870–0.937) | NPV = 99.91%<br>
        • 61 features from electronic health records (demographics, treatment, labs, test-ordering patterns)<br>
        • All inputs are pre-scan data — no contrast agent information required<br><br>
        <b>⚠️ Disclaimer 免责声明:</b> This tool is for research purposes only and does not replace clinical judgment.<br>
        本工具仅用于科研目的，不替代临床判断。
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align:center; font-size:11px; color:#95a5a6;">
    Authors and institution blinded for peer review.<br>
    Model: CatBoost v1.2.5 | 61 features | Temporal validation AUC-ROC = 0.905
</div>
""", unsafe_allow_html=True)
