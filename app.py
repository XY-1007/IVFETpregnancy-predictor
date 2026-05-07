import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="IVF 临床妊娠预测", layout="wide", page_icon="🩺")
st.title("🩺 体外受精-胚胎移植临床妊娠预测系统")
st.markdown("**Stacking 集成模型（RF + CatBoost + LR）**")

# ==================== 尝试加载模型 ====================
try:
    import joblib
    st.success("✅ joblib 导入成功")
    
    models = {
        'RF': joblib.load('models/RF.pkl'),
        'CatBoost': joblib.load('models/CatBoost.pkl'),
        'LR': joblib.load('models/LR.pkl')
    }
    meta_learner = joblib.load('models/meta_learner.pkl')
    feature_names = joblib.load('models/feature_names.pkl')
    
    st.success(f"✅ 所有模型加载成功！共有 {len(feature_names)} 个特征")
    
except Exception as e:
    st.error(f"❌ 模型加载失败: {e}")
    st.info("提示：请检查 requirements.txt 是否正确上传")
    st.stop()

st.info("🎉 系统准备就绪！请在侧边栏输入特征值进行预测。")
