import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import plotly.express as px

st.set_page_config(page_title="体外受精-胚胎移植临床妊娠预测", layout="wide", page_icon="🩺")

# 新标题
st.title("🩺 体外受精-胚胎移植临床妊娠预测系统")
st.markdown("**Stacking 集成模型（RF + CatBoost + Logistic Regression）** | 目标变量：Clinicalpreg")


# ==================== 加载模型 ====================
@st.cache_resource
def load_models():
    try:
        models = {
            'RF': joblib.load('models/RF.pkl'),
            'CatBoost': joblib.load('models/CatBoost.pkl'),
            'LR': joblib.load('models/LR.pkl')
        }
        meta_learner = joblib.load('models/meta_learner.pkl')
        feature_names = joblib.load('models/feature_names.pkl')
        return models, meta_learner, feature_names
    except Exception as e:
        st.error(f"模型加载失败: {e}")
        st.stop()


models, meta_learner, feature_names = load_models()

# ==================== 侧边栏输入 ====================
st.sidebar.header("📋 患者特征输入")

input_data = {}
for feature in feature_names:
    input_data[feature] = st.sidebar.number_input(
        label=feature,
        value=0.0,
        format="%.4f"
    )

# ==================== 预测 ====================
if st.sidebar.button("🚀 开始预测", type="primary", use_container_width=True):
    with st.spinner("正在进行预测..."):
        input_df = pd.DataFrame([input_data])

        meta_features = np.zeros((1, 3))
        base_probs = {}

        for i, (name, model) in enumerate(models.items()):
            proba = model.predict_proba(input_df)[0, 1]
            meta_features[0, i] = proba
            base_probs[name] = proba

        final_proba = meta_learner.predict_proba(meta_features)[0, 1]

        # 显示结果
        col1, col2 = st.columns([2, 3])
        with col1:
            st.subheader("🎯 预测结果")
            if final_proba >= 0.5:
                st.error(f"**临床妊娠概率较高**\n概率: **{final_proba:.4f}**")
            else:
                st.success(f"**临床妊娠概率较低**\n概率: **{final_proba:.4f}**")

        with col2:
            fig = px.bar(
                x=['未妊娠', '临床妊娠'],
                y=[1 - final_proba, final_proba],
                color_discrete_sequence=['#EF553B', '#00CC96'],
                title="临床妊娠概率分布"
            )
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("各基模型预测概率")
        st.dataframe(pd.DataFrame.from_dict(base_probs, orient='index', columns=['概率']))

        # SHAP解释
        st.subheader("📊 SHAP 特征贡献解释 (CatBoost)")
        try:
            explainer = shap.TreeExplainer(models['CatBoost'])
            shap_values = explainer.shap_values(input_df)
            if isinstance(shap_values, list):
                shap_values_pos = shap_values[1]
                expected_value = explainer.expected_value[1]
            else:
                shap_values_pos = shap_values
                expected_value = explainer.expected_value

            shap_fig = shap.plots.waterfall(
                shap.Explanation(
                    values=shap_values_pos[0],
                    base_values=expected_value,
                    data=input_df.iloc[0],
                    feature_names=feature_names
                )
            )
            st.pyplot(shap_fig)
        except Exception as e:
            st.warning(f"SHAP解释生成失败: {e}")

st.caption("模型基于 RF + CatBoost + Logistic Regression Stacking 集成学习")
