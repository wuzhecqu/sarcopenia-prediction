import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import warnings
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score

warnings.filterwarnings('ignore')

# ===================== 0. 全局配置 =====================

# 在设置中文字体的位置，新增兼容云端的字体配置
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
# 若仍乱码，可禁用中文（可选）
# plt.rcParams['font.sans-serif'] = ['DejaVu Sans']


# 页面配置
st.set_page_config(
    page_title="肌少症预测与可解释性分析",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===================== 1. 关键路径配置（务必核对！） =====================
#MODEL_PATH = r'F:\Project\chenxiao\sarcopenia\sarcopenia_model_final.pkl'
#VAL_DATA_PATH = r'F:\Project\chenxiao\sarcopenia\validation_data.xlsx'
#ROC_DIR = r'F:\Project\chenxiao\sarcopenia\roc'

# 新相对路径（云端用，文件和app.py放在同一目录）
MODEL_PATH = "sarcopenia_model_final.pkl"
VAL_DATA_PATH = "validation_data.xlsx"



# ===================== 2. 模型加载（修复解包错误+增加校验） =====================
@st.cache_resource
def load_model():
    """修复模型加载逻辑，避免返回None"""
    # 第一步：检查文件是否存在
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ 模型文件不存在！路径：{MODEL_PATH}")
        st.stop()

    # 第二步：加载模型并处理缺失组件
    try:
        save_dict = joblib.load(MODEL_PATH)
        # 核心组件（必须有）
        model = save_dict.get('model')
        scaler = save_dict.get('scaler')
        encoders = save_dict.get('encoders', {})
        median_dict = save_dict.get('median_dict', {})
        mode_dict = save_dict.get('mode_dict', {})
        # SHAP组件（非必须，缺失则置None）
        explainer = save_dict.get('shap_explainer')
        shap_values_val = save_dict.get('shap_values_val')

        # 校验核心组件
        if model is None or scaler is None:
            st.error("❌ 模型文件损坏！缺失核心组件（model/scaler）")
            st.stop()

        return model, scaler, encoders, median_dict, mode_dict, explainer, shap_values_val
    except Exception as e:
        st.error(f"❌ 模型加载失败：{str(e)}")
        st.error("建议重新运行训练代码生成模型文件！")
        st.stop()


# 分步加载，避免一次性解包出错
model, scaler, encoders, median_dict, mode_dict, explainer, shap_values_val = load_model()


# ===================== 3. 验证集加载（修复预处理逻辑） =====================
@st.cache_data
def load_val_data():
    """加载并预处理验证集"""
    # 检查验证集文件
    if not os.path.exists(VAL_DATA_PATH):
        st.error(f"❌ 验证集文件不存在！路径：{VAL_DATA_PATH}")
        st.stop()

    # 读取原始数据
    val_df = pd.read_excel(VAL_DATA_PATH, header=0, engine='openpyxl')
    # 分离特征和标签
    X_val = val_df.iloc[:, 1:27].copy()  # 必须copy()避免SettingWithCopyWarning
    y_val = val_df.iloc[:, 0].copy()

    # 区分特征类型
    numeric_cols = X_val.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = X_val.select_dtypes(include=['object', 'category']).columns

    # 预处理（严格匹配训练逻辑）
    # 1. 缺失值填充
    for col in numeric_cols:
        if col in median_dict:
            X_val[col].fillna(median_dict[col], inplace=True)
    # app.py中单样本预测/批量预测的类别特征处理部分（添加astype(str)）
    # 示例：单样本预测中
    for col in categorical_cols:
        input_data[col] = st.selectbox(f"{col}", val_df_ori[col].astype(str).unique())

    # 预处理时（关键）
    input_df = pd.DataFrame([input_data])
    for col in categorical_cols:
        # 强制转字符串 + 处理未见过的类别
        input_df_col = input_df[col].astype(str).fillna("unknown")
        # 兼容编码器未见过的类别（替换为0）
        try:
            input_df[col] = encoders[col].transform(input_df_col)
        except:
            input_df[col] = 0  # 未见过的类别默认值

    # 3. 数值特征标准化
    X_val[numeric_cols] = scaler.transform(X_val[numeric_cols])

    return val_df, X_val, y_val, numeric_cols, categorical_cols


# 加载验证集+特征类型
val_df_ori, X_val, y_val, numeric_cols, categorical_cols = load_val_data()

# ===================== 4. 侧边栏功能选择 =====================
st.sidebar.title("功能菜单")
function_choice = st.sidebar.radio(
    "选择功能",
    ["📊 模型评估结果", "🔮 单样本预测", "📤 批量预测", "📈 可解释性分析"]
)

# ===================== 5. 模型评估结果页面（修复指标计算） =====================
if function_choice == "📊 模型评估结果":
    st.title("肌少症预测模型 - 验证集评估结果")
    st.markdown("### 数据集信息")
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"训练集样本量：5593例")
        st.write(f"验证集样本量：{len(y_val)}例")
        st.write(f"特征数量：26个（性别、教育程度、饮酒、吸烟等）")

    # 计算预测结果（核心修复：避免空值）
    y_val_pred = model.predict(X_val)
    y_val_pred_proba = model.predict_proba(X_val)[:, 1]

    with col2:
        # 安全计算指标（避免除零错误）
        accuracy = round(accuracy_score(y_val, y_val_pred), 4)
        precision = round(precision_score(y_val, y_val_pred, zero_division=0), 4)
        recall = round(recall_score(y_val, y_val_pred, zero_division=0), 4)
        auc = round(roc_auc_score(y_val, y_val_pred_proba), 4)

        st.write(f"准确率：{accuracy}")
        st.write(f"精确率：{precision}")
        st.write(f"召回率：{recall}")
        st.write(f"AUC-ROC：{auc}")

    # 混淆矩阵
    st.markdown("### 混淆矩阵")
    cm = confusion_matrix(y_val, y_val_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['无肌少症', '有肌少症'],
                yticklabels=['无肌少症', '有肌少症'], ax=ax)
    ax.set_xlabel('预测标签')
    ax.set_ylabel('真实标签')
    ax.set_title(f'验证集混淆矩阵（{len(y_val)}例）')
    st.pyplot(fig)

    # ROC曲线
    st.markdown("### ROC曲线")
    fpr, tpr, _ = roc_curve(y_val, y_val_pred_proba)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC曲线 (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('假阳性率（FPR）')
    ax.set_ylabel('真阳性率（TPR）')
    ax.set_title('验证集ROC曲线')
    ax.legend(loc="lower right")
    st.pyplot(fig)

# ===================== 6. 单样本预测页面（修复特征输入） =====================
elif function_choice == "🔮 单样本预测":
    st.title("肌少症单样本预测")
    st.markdown("### 输入患者特征")

    # 构建输入表单
    input_data = {}
    col1, col2 = st.columns(2)

    # 数值特征输入
    with col1:
        st.subheader("数值特征")
        for col in numeric_cols:
            # 安全获取统计值（避免空值）
            min_val = float(val_df_ori[col].min()) if not val_df_ori[col].isna().all() else 0.0
            max_val = float(val_df_ori[col].max()) if not val_df_ori[col].isna().all() else 100.0
            mean_val = float(val_df_ori[col].median()) if not val_df_ori[col].isna().all() else 50.0

            input_data[col] = st.number_input(
                f"{col}",
                min_value=min_val,
                max_value=max_val,
                value=mean_val,
                step=0.1
            )

    # 类别特征输入
    with col2:
        st.subheader("类别特征")
        for col in categorical_cols:
            # 安全获取唯一值
            unique_vals = val_df_ori[col].dropna().unique() if col in val_df_ori.columns else []
            if len(unique_vals) == 0:
                unique_vals = ["未知"]
            input_data[col] = st.selectbox(f"{col}", unique_vals)

    # 预测按钮
    if st.button("🚀 开始预测"):
        # 转换为DataFrame
        input_df = pd.DataFrame([input_data])

        # 预处理（兼容缺失值）
        for col in numeric_cols:
            input_df[col].fillna(0.0, inplace=True)
        for col in categorical_cols:
            input_df[col].fillna(unique_vals[0], inplace=True)

        # 编码
        for col in categorical_cols:
            if col in encoders:
                try:
                    # 处理未见过的类别
                    input_df[col] = encoders[col].transform(input_df[col])
                except:
                    input_df[col] = 0

        # 标准化
        input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

        # 预测
        pred_proba = model.predict_proba(input_df)[0, 1]
        pred_label = 1 if pred_proba >= 0.5 else 0
        pred_text = "有肌少症" if pred_label == 1 else "无肌少症"

        # 展示结果
        st.markdown("### 预测结果")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("预测类别", pred_text)
            st.metric("肌少症概率", f"{pred_proba:.4f}")
        with col2:
            # SHAP解释（兼容缺失的explainer）
            if explainer is not None:
                st.markdown("#### 特征影响解释")
                shap_val = explainer.shap_values(input_df)
                fig, ax = plt.subplots(figsize=(10, 4))
                shap.force_plot(
                    explainer.expected_value,
                    shap_val[0],
                    input_data,
                    matplotlib=True,
                    show=False,
                    figsize=(10, 4),
                    ax=ax
                )
                st.pyplot(fig)
            else:
                st.warning("⚠️ SHAP解释器未加载，无法展示特征影响（请重新训练模型并保存SHAP组件）")

# ===================== 7. 批量预测页面（修复数据处理） =====================
elif function_choice == "📤 批量预测":
    st.title("肌少症批量预测")
    st.markdown("### 上传待预测数据（Excel格式）")
    st.markdown("⚠️ 数据格式要求：第一行是特征名，列顺序与训练集一致（无标签列）")

    # 文件上传
    uploaded_file = st.file_uploader("选择Excel文件", type=["xlsx"])
    if uploaded_file is not None:
        # 读取上传数据
        test_df = pd.read_excel(uploaded_file, header=0, engine='openpyxl')
        st.write(f"上传数据样本量：{len(test_df)}例，特征数：{test_df.shape[1]}")

        # 预处理
        X_test = test_df.copy()

        # 缺失值填充
        for col in numeric_cols:
            if col in X_test.columns and col in median_dict:
                X_test[col].fillna(median_dict[col], inplace=True)
        for col in categorical_cols:
            if col in X_test.columns and col in mode_dict:
                X_test[col].fillna(mode_dict[col], inplace=True)

        # 编码
        for col in categorical_cols:
            if col in X_test.columns and col in encoders:
                try:
                    X_test[col] = encoders[col].transform(X_test[col])
                except:
                    X_test[col] = 0

        # 标准化
        for col in numeric_cols:
            if col in X_test.columns:
                X_test[col] = scaler.transform(X_test[col].values.reshape(-1, 1))

        # 批量预测
        if st.button("🚀 批量预测"):
            pred_proba = model.predict_proba(X_test)[:, 1]
            pred_label = (pred_proba >= 0.5).astype(int)
            # 结果整合
            result_df = test_df.copy()
            result_df["肌少症预测概率"] = pred_proba
            result_df["肌少症预测标签"] = pred_label
            result_df["肌少症预测结果"] = result_df["肌少症预测标签"].map({0: "无肌少症", 1: "有肌少症"})

            # 展示结果
            st.markdown("### 预测结果")
            st.dataframe(result_df.head(10))

            # 下载结果（兼容中文）
            csv_data = result_df.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label="📥 下载预测结果（CSV）",
                data=csv_data,
                file_name="肌少症预测结果.csv",
                mime="text/csv"
            )

# ===================== 8. 可解释性分析页面（兼容SHAP缺失） =====================
elif function_choice == "📈 可解释性分析":
    st.title("模型可解释性分析（SHAP）")

    # 检查SHAP组件
    if explainer is None or shap_values_val is None:
        st.warning("⚠️ SHAP组件未加载！请重新运行训练代码并保存SHAP解释器和SHAP值")
    else:
        tab1, tab2, tab3 = st.tabs(["📊 SHAP汇总图", "🔍 特征依赖图", "🧬 单样本解释"])

        # Tab1: SHAP汇总图
        with tab1:
            st.markdown("### 验证集SHAP特征影响汇总图")
            st.markdown("""
            - 纵轴：特征重要性（从上到下影响越大）
            - 横轴：SHAP值（正值=增加肌少症概率，负值=降低）
            - 颜色：特征取值（红色=高值，蓝色=低值）
            """)
            fig, ax = plt.subplots(figsize=(12, 8))
            shap.summary_plot(
                shap_values_val, X_val,
                feature_names=X_val.columns,
                plot_type="dot",
                show=False,
                ax=ax,
                cmap=plt.get_cmap("coolwarm")
            )
            st.pyplot(fig)

        # Tab2: 特征依赖图
        with tab2:
            st.markdown("### 特征依赖图（Top5特征）")
            # 计算Top5特征
            shap_importance = np.abs(shap_values_val).mean(axis=0)
            top5_feat = X_val.columns[np.argsort(shap_importance)[-5:]][::-1]
            selected_feat = st.selectbox("选择特征", top5_feat)

            fig, ax = plt.subplots(figsize=(8, 6))
            shap.dependence_plot(
                selected_feat,
                shap_values_val,
                X_val,
                feature_names=X_val.columns,
                show=False,
                ax=ax,
                alpha=0.6,
                dot_size=20
            )
            ax.set_title(f"特征依赖图 - {selected_feat}")
            st.pyplot(fig)

        # Tab3: 单样本解释
        with tab3:
            st.markdown("### 验证集单样本SHAP解释")
            # 选择样本
            sample_idx = st.slider("选择验证集样本索引", 0, len(X_val) - 1, 100)
            # 展示样本信息
            st.markdown("#### 样本特征")
            sample_data = val_df_ori.iloc[sample_idx]
            st.write(sample_data.iloc[:27])

            # 展示SHAP力图
            st.markdown("#### 特征影响解释")
            fig, ax = plt.subplots(figsize=(12, 4))
            shap.force_plot(
                explainer.expected_value,
                shap_values_val[sample_idx],
                X_val.iloc[sample_idx],
                feature_names=X_val.columns,
                matplotlib=True,
                show=False,
                figsize=(12, 4),
                ax=ax
            )
            ax.set_title(f"样本{sample_idx} - 真实标签：{'有肌少症' if y_val.iloc[sample_idx] == 1 else '无肌少症'}")
            st.pyplot(fig)

# ===================== 9. 页脚 =====================
st.markdown("---")
st.markdown("© 2025 肌少症预测模型 - Streamlit网页版 | 基于XGBoost + SHAP可解释性分析")