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
@st.cache_data
@st.cache_data
@st.cache_data
def load_val_data():
    val_df = None
    X_val = None
    y_val = None
    numeric_cols = []
    categorical_cols = []

    try:
        if not os.path.exists(VAL_DATA_PATH):
            st.error(f"验证集文件不存在：{VAL_DATA_PATH}")
            return val_df, X_val, y_val, numeric_cols, categorical_cols
        
        val_df = pd.read_excel(VAL_DATA_PATH, header=0, engine='openpyxl')
        if val_df.shape[1] < 27:
            st.error(f"验证集列数不足！当前：{val_df.shape[1]}列，要求≥27列")
            return val_df, X_val, y_val, numeric_cols, categorical_cols
        
        X_val = val_df.iloc[:, 1:27].copy()
        y_val = val_df.iloc[:, 0].copy()
        
        # ========== 修复类别特征识别 ==========
        # 1. 扩大筛选范围：包含object/category/string类型
        categorical_cols = X_val.select_dtypes(include=['object', 'category', 'string']).columns.tolist()
        # 2. 手动指定类别特征列名（兜底！替换为你训练集中的类别特征名，比如['性别', '饮酒', '吸烟']）
        manual_categorical_cols = ['性别', '饮酒状态', '吸烟状态', '教育程度']  # 替换为你的实际列名
        for col in manual_categorical_cols:
            if col in X_val.columns and col not in categorical_cols:
                # 强制转为字符串类型，加入类别特征列表
                X_val[col] = X_val[col].astype(str)
                categorical_cols.append(col)
        
        # 3. 数值特征：排除类别特征，避免重复
        numeric_cols = [col for col in X_val.columns if col not in categorical_cols]
        
        # ========== 预处理适配 ==========
        # 数值特征填充+标准化
        if len(numeric_cols) > 0:
            for col in numeric_cols:
                X_val[col].fillna(median_dict.get(col, 0), inplace=True)
            X_val[numeric_cols] = scaler.transform(X_val[numeric_cols])
        
        # 类别特征填充+编码（兼容空列表）
        if len(categorical_cols) > 0:
            for col in categorical_cols:
                X_val[col] = X_val[col].astype(str).fillna("unknown")  # 强制转字符串
                if col in encoders:
                    X_val[col] = encoders[col].transform(X_val[col])
        
        st.success("✅ 验证集加载成功！")
        # 打印特征识别结果（调试用，可选）
        st.write(f"识别到数值特征：{len(numeric_cols)}个 | 类别特征：{len(categorical_cols)}个")
        return val_df, X_val, y_val, numeric_cols, categorical_cols

    except Exception as e:
        st.error(f"验证集加载失败：{str(e)}")
        return val_df, X_val, y_val, numeric_cols, categorical_cols

  



# 加载验证集后的边界检查
val_df_ori, X_val, y_val, numeric_cols, categorical_cols = load_val_data()

# 优化提示（仅在空列表时显示信息，非警告）
if len(numeric_cols) == 0:
    st.info("ℹ️ 未检测到数值特征，部分功能可能受限")  # 蓝色信息，非黄色警告
if len(categorical_cols) == 0:
    st.info("ℹ️ 未检测到类别特征（已使用手动指定列表兜底），不影响核心功能")
    
# 关键修复：补充全局边界检查，避免空列表导致后续报错
# 初始化默认特征列表（防止为空）

    categorical_cols = []

# 补充X_val/y_val的默认值，避免后续预测报错
if X_val is None:
    X_val = pd.DataFrame()  # 空DataFrame
if y_val is None:
    y_val = pd.Series()     # 空Series



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


# -------------------- 5.2 单样本预测 --------------------
elif function_choice == "🔮 单样本预测":
    st.title("肌少症单样本预测")
    st.subheader("输入患者特征")
    
    # 关键修复1：提前初始化input_data（空字典），避免未定义
    input_data = {}
    
    # 关键修复2：补充边界条件，避免categorical_cols/numeric_cols为空
    if len(numeric_cols) == 0:
        st.warning("⚠️ 未检测到数值特征，请检查验证集数据！")
    else:
        col1, col2 = st.columns(2)
        with col1:
            # 只遍历存在的数值特征，避免空循环
            for col in numeric_cols[:13]:
                # 关键修复3：兼容val_df_ori为空的情况
                default_val = 0.0 if val_df_ori is None else float(val_df_ori[col].median())
                input_data[col] = st.number_input(f"{col}", value=default_val)
        with col2:
            # 剩余数值特征
            for col in numeric_cols[13:]:
                default_val = 0.0 if val_df_ori is None else float(val_df_ori[col].median())
                input_data[col] = st.number_input(f"{col}", value=default_val)
            
            # 关键修复4：兼容categorical_cols为空的情况
            if len(categorical_cols) > 0:
                for col in categorical_cols:
                    # 兼容val_df_ori为空/特征无唯一值的情况
                    if val_df_ori is None or col not in val_df_ori.columns:
                        input_data[col] = st.selectbox(f"{col}", ["未知"])
                    else:
                        # 强制转字符串，避免类型错误
                        unique_vals = val_df_ori[col].astype(str).unique()
                        input_data[col] = st.selectbox(f"{col}", unique_vals)
    
    # 预测按钮（关键修复5：只有input_data非空时才执行预测）
    if st.button("🚀 开始预测") and len(input_data) > 0:
        # 预处理
        input_df = pd.DataFrame([input_data])
        # 兼容categorical_cols为空
        if len(categorical_cols) > 0:
            for col in categorical_cols:
                # 强制转字符串 + 处理未见过的类别
                input_df_col = input_df[col].astype(str).fillna("unknown")
                try:
                    input_df[col] = encoders[col].transform(input_df_col)
                except:
                    input_df[col] = 0  # 未见过的类别默认值
        
        # 标准化数值特征（兼容numeric_cols为空）
        if len(numeric_cols) > 0:
            input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])
        
        # 预测
        pred_proba = model.predict_proba(input_df)[0, 1]
        pred_label = 1 if pred_proba >= 0.5 else 0
        pred_text = "有肌少症" if pred_label == 1 else "无肌少症"
        
        # 展示结果
        st.subheader("预测结果")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("预测类别", pred_text)
            st.metric("肌少症概率", f"{pred_proba:.4f}")
        with col2:
            if explainer is not None:
                st.subheader("特征影响解释")
                shap_val = explainer.shap_values(input_df)
                fig, ax = plt.subplots(figsize=(10, 4))
                shap.force_plot(explainer.expected_value, shap_val[0], input_data, matplotlib=True, show=False, ax=ax)
                st.pyplot(fig)

# ===================== 9. 页脚 =====================
st.markdown("---")

st.markdown("© 2025 肌少症预测模型 - Streamlit网页版 | 基于XGBoost + SHAP可解释性分析")




