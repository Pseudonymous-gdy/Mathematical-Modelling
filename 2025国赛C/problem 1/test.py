import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

# ========== 1. 读取数据 ==========
FILE = "附件_男.csv"
df = pd.read_csv(FILE, encoding="gb18030")

# ========== 2. 选择特征和目标 ==========
features = ["年龄", "身高", "体重", "IVF妊娠", "检测抽血次数", "检测孕日", "孕妇BMI"]
target = "Y染色体浓度"

X = df[features].copy()
y = df[target].copy()

# IVF妊娠为类别特征
X["IVF妊娠"] = X["IVF妊娠"].astype(str)

# ========== 3. 预处理 ==========
numeric_cols = [c for c in features if c != "IVF妊娠"]
categorical_cols = ["IVF妊娠"]

preprocess = ColumnTransformer([
    ("num", "passthrough", numeric_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
])

# ========== 4. 模型 ==========
rf_tuned = RandomForestRegressor(
    n_estimators=1000,
    max_depth=12,            # 控制树深度（常见 8~20 之间试）
    min_samples_split=6,     # 节点再分裂的最少样本（常见 4/6/8）
    min_samples_leaf=3,      # 叶子最少样本（常见 1/2/3/5/10）
    max_features="sqrt",     # 每次分裂考虑的特征子集（回归默认是1.0，这里改为sqrt增强随机性）
    random_state=42,
    n_jobs=-1
)

pipe_tuned = Pipeline([
    ("preprocess", preprocess),
    ("model", rf_tuned)
])

# ========== 5. 训练/测试划分 ==========
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

pipe_tuned.fit(X_train, y_train)
y_pred = pipe_tuned.predict(X_test)

print("Test R² :", r2_score(y_test, y_pred))
print("Test RMSE:", mean_squared_error(y_test, y_pred, squared=False))

# ========== 6. 特征重要性 ==========
ohe = pipe_tuned.named_steps["preprocess"].transformers_[1][1]
cat_features = ohe.get_feature_names_out(categorical_cols)
all_features = numeric_cols + list(cat_features)

importances = pipe_tuned.named_steps["model"].feature_importances_
imp = pd.DataFrame({"feature": all_features, "importance": importances}) \
        .sort_values("importance", ascending=False)

print("\nTop feature importances:")
print(imp)

# ========== 7. 可视化并保存 ==========
plt.rcParams['font.sans-serif'] = ['SimHei']    # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False      # 解决负号显示问题

# 特征重要性图
plt.figure(figsize=(8, 5))
plt.barh(imp["feature"].head(10)[::-1], imp["importance"].head(10)[::-1])
plt.xlabel("重要性")
plt.title("前10个特征的重要性 (随机森林)")
plt.tight_layout()
plt.savefig("rf_top10_importances.png", dpi=300)
plt.close()

# 真实值 vs 预测值图
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
plt.xlabel("真实 Y染色体浓度")
plt.ylabel("预测 Y染色体浓度")
plt.title("真实值 vs 预测值")
plt.tight_layout()
plt.savefig("rf_y_true_vs_pred.png", dpi=300)
plt.close()

print("图片已保存: rf_top10_importances.png, rf_y_true_vs_pred.png")
