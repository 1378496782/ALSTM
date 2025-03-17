import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# 设置中文字体和负号显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False  # 坐标轴负数的负号显示

# 模拟表格数据，需按实际表格内容替换以下数据
data = {
    "时间/min": ["15"] * 12 + ["30"] * 12 + ["45"] * 12 + ["60"] * 12,
    "评价指标": ["MAE", "RMSE"] * 24,
    "模型": ["HA", "HA", "SVR", "SVR", "GRU", "GRU", "LSTM", "LSTM", "AGRU", "AGRU", "ALSTM", "ALSTM"] * 4,
    "值": [0.3, 0.4,  # 15分钟 HA 的 MAE、RMSE
           0.2189, 0.3488,  # 15分钟 SVR 的 MAE、RMSE
           0.1410, 0.2333,  # 15分钟 GRU 的 MAE、RMSE
           0.0663, 0.1614,  # 15分钟 LSTM 的 MAE、RMSE
           0.0676, 0.1545,  # 15分钟 AGRU 的 MAE、RMSE
           0.0514, 0.1405,  # 15分钟 ALSTM 的 MAE、RMSE
           # 30分钟数据
           0.35, 0.45, 0.2541, 0.3954, 0.1643, 0.2772, 0.0847, 0.2050, 0.0835, 0.1925, 0.0784, 0.1923,
           # 45分钟数据
           0.4, 0.5, 0.2893, 0.4414, 0.1914, 0.3192, 0.1057, 0.2461, 0.1038, 0.2288, 0.1017, 0.2369,
           # 60分钟数据
           0.45, 0.55, 0.3245, 0.4872, 0.2208, 0.3622, 0.1264, 0.2844, 0.1214, 0.2590, 0.1264, 0.2759]
}

df = pd.DataFrame(data)

# 创建图形和子图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# 绘制MAE对比图
for model in ["LSTM", "GRU", "AGRU", "ALSTM"]:
    subset = df[(df["模型"] == model) & (df["评价指标"] == "MAE")]
    ax1.plot(
        subset["时间/min"], 
        subset["值"], 
        marker="o", 
        label=model
    )

ax1.set_title("不同预测粒度下的MAE对比")
ax1.set_xlabel("预测粒度（分钟）")
ax1.set_ylabel("MAE值")
ax1.legend(title="模型")
ax1.grid(True, alpha=0.3)

# 绘制RMSE对比图
for model in ["LSTM", "GRU", "AGRU", "ALSTM"]:
    subset = df[(df["模型"] == model) & (df["评价指标"] == "RMSE")]
    ax2.plot(
        subset["时间/min"], 
        subset["值"], 
        marker="o", 
        label=model
    )

ax2.set_title("不同预测粒度下的RMSE对比")
ax2.set_xlabel("预测粒度（分钟）")
ax2.set_ylabel("RMSE值")
ax2.legend(title="模型")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()