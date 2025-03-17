import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# 设置中文字体和负号显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False  # 坐标轴负数的负号显示

# 模拟表格数据，需按实际表格内容替换以下数据
data = {
    "时间/min": ["15", "15", "15", "15", "15", "15", 
                 "30", "30", "30", "30", "30", "30",
                 "45", "45", "45", "45", "45", "45",
                 "60", "60", "60", "60", "60", "60"],
    "评价指标": ["MAE"] * 24,
    "模型": ["HA", "SVR", "GRU", "LSTM", "AGRU", "ALSTM"] * 4,
    "值": [0.3, 0.2189, 0.1410, 0.0663, 0.0676, 0.0514,  # 15分钟MAE
           0.35, 0.2541, 0.1643, 0.0847, 0.0835, 0.0784,  # 30分钟MAE
           0.4, 0.2893, 0.1914, 0.1057, 0.1038, 0.1017,  # 45分钟MAE
           0.45, 0.3245, 0.2208, 0.1264, 0.1214, 0.1264]  # 60分钟MAE
}

df = pd.DataFrame(data)

plt.figure(figsize=(12, 6))
sns.barplot(
    x="时间/min", 
    y="值", 
    hue="模型", 
    hue_order=["HA", "SVR", "GRU", "LSTM", "AGRU", "ALSTM"],
    data=df
)
plt.title("不同预测粒度下各模型 MAE 对比")
plt.xlabel("预测粒度")
plt.ylabel("MAE 值")
plt.legend(title="模型")
plt.show()