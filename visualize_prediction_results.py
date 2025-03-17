import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# 设置matplotlib中文字体支持
plt.rcParams['font.sans-serif'] = ['Hiragino Sans GB', 'SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False  # 坐标轴负数的负号显示

# 定义表格中的数据
time_intervals = [15, 30, 45, 60]  # 时间间隔（分钟）
models = ['HA', 'SVR', 'GRU', 'LSTM', 'AGRU', 'ALSTM']  # 模型名称

# 各个指标的数据
# MAE数据
mae_data = {
    15: [0.3, 0.2189, 0.1410, 0.0663, 0.0676, 0.0514],
    30: [0.35, 0.2541, 0.1643, 0.0847, 0.0835, 0.0784],
    45: [0.4, 0.2893, 0.1914, 0.1057, 0.1038, 0.1017],
    60: [0.45, 0.3245, 0.2208, 0.1264, 0.1214, 0.1264]
}

# RMSE数据
rmse_data = {
    15: [0.4, 0.3488, 0.2333, 0.1614, 0.1545, 0.1405],
    30: [0.45, 0.3954, 0.2772, 0.2050, 0.1925, 0.1923],
    45: [0.5, 0.4414, 0.3192, 0.2461, 0.2288, 0.2369],
    60: [0.55, 0.4872, 0.3622, 0.2844, 0.2590, 0.2759]
}

# ACC数据
acc_data = {
    15: [0.5, 0.6480, 0.7645, 0.8765, 0.8818, 0.8925],
    30: [0.55, 0.6009, 0.7202, 0.8435, 0.8531, 0.8532],
    45: [0.6, 0.5537, 0.6773, 0.8125, 0.8257, 0.8195],
    60: [0.65, 0.5062, 0.6327, 0.7836, 0.8029, 0.7901]
}

# R²数据
r2_data = {
    15: [0.6, 0.8750, 0.9439, 0.9721, 0.9745, 0.9789],
    30: [0.65, 0.8377, 0.9191, 0.9531, 0.9589, 0.9577],
    45: [0.7, 0.7944, 0.8901, 0.9305, 0.9404, 0.9340],
    60: [0.75, 0.7449, 0.8546, 0.9054, 0.9227, 0.9094]
}

# Var数据
var_data = {
    15: [0.7, 0.8773, 0.9442, 0.9721, 0.9745, 0.9803],
    30: [0.75, 0.8406, 0.9192, 0.9532, 0.9590, 0.9625],
    45: [0.8, 0.7979, 0.8902, 0.9305, 0.9405, 0.9450],
    60: [0.85, 0.7488, 0.8548, 0.9056, 0.9231, 0.9271]
}

# 创建一个函数来绘制每个指标随时间间隔变化的折线图
def plot_metric_by_time(metric_data, metric_name, ylabel, is_higher_better=True):
    plt.figure(figsize=(12, 8))
    
    for i, model in enumerate(models):
        model_values = [metric_data[t][i] for t in time_intervals]
        plt.plot(time_intervals, model_values, marker='o', linewidth=2, label=model)
    
    plt.title(f'{metric_name}随预测粒度的变化趋势', fontsize=16)
    plt.xlabel('预测粒度（分钟）', fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    # 设置x轴刻度
    plt.xticks(time_intervals)
    
    # 如果是越低越好的指标，添加说明
    if not is_higher_better:
        plt.annotate('↓ 越低越好', xy=(0.02, 0.02), xycoords='axes fraction', fontsize=12)
    else:
        plt.annotate('↑ 越高越好', xy=(0.02, 0.02), xycoords='axes fraction', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(f'{metric_name}_by_time.png', dpi=300, bbox_inches='tight')

# 创建一个函数来绘制每个时间间隔下各模型的性能对比条形图
def plot_models_comparison_by_metric(metric_data, metric_name, ylabel, is_higher_better=True):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for i, time_interval in enumerate(time_intervals):
        ax = axes[i]
        values = metric_data[time_interval]
        
        # 创建条形图
        bars = ax.bar(models, values, color=['skyblue', 'lightgreen', 'lightcoral', 'wheat', 'violet', 'lightpink'])
        
        # 高亮ALSTM和AGRU
        bars[-2].set_color('mediumorchid')
        bars[-1].set_color('crimson')
        
        ax.set_title(f'预测粒度: {time_interval}分钟', fontsize=14)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.grid(axis='y', alpha=0.3)
        
        # 添加数值标签
        for j, v in enumerate(values):
            ax.text(j, v * (1.05 if is_higher_better else 0.95), f'{v:.4f}', ha='center', fontsize=10)
    
    plt.suptitle(f'不同预测粒度下各模型的{metric_name}对比', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig(f'{metric_name}_models_comparison.png', dpi=300, bbox_inches='tight')

# 创建一个综合性能雷达图
def plot_radar_chart():
    # 选择15分钟预测粒度的数据进行雷达图绘制
    time_interval = 15
    
    # 准备雷达图数据
    categories = ['MAE', 'RMSE', 'ACC', 'R²', 'Var']
    
    # 对于MAE和RMSE，值越小越好，所以取反
    mae_values = [1 - v for v in mae_data[time_interval]]
    rmse_values = [1 - v for v in rmse_data[time_interval]]
    
    # 其他指标值越大越好
    acc_values = acc_data[time_interval]
    r2_values = r2_data[time_interval]
    var_values = var_data[time_interval]
    
    # 归一化所有数据到[0,1]区间
    def normalize(values):
        min_val = min(values)
        max_val = max(values)
        return [(v - min_val) / (max_val - min_val) for v in values]
    
    mae_norm = normalize(mae_values)
    rmse_norm = normalize(rmse_values)
    acc_norm = normalize(acc_values)
    r2_norm = normalize(r2_values)
    var_norm = normalize(var_values)
    
    # 组合所有指标数据
    model_data = {
        models[i]: [mae_norm[i], rmse_norm[i], acc_norm[i], r2_norm[i], var_norm[i]]
        for i in range(len(models))
    }
    
    # 设置雷达图
    angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]  # 闭合雷达图
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    # 添加每个类别的标签
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    plt.xticks(angles[:-1], categories, fontsize=12)
    
    # 绘制每个模型的雷达图
    for model, values in model_data.items():
        values += values[:1]  # 闭合雷达图
        ax.plot(angles, values, linewidth=2, label=model)
        ax.fill(angles, values, alpha=0.1)
    
    # 添加图例
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1), fontsize=12)
    
    plt.title('15分钟预测粒度下各模型性能雷达图', fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig('model_performance_radar.png', dpi=300, bbox_inches='tight')

# 创建一个热力图来展示不同模型在不同预测粒度下的综合性能
def plot_heatmap():
    # 计算每个模型在每个预测粒度下的综合得分
    # 综合得分 = (1-MAE) + (1-RMSE) + ACC + R² + Var
    scores = {}
    
    for t in time_intervals:
        scores[t] = []
        for i in range(len(models)):
            # 对于MAE和RMSE，值越小越好，所以用1减去它们
            score = (1 - mae_data[t][i]) + (1 - rmse_data[t][i]) + acc_data[t][i] + r2_data[t][i] + var_data[t][i]
            scores[t].append(score)
    
    # 创建得分矩阵
    score_matrix = np.array([scores[t] for t in time_intervals])
    
    # 绘制热力图
    plt.figure(figsize=(12, 8))
    sns.heatmap(score_matrix, annot=True, fmt='.2f', cmap='YlGnBu',
                xticklabels=models, yticklabels=[f'{t}分钟' for t in time_intervals])
    
    plt.title('不同预测粒度下各模型的综合性能得分', fontsize=16)
    plt.xlabel('模型', fontsize=14)
    plt.ylabel('预测粒度', fontsize=14)
    plt.tight_layout()
    plt.savefig('model_performance_heatmap.png', dpi=300, bbox_inches='tight')

# 创建一个综合图表，展示所有指标在15分钟预测粒度下的对比
def plot_comprehensive_comparison():
    # 选择15分钟预测粒度的数据
    time_interval = 15
    
    # 创建子图
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    axes = axes.flatten()
    
    # 1. MAE对比图
    ax = axes[0]
    values = mae_data[time_interval]
    bars = ax.bar(models, values, color=['skyblue', 'lightgreen', 'lightcoral', 'wheat', 'violet', 'lightpink'])
    bars[-2].set_color('mediumorchid')
    bars[-1].set_color('crimson')
    ax.set_title('平均绝对误差 (MAE)', fontsize=14)
    ax.set_ylabel('MAE', fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    
    # 添加数值标签
    for j, v in enumerate(values):
        ax.text(j, v * 0.95, f'{v:.4f}', ha='center', fontsize=10)
    
    # 2. RMSE对比图
    ax = axes[1]
    values = rmse_data[time_interval]
    bars = ax.bar(models, values, color=['skyblue', 'lightgreen', 'lightcoral', 'wheat', 'violet', 'lightpink'])
    bars[-2].set_color('mediumorchid')
    bars[-1].set_color('crimson')
    ax.set_title('均方根误差 (RMSE)', fontsize=14)
    ax.set_ylabel('RMSE', fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    
    # 添加数值标签
    for j, v in enumerate(values):
        ax.text(j, v * 0.95, f'{v:.4f}', ha='center', fontsize=10)
    
    # 3. ACC对比图
    ax = axes[2]
    values = acc_data[time_interval]
    bars = ax.bar(models, values, color=['skyblue', 'lightgreen', 'lightcoral', 'wheat', 'violet', 'lightpink'])
    bars[-2].set_color('mediumorchid')
    bars[-1].set_color('crimson')
    ax.set_title('准确率 (ACC)', fontsize=14)
    ax.set_ylabel('ACC', fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    
    # 添加数值标签
    for j, v in enumerate(values):
        ax.text(j, v * 1.05, f'{v:.4f}', ha='center', fontsize=10)
    
    # 4. R²对比图
    ax = axes[3]
    values = r2_data[time_interval]
    bars = ax.bar(models, values, color=['skyblue', 'lightgreen', 'lightcoral', 'wheat', 'violet', 'lightpink'])
    bars[-2].set_color('mediumorchid')
    bars[-1].set_color('crimson')
    ax.set_title('决定系数 (R²)', fontsize=14)
    ax.set_ylabel('R²', fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    
    # 添加数值标签
    for j, v in enumerate(values):
        ax.text(j, v * 1.05, f'{v:.4f}', ha='center', fontsize=10)
    
    # 5. Var对比图
    ax = axes[4]
    values = var_data[time_interval]
    bars = ax.bar(models, values, color=['skyblue', 'lightgreen', 'lightcoral', 'wheat', 'violet', 'lightpink'])
    bars[-2].set_color('mediumorchid')
    bars[-1].set_color('crimson')
    ax.set_title('方差解释率 (Var)', fontsize=14)
    ax.set_ylabel('Var', fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    
    # 添加数值标签
    for j, v in enumerate(values):
        ax.text(j, v * 1.05, f'{v:.4f}', ha='center', fontsize=10)
    
    # 6. 模型改进率对比
    ax = axes[5]
    # 计算ALSTM相对于其他模型的改进率
    improvement_rates = []
    for i, model in enumerate(models[:-1]):  # 除了ALSTM本身
        if model != 'AGRU':  # 不计算相对于AGRU的改进率
            # 使用MAE指标计算改进率
            improvement = (mae_data[time_interval][i] - mae_data[time_interval][-1]) / mae_data[time_interval][i] * 100
            improvement_rates.append(improvement)
        else:
            improvement_rates.append(0)  # AGRU位置放0
    
    # 添加ALSTM自身的位置（改进率为0）
    improvement_rates.append(0)
    
    # 只显示相对于HA, SVR, GRU, LSTM的改进率
    bars = ax.bar(models[:-1], improvement_rates[:-1], color=['skyblue', 'lightgreen', 'lightcoral', 'wheat', 'violet'])
    ax.set_title('ALSTM相对于其他模型的MAE改进率 (%)', fontsize=14)
    ax.set_ylabel('改进率 (%)', fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    
    # 添加数值标签
    for j, v in enumerate(improvement_rates[:-1]):
        if models[j] != 'AGRU':  # 不显示AGRU的标签
            ax.text(j, v * 1.05, f'{v:.2f}%', ha='center', fontsize=10)
    
    plt.suptitle('15分钟预测粒度下各模型性能综合对比', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig('comprehensive_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

# 绘制所有图表
def generate_all_charts():
    # 绘制各指标随时间变化的折线图
    plot_metric_by_time(mae_data, 'MAE', '平均绝对误差', is_higher_better=False)
    plot_metric_by_time(rmse_data, 'RMSE', '均方根误差', is_higher_better=False)
    plot_metric_by_time(acc_data, 'ACC', '准确率')
    plot_metric_by_time(r2_data, 'R²', '决定系数')
    plot_metric_by_time(var_data, 'Var', '方差解释率')
    
    # 绘制各时间间隔下模型性能对比的条形图
    plot_models_comparison_by_metric(mae_data, 'MAE', '平均绝对误差', is_higher_better=False)
    plot_models_comparison_by_metric(rmse_data, 'RMSE', '均方根误差', is_higher_better=False)
    plot_models_comparison_by_metric(acc_data, 'ACC', '准确率')
    plot_models_comparison_by_metric(r2_data, 'R²', '决定系数')
    plot_models_comparison_by_metric(var_data, 'Var', '方差解释率')
    
    # 绘制雷达图
    plot_radar_chart()
    
    # 绘制热力图
    plot_heatmap()
    
    # 绘制综合比较图
    plot_comprehensive_comparison()

# 主函数
if __name__ == "__main__":
    print("开始生成预测结果可视化图表...")
    generate_all_charts()
    print("所有图表生成完成！")
    print("生成的图表文件：")
    print("1. MAE_by_time.png - MAE随预测粒度的变化趋势")
    print("2. RMSE_by_time.png - RMSE随预测粒度的变化趋势")
    print("3. ACC_by_time.png - ACC随预测粒度的变化趋势")
    print("4. R²_by_time.png - R²随预测粒度的变化趋势")
    print("5. Var_by_time.png - Var随预测粒度的变化趋势")
    print("6. MAE_models_comparison.png - 不同预测粒度下各模型的MAE对比")
    print("7. RMSE_models_comparison.png - 不同预测粒度下各模型的RMSE对比")
    print("8. ACC_models_comparison.png - 不同预测粒度下各模型的ACC对比")
    print("9. R²_models_comparison.png - 不同预测粒度下各模型的R²对比")
    print("10. Var_models_comparison.png - 不同预测粒度下各模型的Var对比")
    print("11. model_performance_radar.png - 15分钟预测粒度下各模型性能雷达图")
    print("12. model_performance_heatmap.png - 不同预测粒度下各模型的综合性能得分热力图")
    print("13. comprehensive_comparison.png - 15分钟预测粒度下各模型性能综合对比")
    # 选择15分钟预测粒度的数据
    time_interval = 15
    
    # 创建子图
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    axes = axes.flatten()
    
    # 1. MAE对比图
    ax = axes[0]
    values = mae_data[time_interval]
    bars = ax.bar(models, values, color=['skyblue', 'lightgreen', 'lightcoral', 'wheat', 'violet', 'lightpink'])
    bars[-2].set_color('mediumorchid')
    bars[-1].set_color('crimson')
    ax.set_title('平均绝对误差 (MAE)', fontsize=14)
    ax.set_ylabel('MAE', fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    
    # 添加数值标签
    for j, v in enumerate(values):
        ax.text(j, v * 0.95, f'{v:.4f}', ha='center', fontsize=10)
    
    # 2. RMSE对比图
    ax = axes[1]
    values = rmse_data[time_interval]
    bars = ax.bar(models, values, color=['skyblue', 'lightgreen', 'lightcoral', 'wheat', 'violet', 'lightpink'])
    bars[-2].set_color('mediumorchid')
    bars[-1].set_color('crimson')
    ax.set_title('均方根误差 (RMSE)', fontsize=14)
    ax.set_ylabel('RMSE', fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    
    # 添加数值标签
    for j, v in enumerate(values):
        ax.text(j, v * 0.95, f'{v:.4f}', ha='center', fontsize=10)
    
    # 3. ACC对比图
    ax = axes[2]
    values = acc_data[time_interval]
    bars = ax.bar(models, values, color=['skyblue', 'lightgreen', 'lightcoral', 'wheat', 'violet', 'lightpink'])
    bars[-2].set_color('mediumorchid')
    bars[-1].set_color('crimson')
    ax.set_title('准确率 (ACC)', fontsize=14)
    ax.set_ylabel('ACC', fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    
    # 添加数值标签
    for j, v in enumerate(values):
        ax.text(j, v * 1.05, f'{v:.4f}', ha='center', fontsize=10)
    
    # 4. R²对比图
    ax = axes[3]
    values = r2_data[time_interval]
    bars = ax.bar(models, values, color=['skyblue', 'lightgreen', 'lightcoral', 'wheat', 'violet', 'lightpink'])
    bars[-2].set_color('mediumorchid')
    bars[-1].set_color('crimson')
    ax.set_title('决定系数 (R²)', fontsize=14)
    ax.set_ylabel('R²', fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    
    # 添加数值标签
    for j, v in enumerate(values):
        ax.text(j, v * 1.05, f'{v:.4f}', ha='center', fontsize=10)
    
    # 5. Var对比图
    ax = axes[4]
    values = var_data[time_interval]
    bars = ax.bar(models, values, color=['skyblue', 'lightgreen', 'lightcoral', 'wheat', 'violet', 'lightpink'])
    bars[-2].set_color('mediumorchid')
    bars[-1].set_color('crimson')
    ax.set_title('方差解释率 (Var)', fontsize=14)
    ax.set_ylabel('Var', fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    
    # 添加数值标签
    for j, v in enumerate(values):
        ax.text(j, v * 1.05, f'{v:.4f}', ha='center', fontsize=10)
    
    # 6. 模型改进率对比
    ax = axes[5]
    # 计算ALSTM相对于其他模型的改进率
    improvement_rates = []
    for i, model in enumerate(models[:-1]):  # 除了ALSTM本身
        if model != 'AGRU':  # 不计算相对于AGRU的改进率
            # 使用MAE指标计算改进率
            improvement = (mae_data[time_interval][i] - mae_data[time_interval][-1]) / mae_data[time_interval][i] * 100
            improvement_rates.append(improvement)
        else:
            improvement_rates.append(0)  # AGRU位置放0
    
    # 添加ALSTM自身的位置（改进率为0）
    improvement_rates.append(0)
    
    # 只显示相对于HA, SVR, GRU, LSTM的改进率
    bars = ax.bar(models[:-1], improvement_rates[:-1], color=['skyblue', 'lightgreen', 'lightcoral', 'wheat', 'violet'])
    ax.set_title('ALSTM相对于其他模型的MAE改进率 (%)', fontsize=14)
    ax.set_ylabel('改进率 (%)', fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    
    # 添加数值标签
    for j, v in enumerate(improvement_rates[:-1]):
        if models[j] != 'AGRU':  # 不显示AGRU的标签
            ax.text(j, v * 1.05, f'{v:.2f}%', ha='center', fontsize=10)
    
    plt.suptitle('15分钟预测粒度下各模型性能综合对比', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig('comprehensive_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()