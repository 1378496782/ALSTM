import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.utils import plot_model
from alstm_model import create_alstm_model

# 设置matplotlib中文字体支持
plt.rcParams['font.sans-serif'] = ['Hiragino Sans GB', 'SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False  # 坐标轴负数的负号显示

def visualize_alstm_model():
    """
    可视化ALSTM模型结构，确保与论文图保持一致
    """
    # 创建模型
    model = create_alstm_model(sequence_length=14, features=2, lstm_units=64)
    
    # 显示模型摘要
    model.summary()
    
    # 使用plot_model绘制模型结构图
    try:
        # 保存模型结构图到文件
        plot_model(
            model,
            to_file='alstm_model_structure.png',
            show_shapes=True,
            show_layer_names=True,
            rankdir='TB',  # 'TB' for vertical, 'LR' for horizontal
            dpi=96,
            expand_nested=True
        )
        print("模型结构图已保存为 'alstm_model_structure.png'")
    except Exception as e:
        print(f"绘制模型结构图时发生错误: {e}")
        print("可能需要安装graphviz。请运行: pip install pydot graphviz")

def create_alstm_architecture_diagram():
    """
    创建与论文图像风格一致的ALSTM模型架构图
    """
    # 创建画布
    plt.figure(figsize=(12, 8))
    
    # 定义组件位置
    input_positions = [(i*2, 9) for i in range(5)]
    lstm_positions = [(i*2, 7) for i in range(5)]
    attention_positions = [(i*2, 5) for i in range(5)]
    softmax_position = (4, 3)
    output_position = (4, 1)
    
    # 绘制输入节点
    for i, pos in enumerate(input_positions):
        plt.plot(pos[0], pos[1], 'o', markersize=15, color='white', markeredgecolor='black')
        plt.text(pos[0], pos[1]-0.4, f'Input {i+1}', ha='center')
    
    # 绘制LSTM组件
    for i, pos in enumerate(lstm_positions):
        rect = plt.Rectangle((pos[0]-0.7, pos[1]-0.7), 1.4, 1.4, fill=True, color='lightblue', alpha=0.7)
        plt.gca().add_patch(rect)
        plt.text(pos[0], pos[1], f'LSTM {i+1}', ha='center', va='center')
    
    # 绘制注意力组件
    for i, pos in enumerate(attention_positions):
        rect = plt.Rectangle((pos[0]-0.7, pos[1]-0.7), 1.4, 1.4, fill=True, color='lightcoral', alpha=0.7)
        plt.gca().add_patch(rect)
        plt.text(pos[0], pos[1], f'Attention {i+1}', ha='center', va='center')
    
    # 绘制Softmax组件
    rect = plt.Rectangle((softmax_position[0]-2, softmax_position[1]-0.7), 4, 1.4, fill=True, color='lightgreen', alpha=0.7)
    plt.gca().add_patch(rect)
    plt.text(softmax_position[0], softmax_position[1], 'Softmax', ha='center', va='center')
    
    # 绘制输出节点
    plt.plot(output_position[0], output_position[1], 'o', markersize=15, color='white', markeredgecolor='black')
    plt.text(output_position[0], output_position[1]-0.4, 'Output', ha='center')
    
    # 绘制连接线 - 输入到LSTM（变细的箭头）
    for i in range(5):
        plt.arrow(input_positions[i][0], input_positions[i][1]-0.5, 0, -0.8, 
                 head_width=0.05, head_length=0.1, fc='black', ec='black', linewidth=0.5)
    
    # 绘制连接线 - LSTM到注意力（变细的箭头）
    for i in range(5):
        plt.arrow(lstm_positions[i][0], lstm_positions[i][1]-0.8, 0, -0.8, 
                 head_width=0.05, head_length=0.1, fc='black', ec='black', linewidth=0.5)
    
    # 绘制连接线 - 注意力到Softmax（变细的箭头）
    for i in range(5):
        plt.arrow(attention_positions[i][0], attention_positions[i][1]-0.8, 
                 softmax_position[0]-attention_positions[i][0], softmax_position[1]-(attention_positions[i][1]-0.8), 
                 head_width=0.05, head_length=0.1, fc='black', ec='black', linewidth=0.5)
    
    # 绘制连接线 - Softmax到输出（变细的箭头）
    plt.arrow(softmax_position[0], softmax_position[1]-0.8, 0, -0.8, 
             head_width=0.05, head_length=0.1, fc='black', ec='black', linewidth=0.5)
    
    # 设置图表布局
    plt.title('ALSTM模型架构图')
    plt.xlim(-1, 10)
    plt.ylim(0, 10)
    plt.axis('off')
    
    # 保存图表
    plt.savefig('alstm_architecture_diagram.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("自定义ALSTM架构图已保存为 'alstm_architecture_diagram.png'")

if __name__ == "__main__":
    print("可视化ALSTM模型结构...")
    visualize_alstm_model()
    
    print("\n创建自定义ALSTM架构图...")
    create_alstm_architecture_diagram() 