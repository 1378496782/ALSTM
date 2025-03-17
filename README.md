# 基于局部上下文信息增强注意力机制的ALSTM网络流量预测模型

本项目实现了一个基于局部上下文信息增强注意力机制的ALSTM（Attention-enhanced LSTM）模型，严格按照论文中图3和图4所示的结构设计，用于网络流量预测和僵尸网络检测。

## 模型架构

ALSTM模型结合了以下几个关键技术：

1. **多LSTM串联结构**：按照论文图中的结构，使用多个LSTM单元串联，每个单元处理时序数据的不同方面
2. **局部上下文增强机制**：通过卷积层提取时间序列数据的局部上下文特征
3. **注意力机制**：基于提取的局部上下文信息计算注意力权重，增强模型对重要时间步的关注
4. **注意力权重计算**：严格按照论文中的公式(2)至公式(6)实现注意力计算过程

## 论文图对应关系

本实现严格遵循论文中的模型结构：

- **图3和图4 - ALSTM模型框架**：在`alstm_model.py`中通过`create_alstm_model`函数实现
  - 5个串联的LSTM单元
  - 每个LSTM单元后接局部上下文增强注意力机制
  - 最后通过Softmax层输出预测结果

## 文件说明

- `alstm_model.py`: 按照论文图实现的ALSTM模型架构
- `run_alstm_model.py`: 用于运行模型、与基准LSTM比较性能和检测异常的脚本
- `visualize_model.py`: 可视化模型结构，确保与论文图一致
- `cs448b_ipasn.csv`: 网络流量数据集

## 环境要求

- Python 3.6+
- TensorFlow 2.0+
- pandas
- numpy
- matplotlib
- scikit-learn
- pydot和graphviz (用于可视化模型结构)

## 使用方法

1. **安装依赖**

```bash
pip install tensorflow pandas numpy matplotlib scikit-learn pydot graphviz
```

2. **可视化模型结构**

```bash
python visualize_model.py
```

此命令会生成两个文件：
- `alstm_model_structure.png`: TensorFlow生成的模型结构图
- `alstm_architecture_diagram.png`: 自定义的与论文图风格一致的架构图

3. **运行ALSTM模型**

```bash
python run_alstm_model.py
```

该脚本将：
- 加载并预处理网络流量数据
- 训练并比较基准LSTM模型和ALSTM模型
- 使用ALSTM模型检测异常流量
- 生成可视化结果和比较图表

## 注意力机制实现

本项目按照论文中的公式实现了注意力计算机制：

1. 通过公式(4)计算查询向量Q
2. 通过公式(5)计算键向量K
3. 通过公式(6)和softmax函数计算注意力权重
4. 将权重应用于值向量V，得到上下文向量

## 模型优势

与传统LSTM模型相比，论文图中的ALSTM模型具有以下优势：

1. **更高的预测精度**：多LSTM单元和注意力机制能够更好地捕捉时序数据中的重要模式
2. **更好的异常检测能力**：通过关注局部上下文，能更准确地识别异常流量
3. **更好的可解释性**：注意力权重可以显示模型在预测过程中关注的时间步
4. **对局部模式的敏感性**：能够更好地捕捉时间序列中的局部变化

## 模型评估

本实现使用与论文一致的评估指标：
- 均方误差(MSE)
- 平均绝对误差(MAE)
- 改进率（与基准LSTM模型相比）

## 联系方式

如有问题或建议，请联系作者。