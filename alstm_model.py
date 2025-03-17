import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, LSTM, Input, Layer, Conv1D
from tensorflow.keras.layers import Activation, Concatenate, Multiply

class LocalContextAttentionLayer(Layer):
    """
    按照论文图3和图4实现的局部上下文增强注意力机制层
    """
    def __init__(self, units, **kwargs):
        self.units = units
        super(LocalContextAttentionLayer, self).__init__(**kwargs)
        
    def build(self, input_shape):
        # 创建Q、K、V计算所需的权重矩阵
        self.Q_dense = Dense(self.units)
        self.K_dense = Dense(self.units)
        self.V_dense = Dense(self.units)
        
        # 局部上下文增强卷积
        self.conv1d = Conv1D(filters=self.units, kernel_size=3, padding='same', activation='relu')
        
        # 最终的输出映射
        self.output_dense = Dense(self.units)
        
        super(LocalContextAttentionLayer, self).build(input_shape)
    
    def call(self, inputs):
        # 输入是LSTM的输出
        
        # 1. 通过卷积提取局部上下文特征
        context_features = self.conv1d(inputs)
        
        # 2. 计算Q、K、V
        Q = self.Q_dense(inputs)  # 论文公式(4)
        K = self.K_dense(context_features)  # 论文公式(5)
        V = self.V_dense(inputs)
        
        # 3. 计算注意力得分
        score = tf.matmul(Q, tf.transpose(K, [0, 2, 1]))
        score = score / tf.sqrt(tf.cast(self.units, tf.float32))
        
        # 4. 应用softmax得到注意力权重
        attention_weights = tf.nn.softmax(score, axis=-1)  # 论文公式(2)和(3)
        
        # 5. 加权求和得到上下文向量
        context_vector = tf.matmul(attention_weights, V)  # 论文公式(6)
        
        # 6. 最终输出
        output = self.output_dense(context_vector)
        
        return output
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.units)
    
    def get_config(self):
        config = super(LocalContextAttentionLayer, self).get_config()
        config.update({
            'units': self.units
        })
        return config

def create_alstm_model(sequence_length, features, lstm_units=64):
    """
    创建与论文图3和图4一致的ALSTM模型
    
    参数:
    sequence_length: 时间序列长度
    features: 特征数量
    lstm_units: LSTM单元数量
    
    返回:
    完整的ALSTM模型
    """
    # 定义输入
    inputs = Input(shape=(sequence_length, features))
    
    # 初始化LSTM状态和输出列表
    current_input = inputs
    lstm_outputs = []
    
    # 创建多个串联的LSTM单元，每个都有注意力机制
    # 根据图示创建5个LSTM单元串联
    for i in range(5):
        # LSTM处理
        lstm_out = LSTM(lstm_units, return_sequences=True)(current_input)
        lstm_outputs.append(lstm_out)
        
        # 局部上下文注意力机制处理
        # 注意：只有最后一个LSTM单元的输出会被最终使用
        if i < 4:
            attention_out = LocalContextAttentionLayer(lstm_units)(lstm_out)
            current_input = attention_out
    
    # 最后一个LSTM的输出通过注意力机制
    final_attention = LocalContextAttentionLayer(lstm_units)(lstm_outputs[-1])
    
    # 全局平均池化，确保输出是二维的
    global_avg_pool = tf.keras.layers.GlobalAveragePooling1D()(final_attention)
    
    # Softmax输出层（用于分类）或Dense（用于回归）
    output = Dense(1)(global_avg_pool)
    
    # 创建模型
    model = Model(inputs=inputs, outputs=output)
    
    # 编译模型
    model.compile(
        loss='mean_squared_error',
        optimizer='adam'
    )
    
    return model

def create_dataset(dataset, look_back=1):
    """
    创建时间序列数据集
    """
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back)].values
        dataX.append(a)
        dataY.append(dataset['f'].iloc[i + look_back])
    return np.array(dataX), np.array(dataY)

def train_alstm_model(data, look_back=14, epochs=100, batch_size=16, verbose=1):
    """
    训练ALSTM模型
    
    参数:
    data: 包含特征的DataFrame
    look_back: 时间窗口大小
    epochs: 训练轮数
    batch_size: 批次大小
    verbose: 显示模式
    
    返回:
    训练好的模型和训练历史
    """
    # 确保数据类型正确
    data['f'] = data['f'].astype('float32')
    
    # 准备训练数据
    train = data[0:look_back*5].copy()
    trainX, trainY = create_dataset(train, look_back)
    trainX = np.reshape(trainX, (trainX.shape[0], look_back, trainX.shape[2]))
    
    # 创建并训练模型
    model = create_alstm_model(
        sequence_length=trainX.shape[1], 
        features=trainX.shape[2],
        lstm_units=64
    )
    
    # 添加早停机制
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='loss',
        patience=10,
        restore_best_weights=True
    )
    
    # 训练模型
    history = model.fit(
        trainX, trainY,
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose,
        callbacks=[early_stopping]
    )
    
    return model, history

def evaluate_model(model, data, look_back=14):
    """
    评估模型性能
    
    参数:
    model: 训练好的模型
    data: 测试数据
    look_back: 时间窗口大小
    
    返回:
    预测结果和实际值
    """
    # 确保数据类型正确
    data['f'] = data['f'].astype('float32')
    
    # 准备测试数据
    testX, testY = create_dataset(data, look_back)
    testX = np.reshape(testX, (testX.shape[0], look_back, testX.shape[2]))
    
    # 进行预测
    testPredict = model.predict(testX)
    
    # 确保预测结果是二维的
    if len(testPredict.shape) > 2:
        testPredict = testPredict.reshape(testPredict.shape[0], -1)
    
    return testPredict, testY 