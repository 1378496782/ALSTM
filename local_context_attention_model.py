import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, GRU, Input, Concatenate, Attention, LayerNormalization, Dropout
from tensorflow.keras.layers import TimeDistributed, RepeatVector, Reshape, Permute, Multiply, Lambda, Add
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer

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

class LocalContextAttention(Layer):
    """
    局部上下文信息增强注意力机制层
    """
    def __init__(self, hidden_units, window_size=3, **kwargs):
        self.hidden_units = hidden_units
        self.window_size = window_size
        super(LocalContextAttention, self).__init__(**kwargs)
        
    def build(self, input_shape):
        self.time_steps = input_shape[1]
        self.features = input_shape[2]
        
        # 定义所需的层
        self.conv1d = tf.keras.layers.Conv1D(
            filters=self.hidden_units, 
            kernel_size=self.window_size,
            padding='same',
            activation='relu'
        )
        
        self.query_dense = Dense(self.hidden_units)
        self.key_dense = Dense(self.hidden_units)
        self.value_dense = Dense(self.hidden_units)
        self.projection = Dense(self.hidden_units)
        
        self.add = Add()
        self.layer_norm = LayerNormalization()
        
        # 预先创建掩码，避免在call中创建
        if self.window_size < self.time_steps:
            mask = tf.ones((self.time_steps, self.time_steps))
            mask = tf.linalg.band_part(
                mask, self.window_size//2, self.window_size//2)
            self.attention_mask = tf.cast(mask, dtype=tf.bool)
            self.neg_inf_mask = tf.ones((self.time_steps, self.time_steps)) * (-1e9)
        else:
            self.attention_mask = None
            self.neg_inf_mask = None
        
        super(LocalContextAttention, self).build(input_shape)
    
    def call(self, x):
        batch_size = tf.shape(x)[0]
        
        # 1. 生成局部上下文信息
        local_context = self.conv1d(x)
        
        # 2. 计算注意力权重
        query = self.query_dense(x)  # 查询向量
        key = self.key_dense(local_context)  # 键向量
        
        # 计算注意力分数 - 使用Keras操作而非tf.matmul
        score = tf.keras.backend.batch_dot(
            query, key, axes=[2, 2]) / tf.sqrt(float(self.hidden_units))
        
        # 应用掩码以确保关注局部上下文（如果需要）
        if self.attention_mask is not None:
            # 将预先创建的掩码应用到注意力分数
            masked_score = tf.where(
                self.attention_mask, 
                score, 
                self.neg_inf_mask
            )
            # 注意力权重 - 使用Keras的softmax
            attention_weights = tf.keras.activations.softmax(masked_score, axis=-1)
        else:
            # 如果不需要掩码，直接使用softmax
            attention_weights = tf.keras.activations.softmax(score, axis=-1)
        
        # 3. 注意力输出 - 加权求和
        value = self.value_dense(x)  # 值向量
        context_vector = tf.keras.backend.batch_dot(
            attention_weights, value)
        
        # 4. 残差连接和层归一化
        x_projected = self.projection(x)
        output = self.add([context_vector, x_projected])
        output = self.layer_norm(output)
        
        return output
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.hidden_units)
    
    def get_config(self):
        config = super(LocalContextAttention, self).get_config()
        config.update({
            'hidden_units': self.hidden_units,
            'window_size': self.window_size
        })
        return config

def create_lc_attention_lstm_model(input_shape, lstm_units=64, gru_units=32):
    """
    创建基于局部上下文信息增强注意力机制的LSTM模型
    
    参数:
    input_shape: 输入数据形状，例如(look_back, feature_size)
    lstm_units: LSTM单元数量
    gru_units: GRU单元数量
    
    返回:
    Keras模型
    """
    # 定义模型输入
    inputs = Input(shape=input_shape)
    
    # 第一层LSTM
    lstm_out = LSTM(lstm_units, return_sequences=True)(inputs)
    
    # 应用局部上下文增强注意力机制
    attention_out = LocalContextAttention(lstm_units)(lstm_out)
    
    # 第二层可以选择GRU
    gru_out = GRU(gru_units)(attention_out)
    
    # 添加dropout以防止过拟合
    dropout_out = Dropout(0.2)(gru_out)
    
    # 输出层
    outputs = Dense(1)(dropout_out)
    
    # 创建模型
    model = Model(inputs=inputs, outputs=outputs)
    
    # 编译模型
    model.compile(
        loss='mean_squared_error', 
        optimizer='adam'
    )
    
    return model

def train_lc_attention_model(data, look_back=14, epochs=100, batch_size=16, verbose=1):
    """
    训练局部上下文注意力LSTM模型
    
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
    model = create_lc_attention_lstm_model(
        input_shape=(trainX.shape[1], trainX.shape[2]),
        lstm_units=64,
        gru_units=32
    )
    
    # 训练模型
    history = model.fit(
        trainX, trainY,
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose
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
    
    return testPredict, testY 