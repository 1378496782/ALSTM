import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import tensorflow as tf
from tensorflow.keras.layers import Input
from alstm_model import create_dataset, create_alstm_model
import os
import traceback
import time

# 设置matplotlib中文字体支持
plt.rcParams['font.sans-serif'] = ['Hiragino Sans GB', 'SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False  # 坐标轴负数的负号显示

def load_and_preprocess_data(csv_file):
    """
    加载并预处理网络流量数据
    """
    # 读取CSV文件
    df = pd.read_csv(csv_file)
    
    # 转换日期
    df['date'] = pd.to_datetime(df['date'])
    
    # 按日期和IP分组
    df = df.groupby(['date', 'l_ipn'], as_index=False).sum()
    
    # 添加时间特征
    df['yday'] = df['date'].dt.dayofyear
    df['wday'] = df['date'].dt.dayofweek
    
    # 返回处理好的数据
    return df

def create_agru_model(sequence_length, features, gru_units=64):
    """
    创建基于局部上下文信息增强注意力机制的GRU模型
    
    参数:
    sequence_length: 时间序列长度
    features: 特征数量
    gru_units: GRU单元数量
    
    返回:
    完整的AGRU模型
    """
    # 定义输入
    inputs = Input(shape=(sequence_length, features))
    
    # 初始化GRU状态和输出列表
    current_input = inputs
    gru_outputs = []
    
    # 创建多个串联的GRU单元，每个都有注意力机制
    # 根据图示创建5个GRU单元串联
    for i in range(5):
        # GRU处理
        gru_out = tf.keras.layers.GRU(gru_units, return_sequences=True)(current_input)
        gru_outputs.append(gru_out)
        
        # 局部上下文注意力机制处理
        # 只有最后一个GRU单元的输出会被最终使用
        if i < 4:
            # 使用1D卷积提取局部特征
            context_features = tf.keras.layers.Conv1D(filters=gru_units, kernel_size=3, padding='same', activation='relu')(gru_out)
            
            # 注意力计算
            query = tf.keras.layers.Dense(gru_units)(gru_out)
            key = tf.keras.layers.Dense(gru_units)(context_features)
            value = tf.keras.layers.Dense(gru_units)(gru_out)
            
            # 计算注意力得分 - 使用Keras层
            # 使用Keras的Dot层而不是tf.matmul
            score = tf.keras.layers.Dot(axes=[2, 2])([query, key])
            # 缩放
            scale_factor = float(gru_units ** 0.5)
            score = tf.keras.layers.Lambda(lambda x: x / scale_factor)(score)
            attention_weights = tf.keras.layers.Softmax(axis=-1)(score)
            
            # 加权上下文 - 使用Keras的Dot层
            context_vector = tf.keras.layers.Dot(axes=[2, 1])([attention_weights, value])
            
            # 输出投影
            attention_out = tf.keras.layers.Dense(gru_units)(context_vector)
            current_input = attention_out
    
    # 最后一个GRU的输出通过注意力机制
    # 使用1D卷积提取局部特征
    context_features = tf.keras.layers.Conv1D(filters=gru_units, kernel_size=3, padding='same', activation='relu')(gru_outputs[-1])
    
    # 注意力计算
    query = tf.keras.layers.Dense(gru_units)(gru_outputs[-1])
    key = tf.keras.layers.Dense(gru_units)(context_features)
    value = tf.keras.layers.Dense(gru_units)(gru_outputs[-1])
    
    # 计算注意力得分 - 使用Keras层
    # 使用Keras的Dot层而不是tf.matmul
    score = tf.keras.layers.Dot(axes=[2, 2])([query, key])
    # 缩放
    scale_factor = float(gru_units ** 0.5)
    score = tf.keras.layers.Lambda(lambda x: x / scale_factor)(score)
    attention_weights = tf.keras.layers.Softmax(axis=-1)(score)
    
    # 加权上下文 - 使用Keras的Dot层
    context_vector = tf.keras.layers.Dot(axes=[2, 1])([attention_weights, value])
    
    # 输出投影
    final_attention = tf.keras.layers.Dense(gru_units)(context_vector)
    
    # 全局平均池化，确保输出是二维的
    global_avg_pool = tf.keras.layers.GlobalAveragePooling1D()(final_attention)
    
    # 输出层
    output = tf.keras.layers.Dense(1)(global_avg_pool)
    
    # 创建模型
    model = tf.keras.models.Model(inputs=inputs, outputs=output)
    
    # 编译模型
    model.compile(
        loss='mean_squared_error',
        optimizer='adam'
    )
    
    return model

def historical_average(data, window_size=14):
    """
    历史平均模型 (HA)
    
    参数:
    data: 包含特征的DataFrame
    window_size: 用于平均的历史窗口大小
    
    返回:
    预测结果和实际值
    """
    # 确保数据类型正确
    data['f'] = data['f'].astype('float32')
    
    # 准备数据
    values = data['f'].values
    predictions = []
    actuals = []
    
    for i in range(window_size, len(values)):
        # 使用过去window_size个值的平均作为预测
        prediction = np.mean(values[i-window_size:i])
        predictions.append(prediction)
        actuals.append(values[i])
    
    return np.array(predictions), np.array(actuals)

def svr_model(data, window_size=14):
    """
    支持向量回归模型 (SVR)
    
    参数:
    data: 包含特征的DataFrame
    window_size: 时间窗口大小
    
    返回:
    预测结果和实际值
    """
    # 确保数据类型正确
    data['f'] = data['f'].astype('float32')
    
    # 准备训练数据
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data['f'].iloc[i:i+window_size].values)
        y.append(data['f'].iloc[i+window_size])
    
    X = np.array(X)
    y = np.array(y)
    
    # 划分训练集和测试集
    train_size = int(len(X) * 0.7)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # 标准化数据
    scaler_X = StandardScaler()
    X_train = scaler_X.fit_transform(X_train)
    X_test = scaler_X.transform(X_test)
    
    # 训练SVR模型
    svr = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
    svr.fit(X_train, y_train)
    
    # 预测
    predictions = svr.predict(X_test)
    
    return predictions, y_test

def train_basic_model(model_type, trainX, trainY, testX, testY):
    """
    训练基础深度学习模型 (LSTM或GRU)
    
    参数:
    model_type: 'lstm'或'gru'
    trainX, trainY: 训练数据
    testX, testY: 测试数据
    
    返回:
    训练好的模型和预测结果
    """
    # 添加早停回调
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='loss',
        patience=10,
        restore_best_weights=True
    )
    
    # 创建模型
    if model_type == 'lstm':
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(64, input_shape=(trainX.shape[1], trainX.shape[2]), return_sequences=True),
            tf.keras.layers.LSTM(32),
            tf.keras.layers.Dense(1)
        ])
    else:  # gru
        model = tf.keras.Sequential([
            tf.keras.layers.GRU(64, input_shape=(trainX.shape[1], trainX.shape[2]), return_sequences=True),
            tf.keras.layers.GRU(32),
            tf.keras.layers.Dense(1)
        ])
    
    model.compile(loss='mean_squared_error', optimizer='adam')
    
    # 训练模型
    model.fit(
        trainX, trainY, 
        epochs=100, 
        batch_size=16, 
        verbose=1,
        callbacks=[early_stopping]
    )
    
    # 预测
    predictions = model.predict(testX)
    
    return model, predictions

def train_and_evaluate_all_models(data, look_back=14):
    """
    训练和评估所有模型
    
    参数:
    data: 数据集
    look_back: 时间窗口大小
    
    返回:
    所有模型的评估结果
    """
    # 划分数据
    data['f'] = data['f'].astype('float32')
    train_data = data[0:look_back*5].copy()
    test_data = data[look_back*5:].copy()
    
    # 准备训练和测试数据
    trainX, trainY = create_dataset(train_data, look_back)
    trainX = np.reshape(trainX, (trainX.shape[0], look_back, trainX.shape[2]))
    
    testX, testY = create_dataset(test_data, look_back)
    testX = np.reshape(testX, (testX.shape[0], look_back, testX.shape[2]))
    
    results = {}
    training_times = {}
    
    try:
        # 1. 历史平均模型 (HA)
        print("评估历史平均模型 (HA)...")
        start_time = time.time()
        ha_pred, ha_actual = historical_average(data, look_back)
        training_times['HA'] = time.time() - start_time
        
        # 计算指标
        ha_mse = mean_squared_error(ha_actual, ha_pred)
        ha_mae = mean_absolute_error(ha_actual, ha_pred)
        ha_r2 = r2_score(ha_actual, ha_pred)
        
        results['HA'] = {
            'predictions': ha_pred,
            'actual': ha_actual,
            'mse': ha_mse,
            'mae': ha_mae,
            'r2': ha_r2
        }
        
        # 2. 支持向量回归模型 (SVR)
        print("训练和评估SVR模型...")
        start_time = time.time()
        svr_pred, svr_actual = svr_model(data, look_back)
        training_times['SVR'] = time.time() - start_time
        
        # 计算指标
        svr_mse = mean_squared_error(svr_actual, svr_pred)
        svr_mae = mean_absolute_error(svr_actual, svr_pred)
        svr_r2 = r2_score(svr_actual, svr_pred)
        
        results['SVR'] = {
            'predictions': svr_pred,
            'actual': svr_actual,
            'mse': svr_mse,
            'mae': svr_mae,
            'r2': svr_r2
        }
        
        # 3. LSTM模型
        print("训练和评估LSTM模型...")
        start_time = time.time()
        lstm_model, lstm_pred = train_basic_model('lstm', trainX, trainY, testX, testY)
        training_times['LSTM'] = time.time() - start_time
        
        # 计算指标
        lstm_mse = mean_squared_error(testY, lstm_pred)
        lstm_mae = mean_absolute_error(testY, lstm_pred)
        lstm_r2 = r2_score(testY, lstm_pred)
        
        results['LSTM'] = {
            'model': lstm_model,
            'predictions': lstm_pred,
            'actual': testY,
            'mse': lstm_mse,
            'mae': lstm_mae,
            'r2': lstm_r2
        }
        
        # 4. GRU模型
        print("训练和评估GRU模型...")
        start_time = time.time()
        gru_model, gru_pred = train_basic_model('gru', trainX, trainY, testX, testY)
        training_times['GRU'] = time.time() - start_time
        
        # 计算指标
        gru_mse = mean_squared_error(testY, gru_pred)
        gru_mae = mean_absolute_error(testY, gru_pred)
        gru_r2 = r2_score(testY, gru_pred)
        
        results['GRU'] = {
            'model': gru_model,
            'predictions': gru_pred,
            'actual': testY,
            'mse': gru_mse,
            'mae': gru_mae,
            'r2': gru_r2
        }
        
        # 5. ALSTM模型
        print("训练和评估ALSTM模型...")
        start_time = time.time()
        
        # 创建并训练ALSTM模型
        alstm_model = create_alstm_model(
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
        
        alstm_model.fit(
            trainX, trainY,
            epochs=100,
            batch_size=16,
            verbose=1,
            callbacks=[early_stopping]
        )
        
        training_times['ALSTM'] = time.time() - start_time
        
        # 预测
        alstm_pred = alstm_model.predict(testX)
        
        # 确保预测结果是二维的
        if len(alstm_pred.shape) > 2:
            alstm_pred = alstm_pred.reshape(alstm_pred.shape[0], -1)
        
        # 计算指标
        alstm_mse = mean_squared_error(testY, alstm_pred)
        alstm_mae = mean_absolute_error(testY, alstm_pred)
        alstm_r2 = r2_score(testY, alstm_pred)
        
        results['ALSTM'] = {
            'model': alstm_model,
            'predictions': alstm_pred,
            'actual': testY,
            'mse': alstm_mse,
            'mae': alstm_mae,
            'r2': alstm_r2
        }
        
        # 6. AGRU模型
        print("训练和评估AGRU模型...")
        start_time = time.time()
        
        # 创建并训练AGRU模型
        agru_model = create_agru_model(
            sequence_length=trainX.shape[1], 
            features=trainX.shape[2],
            gru_units=64
        )
        
        agru_model.fit(
            trainX, trainY,
            epochs=100,
            batch_size=16,
            verbose=1,
            callbacks=[early_stopping]
        )
        
        training_times['AGRU'] = time.time() - start_time
        
        # 预测
        agru_pred = agru_model.predict(testX)
        
        # 确保预测结果是二维的
        if len(agru_pred.shape) > 2:
            agru_pred = agru_pred.reshape(agru_pred.shape[0], -1)
        
        # 计算指标
        agru_mse = mean_squared_error(testY, agru_pred)
        agru_mae = mean_absolute_error(testY, agru_pred)
        agru_r2 = r2_score(testY, agru_pred)
        
        results['AGRU'] = {
            'model': agru_model,
            'predictions': agru_pred,
            'actual': testY,
            'mse': agru_mse,
            'mae': agru_mae,
            'r2': agru_r2
        }
        
        # 打印所有模型的评估结果
        print("\n================所有模型评估结果================")
        for model_name, metrics in results.items():
            print(f"{model_name} - MSE: {metrics['mse']:.4f}, MAE: {metrics['mae']:.4f}, R²: {metrics['r2']:.4f}, 训练时间: {training_times.get(model_name, 'N/A'):.2f}秒")
        
        # 计算ALSTM和AGRU相对于其他模型的改进率
        print("\nALSTM和AGRU的性能改进率:")
        for model_name, metrics in results.items():
            if model_name not in ['ALSTM', 'AGRU']:
                alstm_improvement_mse = (metrics['mse'] - results['ALSTM']['mse']) / metrics['mse'] * 100
                alstm_improvement_mae = (metrics['mae'] - results['ALSTM']['mae']) / metrics['mae'] * 100
                
                agru_improvement_mse = (metrics['mse'] - results['AGRU']['mse']) / metrics['mse'] * 100
                agru_improvement_mae = (metrics['mae'] - results['AGRU']['mae']) / metrics['mae'] * 100
                
                print(f"相对于{model_name}:")
                print(f"  ALSTM - MSE: {alstm_improvement_mse:.2f}%, MAE: {alstm_improvement_mae:.2f}%")
                print(f"  AGRU  - MSE: {agru_improvement_mse:.2f}%, MAE: {agru_improvement_mae:.2f}%")
        
        # 返回结果
        return {
            'results': results,
            'training_times': training_times
        }
        
    except Exception as e:
        print(f"模型训练和评估过程中发生错误: {e}")
        print("\n详细错误信息:")
        traceback.print_exc()
        return None

def plot_model_comparison(results):
    """
    绘制模型比较图表
    
    参数:
    results: 包含所有模型结果的字典
    """
    if not results:
        print("没有可用的结果来绘制图表")
        return
    
    # 获取结果数据
    model_results = results['results']
    training_times = results['training_times']
    
    # 提取指标，确保使用相同的测试集
    models = []
    mse_values = []
    mae_values = []
    r2_values = []
    times = []
    
    for model_name, metrics in model_results.items():
        models.append(model_name)
        mse_values.append(metrics['mse'])
        mae_values.append(metrics['mae'])
        r2_values.append(metrics['r2'])
        times.append(training_times.get(model_name, 0))
    
    # 1. 绘制MSE比较图
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    bars = plt.bar(models, mse_values, color=['skyblue', 'lightgreen', 'lightcoral', 'wheat', 'violet', 'lightpink'])
    
    # 高亮ALSTM和AGRU
    bars[-2].set_color('mediumorchid')
    bars[-1].set_color('crimson')
    
    plt.title('均方误差 (MSE) 比较')
    plt.ylabel('MSE')
    plt.grid(axis='y', alpha=0.3)
    
    # 添加数值标签
    for i, v in enumerate(mse_values):
        plt.text(i, v * 1.05, f'{v:.4f}', ha='center')
    
    # 2. 绘制MAE比较图
    plt.subplot(2, 2, 2)
    bars = plt.bar(models, mae_values, color=['skyblue', 'lightgreen', 'lightcoral', 'wheat', 'violet', 'lightpink'])
    
    # 高亮ALSTM和AGRU
    bars[-2].set_color('mediumorchid')
    bars[-1].set_color('crimson')
    
    plt.title('平均绝对误差 (MAE) 比较')
    plt.ylabel('MAE')
    plt.grid(axis='y', alpha=0.3)
    
    # 添加数值标签
    for i, v in enumerate(mae_values):
        plt.text(i, v * 1.05, f'{v:.4f}', ha='center')
    
    # 3. 绘制R²比较图
    plt.subplot(2, 2, 3)
    bars = plt.bar(models, r2_values, color=['skyblue', 'lightgreen', 'lightcoral', 'wheat', 'violet', 'lightpink'])
    
    # 高亮ALSTM和AGRU
    bars[-2].set_color('mediumorchid')
    bars[-1].set_color('crimson')
    
    plt.title('决定系数 (R²) 比较')
    plt.ylabel('R²')
    plt.grid(axis='y', alpha=0.3)
    
    # 添加数值标签
    for i, v in enumerate(r2_values):
        plt.text(i, max(0, v * 1.05), f'{v:.4f}', ha='center')
    
    # 4. 绘制训练时间比较图
    plt.subplot(2, 2, 4)
    bars = plt.bar(models, times, color=['skyblue', 'lightgreen', 'lightcoral', 'wheat', 'violet', 'lightpink'])
    
    # 高亮ALSTM和AGRU
    bars[-2].set_color('mediumorchid')
    bars[-1].set_color('crimson')
    
    plt.title('训练时间比较 (秒)')
    plt.ylabel('时间 (秒)')
    plt.grid(axis='y', alpha=0.3)
    
    # 添加数值标签
    for i, v in enumerate(times):
        plt.text(i, v * 1.05, f'{v:.2f}', ha='center')
    
    plt.tight_layout()
    plt.savefig('model_metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 5. 绘制预测结果比较图
    # 选择一部分测试数据进行可视化
    sample_size = 100
    plt.figure(figsize=(15, 10))
    
    # 确保我们使用相同的样本
    reference_actual = model_results['LSTM']['actual'][:sample_size]
    x_axis = range(len(reference_actual))
    
    plt.plot(x_axis, reference_actual, 'k-', label='实际值', linewidth=2)
    
    # 为简洁起见，仅绘制LSTM、GRU、ALSTM和AGRU的预测
    plt.plot(x_axis, model_results['LSTM']['predictions'][:sample_size], 'b--', label='LSTM预测', alpha=0.7)
    plt.plot(x_axis, model_results['GRU']['predictions'][:sample_size], 'g--', label='GRU预测', alpha=0.7)
    plt.plot(x_axis, model_results['ALSTM']['predictions'][:sample_size], 'r-', label='ALSTM预测', linewidth=2)
    plt.plot(x_axis, model_results['AGRU']['predictions'][:sample_size], 'm-', label='AGRU预测', linewidth=2)
    
    plt.title('模型预测结果比较')
    plt.xlabel('时间步')
    plt.ylabel('流量')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('model_predictions_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 6. 绘制ALSTM和AGRU的改进率
    plt.figure(figsize=(12, 6))
    
    # 计算相对于其他模型的改进率
    improvement_models = [m for m in models if m not in ['ALSTM', 'AGRU']]
    alstm_improvement_mse = []
    alstm_improvement_mae = []
    agru_improvement_mse = []
    agru_improvement_mae = []
    
    for model in improvement_models:
        alstm_improvement_mse.append((model_results[model]['mse'] - model_results['ALSTM']['mse']) / model_results[model]['mse'] * 100)
        alstm_improvement_mae.append((model_results[model]['mae'] - model_results['ALSTM']['mae']) / model_results[model]['mae'] * 100)
        agru_improvement_mse.append((model_results[model]['mse'] - model_results['AGRU']['mse']) / model_results[model]['mse'] * 100)
        agru_improvement_mae.append((model_results[model]['mae'] - model_results['AGRU']['mae']) / model_results[model]['mae'] * 100)
    
    width = 0.2
    x = np.arange(len(improvement_models))
    
    plt.subplot(1, 2, 1)
    plt.bar(x - width/2, alstm_improvement_mse, width, label='ALSTM', color='mediumorchid')
    plt.bar(x + width/2, agru_improvement_mse, width, label='AGRU', color='crimson')
    
    plt.title('MSE改进率 (%)')
    plt.xticks(x, improvement_models)
    plt.ylabel('改进率 (%)')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    # 添加数值标签
    for i, v in enumerate(alstm_improvement_mse):
        plt.text(i - width/2, v * 1.05, f'{v:.1f}%', ha='center')
    for i, v in enumerate(agru_improvement_mse):
        plt.text(i + width/2, v * 1.05, f'{v:.1f}%', ha='center')
    
    plt.subplot(1, 2, 2)
    plt.bar(x - width/2, alstm_improvement_mae, width, label='ALSTM', color='mediumorchid')
    plt.bar(x + width/2, agru_improvement_mae, width, label='AGRU', color='crimson')
    
    plt.title('MAE改进率 (%)')
    plt.xticks(x, improvement_models)
    plt.ylabel('改进率 (%)')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    # 添加数值标签
    for i, v in enumerate(alstm_improvement_mae):
        plt.text(i - width/2, v * 1.05, f'{v:.1f}%', ha='center')
    for i, v in enumerate(agru_improvement_mae):
        plt.text(i + width/2, v * 1.05, f'{v:.1f}%', ha='center')
    
    plt.tight_layout()
    plt.savefig('alstm_agru_improvement.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("所有比较图已保存")

def main():
    # 确保TensorFlow使用GPU（如果可用）
    try:
        physical_devices = tf.config.list_physical_devices('GPU')
        if len(physical_devices) > 0:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
            print(f"使用GPU: {physical_devices[0]}")
        else:
            print("未检测到GPU，使用CPU进行训练")
    except Exception as e:
        print(f"设置GPU出错: {e}")
        print("将使用默认设备进行训练")
    
    # 设置TensorFlow日志级别
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=全部，1=INFO，2=WARNING，3=ERROR
    
    # 数据文件路径
    csv_file = 'cs448b_ipasn.csv'
    
    try:
        # 加载数据
        print("正在加载和预处理数据...")
        df = load_and_preprocess_data(csv_file)
        
        # 选择一个IP进行分析
        unique_ips = df['l_ipn'].unique()
        ip0 = df[df['l_ipn'] == unique_ips[0]][['f', 'wday']].copy()
        
        # 训练和评估所有模型
        print("开始训练和评估所有模型...")
        results = train_and_evaluate_all_models(ip0, look_back=14)
        
        if results:
            # 绘制模型比较图表
            print("正在绘制模型比较图表...")
            plot_model_comparison(results)
            
            print("完成！所有模型比较结果已保存为图像文件。")
        
    except Exception as e:
        print(f"发生错误: {e}")
        print("\n详细错误信息:")
        traceback.print_exc()

if __name__ == "__main__":
    main() 