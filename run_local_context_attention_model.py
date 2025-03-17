import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from local_context_attention_model import create_dataset, train_lc_attention_model, evaluate_model
import tensorflow as tf
import os
import traceback

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

def detect_anomalies(model, data, look_back=14, threshold=0.3):
    """
    使用预测结果和真实值之间的误差检测异常
    
    参数:
    model: 训练好的模型
    data: 测试数据
    look_back: 时间窗口大小
    threshold: 异常阈值，基于预测误差的标准差
    
    返回:
    含有异常标记的DataFrame
    """
    # 预测和实际值
    pred, actual = evaluate_model(model, data, look_back)
    
    # 计算误差
    errors = np.abs(pred.flatten() - actual)
    
    # 使用误差的标准差来设置动态阈值
    error_threshold = np.mean(errors) + threshold * np.std(errors)
    
    # 标记异常
    anomalies = errors > error_threshold
    
    # 创建结果DataFrame
    result_df = pd.DataFrame({
        'actual': actual,
        'predicted': pred.flatten(),
        'error': errors,
        'is_anomaly': anomalies
    })
    
    return result_df

def plot_results(result_df, title='网络流量预测结果'):
    """
    绘制预测结果和标记出的异常
    
    参数:
    result_df: 包含预测和异常标记的DataFrame
    title: 图表标题
    """
    plt.figure(figsize=(14, 7))
    
    # 绘制实际值和预测值
    plt.plot(result_df['actual'], label='实际流量')
    plt.plot(result_df['predicted'], label='预测流量', linestyle='--')
    
    # 标记异常点
    anomalies = result_df[result_df['is_anomaly']]
    if not anomalies.empty:
        plt.scatter(
            anomalies.index, 
            anomalies['actual'], 
            color='red', 
            s=50, 
            label='异常点'
        )
    
    plt.title(title)
    plt.xlabel('时间步')
    plt.ylabel('流量')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 保存图片
    plt.savefig('prediction_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def compare_models(data, look_back=14):
    """
    比较原始LSTM模型和局部上下文注意力模型
    
    参数:
    data: 数据集
    look_back: 时间窗口大小
    
    返回:
    两个模型的MSE和MAE
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
    
    # 添加早停回调
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='loss',
        patience=10,
        restore_best_weights=True
    )
    
    # 1. 训练原始LSTM模型
    print("正在训练原始LSTM模型...")
    try:
        # 创建标准LSTM模型
        original_model = tf.keras.Sequential([
            tf.keras.layers.LSTM(64, input_shape=(trainX.shape[1], trainX.shape[2]), return_sequences=True),
            tf.keras.layers.LSTM(32),
            tf.keras.layers.Dense(1)
        ])
        original_model.compile(loss='mean_squared_error', optimizer='sgd')
        original_model.fit(
            trainX, trainY, 
            epochs=100, 
            batch_size=16, 
            verbose=1,
            callbacks=[early_stopping]
        )
        
        # 在测试集上评估
        print("评估原始LSTM模型...")
        original_pred = original_model.predict(testX)
        original_mse = mean_squared_error(testY, original_pred)
        original_mae = mean_absolute_error(testY, original_pred)
        
        # 2. 训练局部上下文注意力模型
        print("\n正在训练局部上下文注意力模型...")
        print("这可能需要一些时间...")
        
        # 创建并训练局部上下文注意力模型
        attention_model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(trainX.shape[1], trainX.shape[2])),
            tf.keras.layers.LSTM(64, return_sequences=True),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True)),
            tf.keras.layers.Attention(),
            tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dense(1)
        ])
        
        attention_model.compile(loss='mean_squared_error', optimizer='adam')
        attention_history = attention_model.fit(
            trainX, trainY, 
            epochs=100, 
            batch_size=16, 
            verbose=1,
            callbacks=[early_stopping]
        )
        
        # 使用标准Keras的Attention作为替代方案
        print("评估基于标准Attention的模型...")
        attention_pred = attention_model.predict(testX)
        attention_mse = mean_squared_error(testY, attention_pred)
        attention_mae = mean_absolute_error(testY, attention_pred)
        
        # 3. 尝试带有注意力机制的GRU模型作为另一种方案
        print("\n训练带有注意力机制的GRU模型作为替代方案...")
        gru_attention_model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(trainX.shape[1], trainX.shape[2])),
            tf.keras.layers.GRU(64, return_sequences=True),
            tf.keras.layers.Bidirectional(tf.keras.layers.GRU(32, return_sequences=True)),
            tf.keras.layers.Attention(),
            tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dense(1)
        ])
        
        gru_attention_model.compile(loss='mean_squared_error', optimizer='adam')
        gru_attention_history = gru_attention_model.fit(
            trainX, trainY, 
            epochs=100, 
            batch_size=16, 
            verbose=1,
            callbacks=[early_stopping]
        )
        
        # 评估GRU-Attention模型
        print("评估GRU-Attention模型...")
        gru_attention_pred = gru_attention_model.predict(testX)
        gru_attention_mse = mean_squared_error(testY, gru_attention_pred)
        gru_attention_mae = mean_absolute_error(testY, gru_attention_pred)
        
        # 打印结果
        print("\n================模型比较结果================")
        print(f"原始LSTM模型 - MSE: {original_mse:.4f}, MAE: {original_mae:.4f}")
        print(f"标准Attention模型 - MSE: {attention_mse:.4f}, MAE: {attention_mae:.4f}")
        print(f"GRU-Attention模型 - MSE: {gru_attention_mse:.4f}, MAE: {gru_attention_mae:.4f}")
        
        # 计算改进率
        best_mse = min(attention_mse, gru_attention_mse)
        best_mae = min(attention_mae, gru_attention_mae)
        best_model_name = "标准Attention模型" if best_mse == attention_mse else "GRU-Attention模型"
        
        print(f"\n最佳模型: {best_model_name}")
        print(f"相比原始LSTM模型的改进率:")
        print(f"MSE: {(original_mse - best_mse) / original_mse * 100:.2f}%, MAE: {(original_mae - best_mae) / original_mae * 100:.2f}%")
        
        # 创建比较图
        plt.figure(figsize=(14, 10))
        
        # 实际值与预测值对比图
        plt.subplot(2, 1, 1)
        plt.plot(testY, label='实际流量')
        plt.plot(original_pred, label='原始LSTM预测', linestyle='--')
        plt.plot(attention_pred, label='标准Attention预测', linestyle='-.')
        plt.plot(gru_attention_pred, label='GRU-Attention预测', linestyle=':')
        plt.title('不同模型预测效果对比')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 误差对比图
        plt.subplot(2, 1, 2)
        plt.plot(np.abs(testY - original_pred.flatten()), label='原始LSTM误差')
        plt.plot(np.abs(testY - attention_pred.flatten()), label='标准Attention误差')
        plt.plot(np.abs(testY - gru_attention_pred.flatten()), label='GRU-Attention误差')
        plt.title('不同模型预测误差对比')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 返回最佳模型和指标
        if best_mse == attention_mse:
            best_model = attention_model
        else:
            best_model = gru_attention_model
            
        return {
            'best_model': best_model,
            'metrics': {
                'original': {'mse': original_mse, 'mae': original_mae},
                'attention': {'mse': attention_mse, 'mae': attention_mae},
                'gru_attention': {'mse': gru_attention_mse, 'mae': gru_attention_mae}
            }
        }
        
    except Exception as e:
        print(f"模型比较过程中发生错误: {e}")
        print("\n详细错误信息:")
        traceback.print_exc()
        return None

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
        
        # 比较模型
        print("正在比较模型性能...")
        result = compare_models(ip0, look_back=14)
        
        if result is not None:
            # 使用最佳模型检测异常
            print("正在使用最佳模型检测异常...")
            best_model = result['best_model']
            anomaly_results = detect_anomalies(best_model, ip0)
            
            # 绘制结果
            print("正在绘制结果...")
            plot_results(anomaly_results, title='基于注意力机制的网络流量预测')
            
            print("完成！模型比较和评估结果已保存为图像文件。")
        
    except Exception as e:
        print(f"发生错误: {e}")
        print("\n详细错误信息:")
        traceback.print_exc()

if __name__ == "__main__":
    main() 