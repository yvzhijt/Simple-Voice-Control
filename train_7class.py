"""
语音指令识别 - 模型训练脚本
使用数据增强训练7类CNN语音识别模型
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import librosa
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ==================== 配置 ====================
# 数据路径
AUGMENTED_DATA = "./output_augmented"   # 增强后的指令数据
UNKNOWN_DATA = "./output_unknown_cmds"    # 无关/噪音数据

# 模型参数
N_CLASSES = 7              # 类别数：6个指令 + 1个未知
EPOCHS = 400               # 训练轮数
BATCH_SIZE = 32            # 批次大小
LEARNING_RATE = 0.001     # 学习率

# 标签映射
cmd_mapping = {
    "cmd01": 1,  # 打开
    "cmd02": 2,  # 关闭
    "cmd03": 3,  # 前进
    "cmd04": 4,  # 后退
    "cmd05": 5,  # 左转
    "cmd06": 6,  # 右转
    "cmd00": 0   # 未知
}

class_labels = ["0", "1", "2", "3", "4", "5", "6"]

# ==================== 数据处理 ====================

def extract_feature(file_path):
    """
    从音频文件提取MFCC特征
    
    Args:
        file_path: 音频文件路径
    
    Returns:
        MFCC特征矩阵 (时间步, 特征维度, 1)
    """
    signal, sr = librosa.load(file_path, sr=16000, duration=1.5)
    
    # 统一音频长度为1.5秒 (16000 * 1.5 = 24000)
    target_length = 24000
    if len(signal) < target_length:
        signal = np.pad(signal, (0, target_length - len(signal)))
    else:
        signal = signal[:target_length]
    
    # 音量归一化
    rms = np.sqrt(np.mean(signal**2))
    if rms > 0:
        signal = signal / rms * 0.1
    
    # 提取40维MFCC特征
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=40)
    mfcc = mfcc.T  # 转置: (时间步, 特征维度)
    mfcc = np.expand_dims(mfcc, axis=-1)  # 添加通道维度
    
    return mfcc

def get_features_and_labels():
    """
    加载并处理所有训练数据
    
    Returns:
        特征数组, 标签数组
    """
    print("提取音频特征...")
    features = []
    labels = []
    
    for cmd, label in cmd_mapping.items():
        print(f"\n处理 {cmd} (标签 {label}):")
        
        # 选择数据目录
        if cmd == "cmd00":
            folder = UNKNOWN_DATA
        else:
            folder = AUGMENTED_DATA
        
        if os.path.exists(folder):
            # 筛选对应类别的文件
            if cmd == "cmd00":
                files = [f for f in os.listdir(folder) if f.startswith("cmd00") and f.endswith(".wav")]
            else:
                files = [f for f in os.listdir(folder) if f.startswith(cmd) and f.endswith(".wav")]
            
            print(f"  数据: {len(files)}")
            
            # 提取特征
            for i, f in enumerate(files):
                try:
                    features.append(extract_feature(os.path.join(folder, f)))
                    labels.append(str(label))
                except Exception as e:
                    print(f"  错误: {f} - {e}")
                
                if (i + 1) % 200 == 0:
                    print(f"  已处理 {i+1}/{len(files)}")
    
    return np.array(features), np.array(labels)

# ==================== 模型定义 ====================

def create_model(input_shape, num_classes):
    """
    创建2D CNN模型
    
    Args:
        input_shape: 输入特征形状
        num_classes: 类别数量
    
    Returns:
        编译好的Keras模型
    """
    model = models.Sequential([
        # 输入层
        layers.Input(shape=input_shape),
        
        # 第一个卷积块
        layers.Conv2D(32, (3, 3), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),
        
        # 第二个卷积块
        layers.Conv2D(64, (3, 3), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.4),
        
        # 第三个卷积块
        layers.Conv2D(128, (3, 3), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.5),
        
        # 全连接层
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        
        # 输出层
        layers.Dense(num_classes, activation='softmax')
    ])
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# ==================== 主流程 ====================

def main():
    """主训练流程"""
    # 1. 加载数据
    X, y_labels = get_features_and_labels()
    
    print(f"\n📊 数据统计: {len(y_labels)} 个样本")
    for label in class_labels:
        count = len([l for l in y_labels if l == label])
        print(f"  命令 {label}: {count}")
    
    # 2. 标签编码
    le = LabelEncoder()
    le.fit(class_labels)
    y = to_categorical(le.transform(y_labels))
    
    # 3. 计算类别权重（处理数据不平衡）
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.arange(N_CLASSES),
        y=np.argmax(y, axis=1)
    )
    class_weights_dict = dict(enumerate(class_weights))
    print(f"\n⚖️ 类别权重: {class_weights_dict}")
    
    # 4. 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    input_shape = X_train.shape[1:]
    print(f"\n输入形状: {input_shape}")
    print(f"训练集: {X_train.shape[0]}, 测试集: {X_test.shape[0]}")
    
    # 5. 创建模型
    model = create_model(input_shape, N_CLASSES)
    model.summary()
    
    # 6. 设置回调函数
    early_stopping = EarlyStopping(
        monitor='val_loss', 
        patience=50,           # 验证loss 50轮不下降则停止
        restore_best_weights=True,
        verbose=1
    )
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.7,             # 学习率衰减因子
        patience=15,           # 15轮不下降则衰减
        min_lr=1e-6,           # 最小学习率
        verbose=1
    )
    
    # 7. 训练模型
    print("\n开始训练...")
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.2,
        verbose=2,
        class_weight=class_weights_dict,
        callbacks=[early_stopping, reduce_lr]
    )
    
    # 8. 评估模型
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n测试准确率: {test_acc:.4f}")
    
    # 9. 保存模型
    model.save('keyword_model_7class.h5')
    print("模型保存成功！")
    
    # 10. 保存标签编码器
    import pickle
    with open('label_encoder_7class.pkl', 'wb') as f:
        pickle.dump(le, f)
    print("标签编码器保存成功！")

if __name__ == "__main__":
    main()