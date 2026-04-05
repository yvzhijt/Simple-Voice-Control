"""
语音指令识别 - 模型量化
将训练好的模型转换为 TFLite 格式，支持 INT8 量化
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import models
import pickle

# ==================== 配置 ====================
MODEL_PATH = 'keyword_model_7class.h5'
OUTPUT_PATH = 'keyword_model_quantized.tflite'

def representative_dataset():
    """
    生成代表性数据集用于量化
    需要提供至少100个样本
    """
    # 加载模型和数据
    model = models.load_model(MODEL_PATH)
    
    with open('label_encoder_7class.pkl', 'rb') as f:
        le = pickle.load(f)
    
    # 加载一些训练数据作为代表
    data_dir = "./output_augmented"
    
    if not os.path.exists(data_dir):
        print("⚠️ 数据目录不存在，使用随机数据")
        for _ in range(100):
            yield [np.random.rand(1, 47, 40, 1).astype(np.float32)]
        return
    
    # 收集样本
    files = []
    for f in os.listdir(data_dir):
        if f.endswith('.wav'):
            files.append(os.path.join(data_dir, f))
            if len(files) >= 100:
                break
    
    print(f"使用 {len(files)} 个样本进行量化...")
    
    for file_path in files:
        try:
            import librosa
            
            signal, sr = librosa.load(file_path, sr=16000, duration=1.5)
            
            target_length = 24000
            if len(signal) < target_length:
                signal = np.pad(signal, (0, target_length - len(signal)))
            else:
                signal = signal[:target_length]
            
            rms = np.sqrt(np.mean(signal**2))
            if rms > 0:
                signal = signal / rms * 0.1
            
            mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=40)
            mfcc = mfcc.T
            mfcc = np.expand_dims(mfcc, axis=0)
            mfcc = np.expand_dims(mfcc, axis=-1)
            
            yield [mfcc.astype(np.float32)]
            
        except Exception as e:
            continue

def convert_to_tflite():
    """转换模型到 TFLite 格式"""
    print("=" * 50)
    print("🔄 模型量化")
    print("=" * 50)
    
    # 加载模型
    print(f"\n1. 加载模型: {MODEL_PATH}")
    model = models.load_model(MODEL_PATH)
    
    # 获取输入形状
    input_shape = model.input_shape
    print(f"   输入形状: {input_shape}")
    print(f"   输出类别: 7")
    
    # 创建量化转换器
    print("\n2. 创建量化转换器...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # 配置量化选项
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
        tf.lite.OpsSet.TFLITE_BUILTINS
    ]
    converter.inference_input_type = tf.float32
    converter.inference_output_type = tf.float32
    
    # 设置代表性数据集
    print("3. 量化中...")
    converter.representative_dataset = representative_dataset
    
    # 转换为 TFLite
    tflite_model = converter.convert()
    
    # 保存模型
    print(f"\n4. 保存模型: {OUTPUT_PATH}")
    with open(OUTPUT_PATH, 'wb') as f:
        f.write(tflite_model)
    
    # 输出文件大小
    original_size = os.path.getsize(MODEL_PATH) / (1024 * 1024)
    quantized_size = os.path.getsize(OUTPUT_PATH) / (1024 * 1024)
    compression = (1 - quantized_size / original_size) * 100
    
    print("\n" + "=" * 50)
    print("📊 量化结果")
    print("=" * 50)
    print(f"原始模型大小: {original_size:.2f} MB")
    print(f"量化后大小:   {quantized_size:.2f} MB")
    print(f"压缩率:       {compression:.1f}%")
    print("=" * 50)
    
    return OUTPUT_PATH

def test_tflite():
    """测试量化后的模型"""
    print("\n5. 测试量化模型...")
    
    # 加载 TFLite 模型
    interpreter = tf.lite.Interpreter(model_path=OUTPUT_PATH)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"   输入形状: {input_details[0]['shape']}")
    print(f"   输出形状: {output_details[0]['shape']}")
    
    # 生成测试数据
    test_input = np.random.rand(1, 47, 40, 1).astype(np.float32)
    
    # 推理
    interpreter.set_tensor(input_details[0]['index'], test_input)
    interpreter.invoke()
    
    output = interpreter.get_tensor(output_details[0]['index'])
    print(f"   测试输出形状: {output.shape}")
    print("   ✓ TFLite 模型测试通过！")

if __name__ == "__main__":
    convert_to_tflite()
    test_tflite()
    print("\n✅ 量化完成！")