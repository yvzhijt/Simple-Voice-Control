# 语音指令识别模型

基于MFCC特征提取的CNN语音指令识别模型，支持6个基础指令和1个"未知"类别。

## 功能特点

- ✅ 支持6个语音指令：打开、关闭、前进、后退、左转、右转
- ✅ 自动识别无效输入为"未知"类别
- ✅ 数据增强提升模型鲁棒性
- ✅ 实时麦克风录音推理
- ✅ 测试准确率：97%+

## 环境要求

```
Python 3.8+
TensorFlow 2.x
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 快速开始

### 1. 推理测试

```bash
python infer_mic.py
```

按 Enter 录音，说出指令后会显示识别结果。

### 2. 重新训练模型

```bash
python train_7class.py
```

训练完成后会自动保存模型和标签编码器。

## 指令说明

| 类别 | 指令 | 说明 |
|------|------|------|
| 0 | 未知/无指令 | 非指令输入 |
| 1 | 打开 | 打开某物 |
| 2 | 关闭 | 关闭某物 |
| 3 | 前进 | 向前移动 |
| 4 | 后退 | 向后移动 |
| 5 | 左转 | 向左转 |
| 6 | 右转 | 向右转 |

## 项目结构

```
.
├── keyword_model_7class.h5    # 训练好的模型
├── label_encoder_7class.pkl   # 标签编码器
├── train_7class.py           # 训练脚本
├── infer_mic.py              # 推理脚本
├── requirements.txt          # 依赖清单
└── README.md                 # 说明文档
```

## 模型配置

训练参数可在 `train_7class.py` 中调整：

```python
N_CLASSES = 7              # 类别数
EPOCHS = 400               # 训练轮数
BATCH_SIZE = 32            # 批次大小
LEARNING_RATE = 0.001     # 学习率
```

## 数据来源

- 原始语音数据（6个类别）
- EdgeTTS合成多样化表达
- 数据增强（变调、变速、噪声等）

## 推理参数

推理参数可在 `infer_mic.py` 中调整：

```python
RATE = 16000              # 采样率
RECORD_SECONDS = 1.5      # 录音时长（秒）
```

## 注意事项

1. 录音环境尽量安静
2. 说话时保持稳定的语速
3. 可根据实际效果调整置信度阈值

## License

MIT