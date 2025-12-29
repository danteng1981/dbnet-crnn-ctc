# DBNet + CRNN OCR System

端到端文本检测与识别系统，基于 DBNet（文本检测）和 CRNN（文本识别）。

## 🌟 特性

- ✅ **DBNet 文本检测**：可微分二值化，自适应阈值
- ✅ **CRNN 文本识别**：CNN + BiLSTM + CTC 解码
- ✅ **SAR 识别支持**：基于 Attention 机制的识别
- ✅ **端到端 OCR**：一键完成检测和识别
- ✅ **易于部署**：支持 ONNX 导出和加速推理
- ✅ **完整训练代码**：包含数据处理和训练脚本

## 🚀 快速开始

### 安装

```bash
git clone https://github.com/danteng1981/dbnet-crnn-ctc.git
cd dbnet-crnn-ctc
pip install -r requirements.txt
```

### 使用示例

```python
from models.ocr_system import OCRSystem

# 初始化 OCR 系统
ocr = OCRSystem(
    det_model_path='weights/dbnet.pth',
    rec_model_path='weights/crnn.pth',
    rec_type='crnn',
    device='cuda'
)

# 识别图像
results = ocr('test_image.jpg')

# 打印结果
for result in results:
    print(f"Text: {result['text']}, Score: {result['score']:.3f}")
```

## 📁 项目结构

完整的代码结构正在构建中...

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 许可证

MIT License

## 👨‍💻 作者

- danteng1981
- GitHub: [@danteng1981](https://github.com/danteng1981)