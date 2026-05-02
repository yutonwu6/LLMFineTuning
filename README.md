# LLMFineTuning

## 项目简介

`LLMFineTuning` 是一个面向大语言模型（Large Language Model，简称LLM）的高效微调工具集。它主要利用**参数高效微调**技术，帮助开发者和研究者在本地轻松微调大语言模型，让通用基础模型能更好地适应特定场景。

该项目提供了一个从数据集准备、预处理、模型微调到效果评估的完整工作流，让普通用户可以在配置较有限的设备上，高效完成针对大语言模型的定制化适配。

## 主要特性

- **支持多种微调模式**：
  - **全量参数微调（SFT）**：完整的监督式微调。
  - **LoRA微调（Low-Rank Adaptation）**：引入低秩矩阵，只训练增量部分，大幅降低训练参数量。
  - **QLoRA微调**：在 LoRA 的基础上引入 4-bit 量化，显著减少显存占用，适合消费级 GPU。
- **数据集自动预处理**：提供 `create_dataset.ipynb` 笔记本，支持从原始文本或对话构造训练数据集。
- **可选评估功能**：集成简易推理测评，可按 Prompt 测试微调后的模型输出效果。
- **命令行灵活参数**：所有关键超参数（模型名称、训练轮数、学习率、微调类型等）都可通过命令行直接指定，便于实验调参。

## 项目结构

```text
LLMFineTuning/
├── config.py               # 全局配置（模型路径、微调类型、批量大小、学习率等）
├── data.py                 # 加载 dataset 目录下的数据集文件（JSON格式）
├── model.py                # 加载模型及适配器（SFT/LoRA/QLoRA）
├── train.py                # 训练模块，基于 Hugging Face Trainer
├── eval.py                 # 简易推理评测模块
├── main.py                 # 主入口，串联数据、模型、训练和评测
├── create_dataset.ipynb    # Jupyter Notebook 用于构建自定义训练集
├── requirements.txt        # Python 依赖环境列表
└── evalscope/              # 模型评估相关扩展目录
```

## 环境配置（venv / conda）

使用 Python 3.10 或更高版本，推荐创建独立虚拟环境。依赖项已整理在 requirements.txt 中：

```bash
# 克隆仓库
git clone https://github.com/yutonwu6/LLMFineTuning.git
cd LLMFineTuning

# 创建虚拟环境（使用 venv）
python3 -m venv .venv
source .venv/bin/activate   # Linux / macOS
.venv\Scripts\activate      # Windows

# 安装依赖
pip install -r requirements.txt
```

## 使用方法

- 直接在 Jupyter Lab/Notebook 中打开 `create_dataset.ipynb`，按步骤执行单元格，根据注释完成自定义数据集的预处理。

- 模型训练的主入口是 `main.py`，支持通过 `--fine_tuning` 参数选择微调方式：

    ```bash
    python main.py
    ```

## 模型评测（EvalScope）
`evalscope` 目录为评测任务和结果示例展示，结果包括基础模型和三种微调后模型。

## Reference

- https://github.com/gazelle93/llm-fine-tuning-sft-lora-qlora
