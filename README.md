# 5G动态网络切片优化系统

基于Transformer架构的5G动态网络切片实时优化系统，通过分析用户移动轨迹、速度和行为数据，实现智能的网络资源分配和带宽优化。

## 🌟 项目特色

- **🧠 Transformer架构**: 采用先进的注意力机制进行时间序列预测
- **📱 实时预测**: 基于用户轨迹和行为的动态预测系统
- **🔧 智能切片**: 自适应网络切片分配和资源优化
- **📊 数据驱动**: 完整的数据处理和特征工程流水线
- **🎯 多任务学习**: 同时预测网络需求、切片类型和带宽分配

## 🏗️ 系统架构

```
5Gslices/
├── src/                          # 源代码
│   ├── data_processing/          # 数据处理模块
│   │   ├── data_processor.py     # 核心数据处理器
│   │   ├── feature_extractor.py  # 特征提取器
│   │   ├── data_generator.py     # 数据生成器
│   │   └── preprocessing.py      # 数据预处理
│   ├── models/                   # 模型模块
│   │   ├── transformer_model.py  # Transformer模型
│   │   └── slice_optimizer.py    # 网络切片优化器
│   ├── prediction_engine/        # 预测引擎
│   │   ├── time_series_predictor.py  # 时间序列预测器
│   │   └── realtime_predictor.py     # 实时预测引擎
│   └── utils/                    # 工具模块
│       ├── config_manager.py     # 配置管理
│       └── logger.py            # 日志管理
├── config.yaml                  # 系统配置
├── requirements.txt             # 依赖包
├── main.py                     # 主程序
└── train.py                    # 训练脚本
```

## 🚀 快速开始

### 1. 环境准备

```bash
# 克隆项目
git clone <project-url>
cd 5Gslices

# 创建虚拟环境 (推荐)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

### 2. 运行演示

```bash
# 运行完整系统演示
python main.py demo

# 或使用交互模式
python main.py interactive
```

### 3. 模型训练

```bash
# 训练Transformer模型
python main.py train

# 或直接运行训练脚本
python train.py
```

## 📊 功能模块

### 数据处理 (`data_processing/`)

- **用户轨迹处理**: 分析用户移动模式、速度变化和活动区域
- **行为特征提取**: 提取应用使用、通话模式、数据消费等特征
- **时空特征工程**: 基于时间和空间的高级特征构建
- **数据预处理**: 标准化、缺失值处理、异常值检测

### Transformer模型 (`models/`)

#### TransformerPredictor
- 多头注意力机制
- 位置编码
- 多任务预测头（切片类型、带宽需求、特征预测）

#### NetworkSliceOptimizer
- 强化学习优化算法
- 多切片类型支持 (eMBB, URLLC, mMTC)
- 动态资源分配策略

### 实时预测引擎 (`prediction_engine/`)

- **实时数据流处理**: 支持连续数据输入和预测
- **多用户并发**: 同时处理多个用户的预测请求
- **预测缓存**: 智能缓存机制提高响应速度
- **性能监控**: 实时监控预测延迟和准确率

## 🎯 网络切片类型

系统支持三种主要的5G网络切片类型：

### eMBB (增强型移动宽带)
- **特点**: 高带宽、中等延迟
- **应用**: 视频流、文件下载、VR/AR
- **带宽**: 10-1000 Mbps
- **延迟**: ≤20ms

### URLLC (超可靠低时延通信)
- **特点**: 超低延迟、高可靠性
- **应用**: 自动驾驶、工业控制、远程手术
- **带宽**: 1-50 Mbps
- **延迟**: ≤1ms

### mMTC (大规模机器类型通信)
- **特点**: 大连接数、低功耗
- **应用**: IoT设备、传感器网络
- **带宽**: 0.1-10 Mbps
- **延迟**: ≤100ms

## 📈 核心算法

### 1. 时间序列预测
```python
# Transformer预测器核心
predictions = model.forward(input_sequence)
# 输出: {features, slice_type, bandwidth}
```

### 2. 网络切片优化
```python
# 切片优化策略
allocations = optimizer.optimize_slices(
    predictions, network_resources, user_demands
)
```

### 3. 实时处理流程
```python
# 实时预测流程
engine.add_user_data(user_data)  # 添加数据
result = engine.predict_for_user(user_id)  # 执行预测
allocations = result.slice_allocations  # 获取切片分配
```

## 🔧 配置选项

系统配置通过 `config.yaml` 管理，主要配置项：

```yaml
model:
  transformer:
    d_model: 512          # 模型维度
    nhead: 8              # 注意力头数
    num_layers: 6         # Transformer层数
    dropout: 0.1          # Dropout率

data:
  sampling:
    window_size: 60       # 时间窗口大小
    prediction_horizon: 30 # 预测时间范围
    frequency: 1.0        # 采样频率

prediction:
  update_interval: 1.0    # 预测更新间隔
  confidence_threshold: 0.8 # 置信度阈值
```

## 📊 性能指标

系统提供多维度的性能评估：

- **预测准确率**: 切片类型分类准确率
- **预测延迟**: 单次预测处理时间
- **带宽利用率**: 网络资源使用效率
- **用户满意度**: 服务质量满足程度

## 🛠️ 开发指南

### 扩展新的切片类型

```python
# 在 slice_optimizer.py 中添加新类型
class NewSliceType(Enum):
    NEW_TYPE = 3

# 更新切片需求配置
slice_requirements[SliceType.NEW_TYPE] = SliceRequirements(
    slice_type=SliceType.NEW_TYPE,
    min_bandwidth=5.0,
    max_bandwidth=100.0,
    latency_req=10.0,
    reliability_req=0.98,
    priority=2
)
```

### 自定义特征提取

```python
# 继承 FeatureExtractor 类
class CustomFeatureExtractor(FeatureExtractor):
    def extract_custom_features(self, data):
        # 实现自定义特征提取逻辑
        return custom_features
```

## 📋 依赖项

主要依赖包：

- **PyTorch**: 深度学习框架
- **Transformers**: Transformer模型实现
- **NumPy/Pandas**: 数据处理
- **Matplotlib/Seaborn**: 数据可视化
- **Scikit-learn**: 机器学习工具
- **PyYAML**: 配置文件处理

## 🤝 贡献指南

1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 📞 联系方式

- 项目维护者: 5G Slicing Team
- 邮箱: team@5gslicing.com
- 项目主页: https://github.com/5g-slicing/dynamic-optimization

## 🙏 致谢

感谢以下开源项目的支持：
- PyTorch团队
- Hugging Face Transformers
- 所有贡献者和测试用户

---

🌟 如果此项目对您有帮助，请给我们一个星标！