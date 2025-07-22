"""
多任务YOLO模型配置和使用指南
==================================

这个文件详细说明了如何配置和使用MultiBaseModel和MultiModel类进行多任务学习。

作者注释：由于你是深度学习小白，我会从最基础的概念开始解释。
"""

import torch
from ultralytics.nn.tasks import MultiModel, MultiBaseModel
from ultralytics.utils import LOGGER

# ================================================================
# 1. 理解多任务学习的基本概念
# ================================================================

"""
什么是多任务学习？
- 传统YOLO只做一个任务，比如只检测物体位置
- 多任务YOLO可以同时做多个任务：检测物体 + 分割物体 + 检测人体姿态
- 就像一个人既会开车，又会做饭，还会画画

为什么要多任务学习？
1. 效率高：一个模型干多个活，省资源
2. 性能好：不同任务可以互相帮助学习
3. 实用：实际应用中常常需要多种信息
"""

# ================================================================
# 2. 理解模型的三个组成部分
# ================================================================

"""
YOLO模型就像一个工厂的流水线：

1. Backbone（主干网络）- 相当于原料处理车间
   - 输入：原始图片 (例如: 640x640x3的RGB图片)
   - 功能：提取图片的基础特征 (边缘、纹理、形状等)
   - 输出：多层次的特征图 (P3, P4, P5 - 不同分辨率的特征)

2. Neck（颈部网络）- 相当于特征融合车间  
   - 输入：Backbone输出的特征图
   - 功能：融合不同层次的特征，让大物体和小物体都能被很好地检测
   - 输出：融合后的特征图

3. Head（头部网络）- 相当于最终产品车间
   - 输入：Neck输出的融合特征
   - 功能：针对具体任务做预测
   - 输出：
     * Detect Head -> 物体位置和类别
     * Segment Head -> 物体的像素级别分割掩码  
     * Pose Head -> 人体关键点位置
"""

# ================================================================
# 3. 配置文件详解 (yolo11n-multi.yaml)
# ================================================================

"""
我们创建的配置文件结构：

# 全局参数设置
nc: 80              # 类别数量 (COCO数据集有80个类别)
kpt_shape: [17, 3]  # 人体姿态：17个关键点，每个点有(x,y,可见性)
channels: 3         # 输入图片通道数 (RGB=3)

# 模型尺寸配置 (可以选择不同大小的模型)
scales:
  n: [0.50, 0.25, 1024]  # nano版本：小而快
  s: [0.50, 0.50, 1024]  # small版本：平衡
  m: [0.50, 1.00, 512]   # medium版本：准确
  l: [1.00, 1.00, 512]   # large版本：更准确
  x: [1.00, 1.50, 512]   # extra large版本：最准确但最慢

# backbone部分配置说明：
每一行格式：[from, repeats, module, args]
- from: 从哪一层获取输入 (-1表示上一层)
- repeats: 这个模块重复几次
- module: 使用什么类型的模块 (Conv=卷积, C3k2=特殊的卷积块)
- args: 模块的参数 [输出通道数, 卷积核大小, 步长]

例如：[-1, 1, Conv, [64, 3, 2]]
意思是：从上一层输入，使用1个Conv模块，输出64个通道，3x3卷积核，步长为2

# head部分配置说明：
这里定义了三个任务头：
- Detect: 检测物体边界框和类别
- Segment: 分割物体轮廓  
- Pose: 检测人体关键点
"""

# ================================================================
# 4. 代码使用示例
# ================================================================

def create_multi_task_model():
    """创建并配置多任务模型"""
    
    # 方法1: 使用配置文件创建模型
    model_config = "yolo11n-multi.yaml"  # 我们创建的配置文件
    model = MultiModel(cfg=model_config, ch=3, nc=80, verbose=True)
    
    print("模型创建成功！")
    print(f"模型包含的任务头: {model.task_heads}")
    
    return model

def understand_model_structure():
    """理解模型结构"""
    
    model = create_multi_task_model()
    
    # 查看模型结构
    print("\n模型层数:", len(model.model))
    print("保存特征的层:", model.save)
    print("模型步长:", model.stride)
    
    # 找出各个任务头的位置
    for i, layer in enumerate(model.model):
        layer_type = type(layer).__name__
        if layer_type in ['Detect', 'Segment', 'Pose']:
            print(f"第{i}层是{layer_type}头")

def test_model_inference():
    """测试模型推理过程"""
    
    model = create_multi_task_model()
    model.eval()  # 设置为评估模式
    
    # 创建测试输入 (batch_size=1, channels=3, height=640, width=640)
    test_input = torch.randn(1, 3, 640, 640)
    
    print("\n开始推理...")
    
    # 前向传播
    with torch.no_grad():
        outputs = model(test_input)
    
    print("推理完成！")
    print("输出包含的任务:", list(outputs.keys()))
    
    # 分析每个任务的输出
    for task_name, task_output in outputs.items():
        if isinstance(task_output, (list, tuple)):
            print(f"{task_name}任务输出: {len(task_output)}个尺度")
            for i, out in enumerate(task_output):
                print(f"  尺度{i}: {out.shape}")
        else:
            print(f"{task_name}任务输出形状: {task_output.shape}")

# ================================================================
# 5. 与单任务模型的对比
# ================================================================

def compare_with_single_task():
    """对比多任务模型和单任务模型的区别"""
    
    print("\n=== 单任务 vs 多任务模型对比 ===")
    
    # 单任务模型（只做检测）
    print("单任务模型特点：")
    print("- 只有一个Detect头")
    print("- 只输出边界框和类别")
    print("- 模型简单，专注单一任务")
    print("- 如果要做分割，需要另外训练一个模型")
    
    # 多任务模型
    print("\n多任务模型特点：")
    print("- 有多个头：Detect + Segment + Pose")
    print("- 共享backbone和neck，节省计算")
    print("- 一次推理得到多种结果")
    print("- 不同任务互相促进学习效果")

# ================================================================
# 6. 关键实现细节解释
# ================================================================

def explain_key_implementations():
    """解释关键实现细节"""
    
    print("\n=== MultiBaseModel关键实现 ===")
    
    print("1. _predict_once方法的改进：")
    print("   - 原始BaseModel只返回最后一层输出")
    print("   - MultiBaseModel收集所有任务头的输出")
    print("   - 返回字典格式：{'Detect': output1, 'Segment': output2, 'Pose': output3}")
    
    print("\n2. 多头类型定义：")
    print("   - multi_head_types = (Detect, Segment, Pose)")
    print("   - 在前向传播中识别这些头并收集输出")
    
    print("\n3. _apply方法的适配：")
    print("   - 处理多个头的stride、anchors等属性")
    print("   - 确保所有头都能正确应用设备/数据类型变换")

# ================================================================
# 7. 训练和使用建议
# ================================================================

def training_recommendations():
    """训练和使用建议"""
    
    print("\n=== 训练建议 ===")
    
    print("1. 数据准备：")
    print("   - 需要包含多任务标注的数据集")
    print("   - 检测：边界框 + 类别")
    print("   - 分割：像素级掩码")
    print("   - 姿态：关键点坐标")
    
    print("\n2. 损失函数：")
    print("   - 每个任务有独立的损失函数")
    print("   - 总损失 = λ1*检测损失 + λ2*分割损失 + λ3*姿态损失")
    print("   - λ是权重系数，需要调整平衡")
    
    print("\n3. 训练策略：")
    print("   - 可以先用预训练的单任务模型初始化")
    print("   - 逐步增加任务，避免训练不稳定")
    print("   - 监控各任务的损失变化")
    
    print("\n4. 推理应用：")
    print("   - 可以选择性使用某些任务的输出")
    print("   - 根据应用场景调整后处理参数")
    print("   - 注意多任务输出的数据格式")

# ================================================================
# 主函数演示
# ================================================================

if __name__ == "__main__":
    print("多任务YOLO模型配置指南")
    print("=" * 50)
    
    # 创建模型并理解结构
    create_multi_task_model()
    understand_model_structure()
    
    # 测试推理
    test_model_inference()
    
    # 对比说明
    compare_with_single_task()
    
    # 关键实现解释
    explain_key_implementations()
    
    # 训练建议
    training_recommendations()
    
    print("\n恭喜！你已经了解了多任务YOLO模型的基本配置和使用方法！")
