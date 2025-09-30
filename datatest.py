import mediapipe as mp
from tensorflow.keras import layers, Model, Sequential
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities

sequence = []
sentence = []
threshold = 0.8
actions = np.array(['1','2','3','4','5','6','7','8','9','10'])
# 设置字体为 SimHei（黑体）
plt.rcParams['font.sans-serif'] = ['SimHei']
# 解决负号显示为方块的问题
plt.rcParams['axes.unicode_minus'] = False
# 1. 优化后的通道注意力 (使用更高效的全连接层和激活函数)
class ChannelAttention(layers.Layer):
    def __init__(self, ratio=8, activation='relu', **kwargs):
        super(ChannelAttention, self).__init__(**kwargs)
        self.ratio = ratio
        self.activation = activation

    def build(self, input_shape):
        self.channel = input_shape[-1]
        # 使用更少的参数：共享的全连接层
        self.shared_dense = layers.Dense(self.channel // self.ratio, activation=self.activation, use_bias=False)
        self.output_dense = layers.Dense(self.channel, activation='sigmoid', use_bias=False)

    def call(self, inputs):
        # 使用全局平均池化和最大池化
        avg_pool = tf.reduce_mean(inputs, axis=1)
        max_pool = tf.reduce_max(inputs, axis=1)

        # 共享权重的MLP
        avg_out = self.output_dense(self.shared_dense(avg_pool))
        max_out = self.output_dense(self.shared_dense(max_pool))

        scale = avg_out + max_out
        return inputs * scale[:, None, :]  # 广播


# 2. 优化后的时间注意力 (查询键共享，减少参数)
class TemporalAttention(layers.Layer):
    def __init__(self, units, **kwargs):
        super(TemporalAttention, self).__init__(**kwargs)
        self.units = units
        # 使用共享的投影矩阵用于key和query，减少参数
        self.shared_projection = layers.Dense(units, activation='tanh', use_bias=False)
        self.V = layers.Dense(1, use_bias=False)

    def call(self, query, values):
        # 投影查询和键到相同空间
        projected_query = self.shared_projection(query)  # [batch_size, units]
        projected_values = self.shared_projection(values)  # [batch_size, timesteps, units]

        # 扩展查询维度以匹配值
        projected_query = tf.expand_dims(projected_query, 1)  # [batch_size, 1, units]

        # 计算注意力分数
        score = self.V(tf.nn.tanh(projected_values + projected_query))  # [batch_size, timesteps, 1]
        attention_weights = tf.nn.softmax(score, axis=1)  # [batch_size, timesteps, 1]

        # 应用注意力权重
        context_vector = tf.reduce_sum(attention_weights * values, axis=1)  # [batch_size, feature_dim]
        return context_vector


# 3. 优化后的时空注意力模块
class SpatioTemporalAttention(layers.Layer):
    def __init__(self, ratio=8, temporal_units=64, **kwargs):
        super(SpatioTemporalAttention, self).__init__(**kwargs)
        self.ratio = ratio
        self.temporal_units = temporal_units
        self.channel_attention = ChannelAttention(ratio)
        self.temporal_attention = TemporalAttention(temporal_units)

    def call(self, inputs):
        # 通道注意力
        channel_refined = self.channel_attention(inputs)  # [batch_size, timesteps, features]

        # 时间注意力 - 使用最后一个时间步作为查询
        temporal_refined = self.temporal_attention(
            query=channel_refined[:, -1, :],  # 最后一个时间步
            values=channel_refined
        )
        return temporal_refined


# 4. 轻量级可分离卷积块
def lightweight_separable_conv_block(inputs, filters, kernel_size, pool_size=2):
    # 深度可分离卷积减少参数
    x = layers.SeparableConv1D(filters, kernel_size, padding='same', use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    # 可选的下采样
    if pool_size > 1:
        x = layers.AveragePooling1D(pool_size=pool_size, padding='same')(x)
    return x


# 5. 构建优化后的完整模型
def build_enhanced_model(input_shape=(150, 120), num_classes=10):
    inputs = layers.Input(shape=input_shape)

    # 阶段1：多尺度特征提取 (参数减少)
    # 分支1：原始尺度 - 使用padding='same'保持尺寸
    branch1 = layers.SeparableConv1D(64, 5, padding='same', use_bias=False)(inputs)
    branch1 = layers.BatchNormalization()(branch1)
    branch1 = layers.ReLU()(branch1)

    # 分支2：下采样尺度 - 确保上采样后尺寸匹配
    branch2 = layers.AveragePooling1D(2, padding='same')(inputs)  # 使用padding='same'[1](@ref)
    branch2 = layers.SeparableConv1D(64, 3, padding='same', use_bias=False)(branch2)
    branch2 = layers.BatchNormalization()(branch2)
    branch2 = layers.ReLU()(branch2)
    branch2 = layers.UpSampling1D(2)(branch2)

    # 检查并调整尺寸（如果需要）
    # 如果尺寸仍有微小差异，可以使用Cropping1D或ZeroPadding1D[1](@ref)
    # branch2 = layers.Cropping1D(cropping=(0, 1))(branch2)  # 如果需要裁剪
    # 或者 branch1 = layers.ZeroPadding1D(padding=(0, 1))(branch1)  # 如果需要填充

    # 特征融合 - 现在形状应该匹配了
    merged = layers.Concatenate()([branch1, branch2])

    # 阶段2：双向GRU替代LSTM (速度更快)
    gru_out = layers.Bidirectional(
        layers.GRU(96, return_sequences=True, dropout=0.2)  # 减少单元数
    )(merged)

    # 阶段3：优化的时空注意力机制
    attention_output = SpatioTemporalAttention()(gru_out)

    # 阶段4：残差特征强化 + 全局特征
    shortcut = layers.GlobalAvgPool1D()(merged)
    combined = layers.Concatenate()([attention_output, shortcut])

    # 阶段5：动态分类头 (减少层数)
    x = layers.Dense(128, activation='relu')(combined)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    return Model(inputs, outputs)
def load_test_data_structured(test_data_path):
    """
    加载结构化测试数据（适配图片中的目录结构）
    test_data_path: 测试数据根目录（如：MP_Data5）
    """
    X_test = []
    y_test = []
    sample_info = []  # 存储样本信息用于调试

    # 遍历所有标签文件夹（1, 2, 3, 4, 5）
    for label_name in os.listdir(test_data_path):
        label_path = os.path.join(test_data_path, label_name)

        if not os.path.isdir(label_path):
            continue

        # 检查标签是否在actions中定义
        if label_name not in actions:
            print(f"警告: 标签 {label_name} 不在定义的actions中，跳过")
            continue

        label_idx = np.where(actions == label_name)[0][0]

        # 遍历该标签下的所有样本文件夹（如：0, 1, 2...）
        for sample_name in os.listdir(label_path):
            sample_path = os.path.join(label_path, sample_name)

            if not os.path.isdir(sample_path):
                continue

            # 收集该样本的所有帧数据
            frame_data = []
            valid_frames = 0

            # 遍历样本文件夹中的所有npy文件
            for file_name in sorted(os.listdir(sample_path)):
                if file_name.endswith('.npy'):
                    file_path = os.path.join(sample_path, file_name)

                    try:
                        data = np.load(file_path)
                        # 检查数据形状是否符合要求
                        if data.ndim == 1 and data.shape[0] == 120:
                            frame_data.append(data)
                            valid_frames += 1
                        else:
                            print(f"警告: 文件 {file_path} 形状不正确，期望 (120,)，实际 {data.shape}")
                    except Exception as e:
                        print(f"错误: 无法加载文件 {file_path}: {e}")

            # 如果收集到足够多的帧数据（至少1帧），则作为一个样本
            if valid_frames >= 1:
                # 将所有帧堆叠成 (frames, 120) 的形状
                sample_array = np.stack(frame_data, axis=0)

                # 如果帧数不足30，进行填充
                if sample_array.shape[0] < 150:
                    padding = np.zeros((30 - sample_array.shape[0], 120))
                    sample_array = np.vstack([sample_array, padding])
                # 如果帧数超过30，进行截断
                elif sample_array.shape[0] > 150:
                    sample_array = sample_array[:30, :]

                X_test.append(sample_array)
                y_test.append(label_idx)
                sample_info.append({
                    'label': label_name,
                    'sample': sample_name,
                    'frames': valid_frames
                })

    print(f"加载了 {len(X_test)} 个样本")
    for info in sample_info[:5]:  # 显示前5个样本的信息
        print(f"标签: {info['label']}, 样本: {info['sample']}, 有效帧数: {info['frames']}")

    return np.array(X_test), np.array(y_test)


def evaluate_and_visualize_structured(model, test_data_path):
    """
    评估模型并生成可视化图表（适配结构化数据）
    """
    # 加载测试数据
    print("正在加载结构化测试数据...")
    X_test, y_test = load_test_data_structured(test_data_path)

    if len(X_test) == 0:
        print("错误: 未找到测试数据")
        return None, None, None

    print(f"成功加载 {len(X_test)} 个测试样本")
    print(f"数据形状: {X_test.shape}")

    # 进行预测
    print("正在进行预测...")
    y_pred_proba = model.predict(X_test)
    y_pred = np.argmax(y_pred_proba, axis=1)

    # 计算准确率
    accuracy = accuracy_score(y_test, y_pred)
    print(f"测试准确率: {accuracy:.4f}")

    # 生成分类报告
    print("\n详细分类报告:")
    print(classification_report(y_test, y_pred, target_names=actions))

    # 创建可视化图表
    plt.figure(figsize=(16, 12))

    # 1. 准确率显示
    plt.subplot(2, 3, 1)
    plt.bar(['测试准确率'], [accuracy], color=['skyblue'])
    plt.ylim(0, 1)
    plt.title(f'模型测试准确率: {accuracy:.4f}')
    plt.ylabel('准确率')

    # 2. 混淆矩阵
    plt.subplot(2, 3, 2)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=actions, yticklabels=actions)
    plt.title('混淆矩阵')
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')

    # 3. 每个类别的准确率
    plt.subplot(2, 3, 3)
    class_accuracy = []
    class_counts = []
    for i in range(len(actions)):
        class_mask = (y_test == i)
        class_counts.append(np.sum(class_mask))
        if np.any(class_mask):
            class_acc = accuracy_score(y_test[class_mask], y_pred[class_mask])
            class_accuracy.append(class_acc)
        else:
            class_accuracy.append(0)

    bars = plt.bar(actions, class_accuracy, color='lightcoral')
    plt.ylim(0, 1)
    plt.title('每个类别的准确率')
    plt.xlabel('类别')
    plt.ylabel('准确率')
    plt.xticks(rotation=45)

    # 在柱状图上添加样本数量标注
    for i, (bar, count) in enumerate(zip(bars, class_counts)):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 0.02,
                 f'n={count}', ha='center', va='bottom', fontsize=9)

    # 4. 预测置信度分布
    plt.subplot(2, 3, 4)
    max_probs = np.max(y_pred_proba, axis=1)
    plt.hist(max_probs, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
    plt.title('预测置信度分布')
    plt.xlabel('最大预测概率')
    plt.ylabel('样本数量')

    # 5. 错误分析：按类别显示错误率
    plt.subplot(2, 3, 5)
    error_rates = []
    for i in range(len(actions)):
        class_mask = (y_test == i)
        if np.any(class_mask):
            error_rate = 1 - accuracy_score(y_test[class_mask], y_pred[class_mask])
            error_rates.append(error_rate)
        else:
            error_rates.append(0)

    plt.bar(actions, error_rates, color='orange')
    plt.title('每个类别的错误率')
    plt.xlabel('类别')
    plt.ylabel('错误率')
    plt.xticks(rotation=45)

    # 6. 置信度与准确率的关系
    plt.subplot(2, 3, 6)
    confidence_bins = np.linspace(0, 1, 11)
    bin_accuracy = []
    for i in range(len(confidence_bins) - 1):
        bin_mask = (max_probs >= confidence_bins[i]) & (max_probs < confidence_bins[i + 1])
        if np.any(bin_mask):
            bin_acc = accuracy_score(y_test[bin_mask], y_pred[bin_mask])
            bin_accuracy.append(bin_acc)
        else:
            bin_accuracy.append(0)

    bin_centers = (confidence_bins[:-1] + confidence_bins[1:]) / 2
    plt.plot(bin_centers, bin_accuracy, 'o-', color='purple')
    plt.xlabel('预测置信度')
    plt.ylabel('准确率')
    plt.title('置信度 vs 准确率')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('model_evaluation_structured.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 显示一些错误预测的详细信息
    print("\n错误预测示例:")
    errors = np.where(y_pred != y_test)[0]
    if len(errors) > 0:
        for i in errors[:5]:  # 显示前5个错误
            true_label = actions[y_test[i]]
            pred_label = actions[y_pred[i]]
            confidence = max_probs[i]
            print(f"样本 {i}: 真实={true_label}, 预测={pred_label}, 置信度={confidence:.3f}")

    return accuracy, y_test, y_pred

# 使用示例
if __name__ == "__main__":
    # 指定测试数据路径（根据您的图片，应该是MP_Data5）
    test_data_path = "D:\part\GNM\dataset"  # 请确保这个路径正确

    model = build_enhanced_model(input_shape=(150, 120),
                                 num_classes=actions.shape[0])

    model.load_weights('best.h5')

    # 评估模型并可视化结果
    accuracy, y_test, y_pred = evaluate_and_visualize_structured(model, test_data_path)