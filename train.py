
import tensorflow as tf
from tensorflow.keras import layers, Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

no_sequences = 32
# Videos are going to be 30 frames in length
sequence_length = 150
DATA_PATH = os.path.join('D:\part\GNM\dataset')
actions = np.array(['1','2','3','4','5','6','7','8','9','10'])

label_map = {label: num for num, label in enumerate(actions)}
sequences, labels = [], []
for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])

X = np.array(sequences)
y = to_categorical(labels).astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)
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
# 改进3：优化训练配置
def create_optimizer():
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-4,
        decay_steps=1000,
        decay_rate=0.9
    )
    return Adam(learning_rate=lr_schedule)


# 构建模型
model = build_enhanced_model(input_shape=(150, 120),
                             num_classes=actions.shape[0])

# 编译模型
model.compile(
    optimizer=create_optimizer(),
    loss='categorical_crossentropy',
    metrics=['categorical_accuracy', tf.keras.metrics.Precision()]
)

# 训练模型并获取训练历史
history = model.fit(X_train, y_train, epochs=150, callbacks=[tb_callback])

# 绘制损失值和准确度折线图
plt.figure(figsize=(12, 4))

# 绘制损失值折线图
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.title('Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# 绘制准确度折线图
plt.subplot(1, 2, 2)
plt.plot(history.history['categorical_accuracy'], label='Training Accuracy')
plt.title('Training Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

model.summary()
model.save('best.h5')