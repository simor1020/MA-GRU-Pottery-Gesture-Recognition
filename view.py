import cv2
import mediapipe as mp
from tensorflow.keras import layers, Model, Sequential
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from tensorflow.keras import layers, Model, Sequential
import numpy as np
import socket
import os
from pathlib import Path

mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities

sequence = []
sentence = []
threshold = 0.8
actions = np.array(['1','2','3','4','5','6','7','8','9','10'])

# MediaPipe Hands初始化
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=2,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)


def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results

def draw_styled_landmarks(image, results):
    # Draw face connections
    """
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
                             mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                             mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                             )
    """
    # Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                             mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                             )
    # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                             )
    # Draw right hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                             )

def landmarks_to_numpy(results):
    shape = (2, 21, 3)
    landmarks = results.multi_hand_landmarks
    if landmarks is None:
        return np.zeros(shape)
    elif len(landmarks) == 1:
        label = results.multi_handedness[0].classification[0].label
        hand = landmarks[0]
        if label == "Left":
            return np.array([np.array([[hand.landmark[i].x, hand.landmark[i].y, hand.landmark[i].z] for i in range(21)]),
                             np.zeros((21, 3))])
        else:
            return np.array([np.zeros((21, 3)),
                             np.array([[hand.landmark[i].x, hand.landmark[i].y, hand.landmark[i].z] for i in range(21)])])
    elif len(landmarks) == 2:
        lh_idx, rh_idx = 0, 0
        for idx, hand_type in enumerate(results.multi_handedness):
            label = hand_type.classification[0].label
            if label == 'Left':
                lh_idx = idx
            if label == 'Right':
                rh_idx = idx
        lh = np.array([[landmarks[lh_idx].landmark[i].x, landmarks[lh_idx].landmark[i].y, landmarks[lh_idx].landmark[i].z] for i in range(21)])
        rh = np.array([[landmarks[rh_idx].landmark[i].x, landmarks[rh_idx].landmark[i].y, landmarks[rh_idx].landmark[i].z] for i in range(21)])
        return np.array([lh, rh])
    else:
        return np.zeros((2, 21, 3))

def relative_coordinate(arr, point):
    return arr - point

def standardization(hand_arr):
    if np.all(hand_arr == 0):
        return hand_arr
    mean = np.mean(hand_arr, axis=0)
    std = np.std(hand_arr, axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return (hand_arr - mean) / std

def process_mark_data(hand_arr):
    lh_root = hand_arr[0, 0]
    rh_root = hand_arr[1, 0]

    lh_marks = relative_coordinate(hand_arr[0, 1:], lh_root)
    rh_marks = relative_coordinate(hand_arr[1, 1:], rh_root)

    lh_marks = standardization(lh_marks)
    rh_marks = standardization(rh_marks)

    Lmark=lh_marks.flatten()
    Rmark=rh_marks.flatten()
    return np.concatenate([Lmark, Rmark])
def extract_keypoints(results):
    #pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    #face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([lh, rh])

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
def build_enhanced_model(input_shape=(150, 120), num_classes=3):
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


model = build_enhanced_model(input_shape=(150, 120),
                             num_classes=actions.shape[0])

model.load_weights('best.h5')

colors = [(245, 117, 16), (117, 245, 16), (16, 117, 245),(16, 117, 245),(100,200,255),(245, 117, 16),
          (150, 123, 16), (185, 220, 245),(210, 117, 245),(50,100,255)]#四个动作的框框，要增加动作数目，就多加RGB元组


def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0, 60 + num * 40), (int(prob * 100), 90 + num * 40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85 + num * 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                    cv2.LINE_AA)
    return output_frame

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
# Set mediapipe model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        # Read feed
        ret, frame = cap.read()
        # Make detections
        image, results = mediapipe_detection(frame, holistic)
        #print(results)
        # Draw landmarks
        draw_styled_landmarks(image, results)
        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        hand_array = landmarks_to_numpy(results)
        keypoints = process_mark_data(hand_array)
        sequence.append(keypoints)
        sequence = sequence[-150:]
        if len(sequence) == 150:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            # print(actions[np.argmax(res)])
            # 3. Viz logic
            if res[np.argmax(res)] > threshold:
                if len(sentence) > 0:
                    if actions[np.argmax(res)] != sentence[-1]:
                        sentence.append(actions[np.argmax(res)])
                        print(np.argmax(res))
                else:
                    sentence.append(actions[np.argmax(res)])
            if len(sentence) > 5:
                sentence = sentence[-5:]
            # Viz probabilities
            image = prob_viz(res, actions, image, colors)

        cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
        cv2.putText(image, ' '.join(sentence), (3, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        # Show to screen

        cv2.imshow('OpenCV Feed', image)
        # Break gracefully
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()
