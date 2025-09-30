import cv2
import numpy as np
import os
import mediapipe as mp

# MediaPipe 初始化
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# 手部检测模型（static_image_mode=True 适用于图片）
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Holistic 模型（用于 draw_styled_landmarks，可选）
holistic = mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


# 数据预处理函数（保持不变）
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results


def draw_styled_landmarks(image, results):
    # Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2))
    # Draw left hand
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2))
    # Draw right hand
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))


def landmarks_to_numpy(results):
    shape = (2, 21, 3)
    landmarks = results.multi_hand_landmarks
    if landmarks is None:
        return np.zeros(shape)
    elif len(landmarks) == 1:
        label = results.multi_handedness[0].classification[0].label
        hand = landmarks[0]
        if label == "Left":
            return np.array([
                np.array([[hand.landmark[i].x, hand.landmark[i].y, hand.landmark[i].z] for i in range(21)]),
                np.zeros((21, 3))
            ])
        else:
            return np.array([
                np.zeros((21, 3)),
                np.array([[hand.landmark[i].x, hand.landmark[i].y, hand.landmark[i].z] for i in range(21)])
            ])
    elif len(landmarks) == 2:
        lh_idx = rh_idx = 0
        for idx, hand_type in enumerate(results.multi_handedness):
            if hand_type.classification[0].label == 'Left':
                lh_idx = idx
            else:
                rh_idx = idx
        lh = np.array(
            [[landmarks[lh_idx].landmark[i].x, landmarks[lh_idx].landmark[i].y, landmarks[lh_idx].landmark[i].z] for i
             in range(21)])
        rh = np.array(
            [[landmarks[rh_idx].landmark[i].x, landmarks[rh_idx].landmark[i].y, landmarks[rh_idx].landmark[i].z] for i
             in range(21)])
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

    Lmark = lh_marks.flatten()
    Rmark = rh_marks.flatten()
    return np.concatenate([Lmark, Rmark])


# 设置路径和参数
DATA_PATH = r"D:\part\GNM\data"  # 使用原始字符串避免转义问题
actions = np.array(['1'])  # 动作名称
no_sequences = 45  # 45 个样本
sequence_length = 150  # 每个样本 150 帧


def process_sequence(action, sequence):
    # 格式化子文件夹名为两位数，如 '01', '02', ..., '45'
    sequence_folder = f"{sequence:02d}"
    action_path = os.path.join(DATA_PATH, action, sequence_folder)

    # 创建 .npy 输出目录（如果不存在）
    output_dir = os.path.join('D:\part\GNM\MYDara', action, sequence_folder)
    os.makedirs(output_dir, exist_ok=True)

    for frame_num in range(1, sequence_length + 1):  # 图片从 1.jpg 到 150.jpg
        # 构造图片路径：D:\part\GNM\data\1\01\1.jpg
        image_path = os.path.join(action_path, f"{frame_num}.jpg")

        if not os.path.exists(image_path):
            print(f"图片不存在: {image_path}")
            continue

        image = cv2.imread(image_path)
        if image is None:
            print(f"无法读取图片（cv2.imread失败）: {image_path}")
            continue

        # 转为 RGB 并进行 MediaPipe 检测
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 使用预先创建的 holistic 和 hands 模型
        results_holistic = holistic.process(image_rgb)
        results_hands = hands.process(image_rgb)

        # 提取并处理关键点
        hand_array = landmarks_to_numpy(results_hands)
        keypoints = process_mark_data(hand_array)

        # 保存为 .npy 文件，帧号从 0 开始（兼容训练代码）
        npy_path = os.path.join(output_dir, f"{frame_num - 1}.npy")
        np.save(npy_path, keypoints)
        print(f"已保存: {npy_path}")


if __name__ == "__main__":
    for action in actions:
        for sequence in range(1, no_sequences + 1):  # 从 1 到 45
            print(f"正在处理动作: {action}, 样本: {sequence:02d}")
            process_sequence(action, sequence)

    # 释放资源
    hands.close()
    holistic.close()
    print("所有图片处理完成，npy 文件已保存。")