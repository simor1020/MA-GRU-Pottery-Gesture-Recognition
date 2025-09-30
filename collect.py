import cv2
import numpy as np
import os
import mediapipe as mp


mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities

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
    """
    要训练脸部坐标就取消注释
    # Draw face connections
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

# 初始化摄像头
cap = cv2.VideoCapture(0)
DATA_PATH = os.path.join('MP_Data')

# Actions that we try to detect
actions = np.array(['1'])
# Thirty videos worth of data
no_sequences = 10
# Videos are going to be 30 frames in length
sequence_length = 30
for action in actions:
    for sequence in range(no_sequences):
        try:
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
        except:
            pass

cap = cv2.VideoCapture(0)
# Set mediapipe model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    # NEW LOOP
    # Loop through actions
    for action in actions:
        # Loop through sequences aka videos
        for sequence in range(no_sequences):
            # Loop through video length aka sequence length
            for frame_num in range(sequence_length):
                # Read feed
                ret, frame = cap.read()
                # Make detections
                image, results = mediapipe_detection(frame, holistic)
                #                 print(results)
                # Draw landmarks
                draw_styled_landmarks(image, results)
                # NEW Apply wait logic
                if frame_num == 0:
                    cv2.putText(image, 'STARTING COLLECTION', (120, 200),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
                    cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15, 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    # Show to screen
                    cv2.imshow('OpenCV Feed', image)
                    cv2.waitKey(2000)
                else:
                    cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15, 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    # Show to screen
                    cv2.imshow('OpenCV Feed', image)
                # NEW Export keypoints
                results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                hand_array = landmarks_to_numpy(results)
                keypoints = process_mark_data(hand_array)
                npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                np.save(npy_path, keypoints)
                # Break gracefully
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
cap.release()
cv2.destroyAllWindows()
