import cv2
import mediapipe as mp
import numpy as np

# 设置MediaPipe姿态估计
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# 计算向量夹角的函数
def calculate_angle(a, b, c):
    """
    计算三点形成的角度，a, b, c 是(x, y)坐标。
    计算角度∠abc，b为角的顶点。
    """
    # 向量AB与向量BC
    ab = np.array([a[0] - b[0], a[1] - b[1]])
    bc = np.array([c[0] - b[0], c[1] - b[1]])

    # 计算向量的点积
    dot_product = np.dot(ab, bc)
    # 计算向量的模长
    norm_ab = np.linalg.norm(ab)
    norm_bc = np.linalg.norm(bc)

    # 计算角度
    cos_theta = dot_product / (norm_ab * norm_bc)
    angle = np.arccos(np.clip(cos_theta, -1.0, 1.0))  # 计算夹角，并限制cos值范围
    return np.degrees(angle)  # 转换为度数

# 处理视频并标注姿势
def process_video(input_video_path, output_video_path):
    cap = cv2.VideoCapture(input_video_path)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # 获取下肢相关的关节点，检查每个关键点的置信度
            keypoints = {}
            for side in ['left', 'right']:
                # 对于每个关节，获取其置信度并判断是否显示
                hip_confidence = landmarks[mp_pose.PoseLandmark.LEFT_HIP if side == 'left' else mp_pose.PoseLandmark.RIGHT_HIP].visibility
                knee_confidence = landmarks[mp_pose.PoseLandmark.LEFT_KNEE if side == 'left' else mp_pose.PoseLandmark.RIGHT_KNEE].visibility
                ankle_confidence = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE if side == 'left' else mp_pose.PoseLandmark.RIGHT_ANKLE].visibility
                heel_confidence = landmarks[mp_pose.PoseLandmark.LEFT_HEEL if side == 'left' else mp_pose.PoseLandmark.RIGHT_HEEL].visibility
                foot_confidence = landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX if side == 'left' else mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].visibility

                # 设置置信度阈值，低于阈值的关节不显示
                threshold = 0.8  # 可以根据需求调整

                if hip_confidence > threshold:
                    keypoints[f"hip_{side}"] = (landmarks[mp_pose.PoseLandmark.LEFT_HIP if side == 'left' else mp_pose.PoseLandmark.RIGHT_HIP].x * width, landmarks[mp_pose.PoseLandmark.LEFT_HIP if side == 'left' else mp_pose.PoseLandmark.RIGHT_HIP].y * height)
                if knee_confidence > threshold:
                    keypoints[f"knee_{side}"] = (landmarks[mp_pose.PoseLandmark.LEFT_KNEE if side == 'left' else mp_pose.PoseLandmark.RIGHT_KNEE].x * width, landmarks[mp_pose.PoseLandmark.LEFT_KNEE if side == 'left' else mp_pose.PoseLandmark.RIGHT_KNEE].y * height)
                if ankle_confidence > threshold:
                    keypoints[f"ankle_{side}"] = (landmarks[mp_pose.PoseLandmark.LEFT_ANKLE if side == 'left' else mp_pose.PoseLandmark.RIGHT_ANKLE].x * width, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE if side == 'left' else mp_pose.PoseLandmark.RIGHT_ANKLE].y * height)
                if heel_confidence > threshold:
                    keypoints[f"heel_{side}"] = (landmarks[mp_pose.PoseLandmark.LEFT_HEEL if side == 'left' else mp_pose.PoseLandmark.RIGHT_HEEL].x * width, landmarks[mp_pose.PoseLandmark.LEFT_HEEL if side == 'left' else mp_pose.PoseLandmark.RIGHT_HEEL].y * height)
                if foot_confidence > threshold:
                    keypoints[f"foot_{side}"] = (landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX if side == 'left' else mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].x * width, landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX if side == 'left' else mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].y * height)

            # 计算膝关节和踝关节的角度
            if "hip_left" in keypoints and "knee_left" in keypoints and "ankle_left" in keypoints:
                knee_angle_left = calculate_angle(keypoints["hip_left"], keypoints["knee_left"], keypoints["ankle_left"])
            if "hip_right" in keypoints and "knee_right" in keypoints and "ankle_right" in keypoints:
                knee_angle_right = calculate_angle(keypoints["hip_right"], keypoints["knee_right"], keypoints["ankle_right"])
            if "knee_left" in keypoints and "ankle_left" in keypoints and "heel_left" in keypoints:
                ankle_angle_left = calculate_angle(keypoints["knee_left"], keypoints["ankle_left"], keypoints["heel_left"])
            if "knee_right" in keypoints and "ankle_right" in keypoints and "heel_right" in keypoints:
                ankle_angle_right = calculate_angle(keypoints["knee_right"], keypoints["ankle_right"], keypoints["heel_right"])

            # 绘制下肢的关键点
            for key, (x, y) in keypoints.items():
                cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)

            # 绘制下肢连线
            if "hip_left" in keypoints and "knee_left" in keypoints and "ankle_left" in keypoints:
                cv2.line(frame, (int(keypoints["hip_left"][0]), int(keypoints["hip_left"][1])), (int(keypoints["knee_left"][0]), int(keypoints["knee_left"][1])), (0, 255, 0), 2)
                cv2.line(frame, (int(keypoints["knee_left"][0]), int(keypoints["knee_left"][1])), (int(keypoints["ankle_left"][0]), int(keypoints["ankle_left"][1])), (0, 255, 0), 2)
            if "hip_right" in keypoints and "knee_right" in keypoints and "ankle_right" in keypoints:
                cv2.line(frame, (int(keypoints["hip_right"][0]), int(keypoints["hip_right"][1])), (int(keypoints["knee_right"][0]), int(keypoints["knee_right"][1])), (0, 255, 0), 2)
                cv2.line(frame, (int(keypoints["knee_right"][0]), int(keypoints["knee_right"][1])), (int(keypoints["ankle_right"][0]), int(keypoints["ankle_right"][1])), (0, 255, 0), 2)

            # 显示膝关节和踝关节角度
            if "knee_angle_left" in locals():
                cv2.putText(frame, f"Knee Angle (left): {int(knee_angle_left)}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            if "knee_angle_right" in locals():
                cv2.putText(frame, f"Knee Angle (right): {int(knee_angle_right)}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            if "ankle_angle_left" in locals():
                cv2.putText(frame, f"Ankle Angle (left): {int(ankle_angle_left)}", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            if "ankle_angle_right" in locals():
                cv2.putText(frame, f"Ankle Angle (right): {int(ankle_angle_right)}", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        out.write(frame)

    cap.release()
    out.release()
    print(f"分析视频已保存为：{output_video_path}")

if __name__ == "__main__":
    input_video = "run.mp4"  # 输入视频路径
    output_video = "analysis.mp4"  # 输出视频路径
    process_video(input_video, output_video)
