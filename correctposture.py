import cv2, time, math, numpy as np, pandas as pd
from deepface import DeepFace
import mediapipe as mp
from collections import Counter
from datetime import datetime
from tqdm import tqdm

# --------------------------------------------------------------------------- #
### CONFIGURATION #############################################################
SOURCE          = 0          # 0 = default webcam | or "interview.mp4"
PROCESS_EVERY_N = 5          # analyse 1 frame out of every N to save CPU
EAR_BLINK_THR   = 0.21
POSTURE_THR_DEG = 155        # ≥ = upright, below = slouched
SESSION_PREFIX  = datetime.now().strftime("%Y%m%d_%H%M%S")
###############################################################################

LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

# --------------------------------------------------------------------------- #
def eye_aspect_ratio(pts, idx):
    v1 = np.linalg.norm(np.array(pts[idx[1]]) - np.array(pts[idx[5]]))
    v2 = np.linalg.norm(np.array(pts[idx[2]]) - np.array(pts[idx[4]]))
    h  = np.linalg.norm(np.array(pts[idx[0]]) - np.array(pts[idx[3]]))
    return (v1 + v2) / (2.0 * h)

def torso_angle(landmarks, idx_ear, idx_sh, idx_hip, W, H):
    pts = []
    for idx in (idx_ear, idx_sh, idx_hip):
        lm = landmarks[idx]
        if lm.visibility < .5:                # not reliable
            return None
        pts.append((lm.x * W, lm.y * H))
    a, b, c = pts        # (ear, shoulder, hip)
    ba = (a[0]-b[0], a[1]-b[1])
    bc = (c[0]-b[0], c[1]-b[1])
    dot = ba[0]*bc[0] + ba[1]*bc[1]
    mag = math.hypot(*ba) * math.hypot(*bc)
    if mag == 0: return None
    cosang = max(-1., min(1., dot/mag))
    return math.degrees(math.acos(cosang))

# --------------------------------------------------------------------------- #
mp_face = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
mp_pose = mp.solutions.pose.Pose()
pose_idx = mp.solutions.pose.PoseLandmark

cap = cv2.VideoCapture(SOURCE)
if not cap.isOpened():
    raise IOError(f"❌ Cannot open source {SOURCE}")

fps = cap.get(cv2.CAP_PROP_FPS) or 30
print(f"🎥  Stream opened (fps≈{fps:.1f}).  Press q to quit.")

data = {"ts":[], "frame":[], "ear":[], "blink":[], "posture":[], "emotion":[]}
frame_no = -1
blink_cnt, blink_flag = 0, False

try:
    with tqdm(total=float('inf'), desc="Processing", unit="f") as pbar:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_no += 1
            pbar.update(1)

            if frame_no % PROCESS_EVERY_N:
                continue

            h, w = frame.shape[:2]
            rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # ---------------- Emotion (DeepFace) --------------------------- #
            try:
                emo = DeepFace.analyze(rgb, actions=['emotion'],
                                       enforce_detection=False)[0]['dominant_emotion']
            except Exception:
                emo = "unknown"

            # ---------------- Blink (Face Mesh) --------------------------- #
            ear = 0
            blink = False
            fm = mp_face.process(rgb)
            if fm.multi_face_landmarks:
                pts = [(int(lm.x*w), int(lm.y*h))
                       for lm in fm.multi_face_landmarks[0].landmark]
                if len(pts) >= 468:
                    ear = (eye_aspect_ratio(pts, LEFT_EYE) +
                           eye_aspect_ratio(pts, RIGHT_EYE)) / 2
            if ear and ear < EAR_BLINK_THR and not blink_flag:
                blink_cnt += 1
                blink = True
                blink_flag = True
            elif ear and ear >= EAR_BLINK_THR and blink_flag:
                blink_flag = False

            # ---------------- Posture (Pose) ------------------------------ #
            posture = "unknown"
            ps = mp_pose.process(rgb)
            if ps.pose_landmarks:
                lm   = ps.pose_landmarks.landmark
                left = torso_angle(lm, pose_idx.LEFT_EAR.value,
                                   pose_idx.LEFT_SHOULDER.value,
                                   pose_idx.LEFT_HIP.value, w, h)
                right= torso_angle(lm, pose_idx.RIGHT_EAR.value,
                                   pose_idx.RIGHT_SHOULDER.value,
                                   pose_idx.RIGHT_HIP.value, w, h)
                vals = [a for a in (left, right) if a]
                if vals:
                    avg = sum(vals) / len(vals)
                    posture = "Upright" if avg >= POSTURE_THR_DEG else "Slouching"

            # ---------------- Store --------------------------------------- #
            data["ts"].append(frame_no / fps)
            data["frame"].append(frame_no)
            data["ear"].append(ear)
            data["blink"].append(blink)
            data["posture"].append(posture)
            data["emotion"].append(emo)

            # ---------------- OPTIONAL live preview ----------------------- #
            cv2.putText(frame, f"{posture}", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (0,255,0) if posture=="Upright" else (0,0,255), 2)
            cv2.putText(frame, f"Blink {'YES' if blink else 'NO '}", (10,60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
            cv2.putText(frame, f"Emotion {emo}", (10,90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.imshow("Live posture‑emotion", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
finally:
    cap.release()
    cv2.destroyAllWindows()

# --------------------------------------------------------------------------- #
print("\n✔️  Capture ended, compiling report ...")
df = pd.DataFrame(data)
csv_name = f"{SESSION_PREFIX}_session_log.csv"
df.to_csv(csv_name, index=False)

duration_min = df["ts"].iloc[-1] / 60 if len(df) else 0
blink_rate   = blink_cnt / duration_min if duration_min else 0
posture_mode = Counter(df["posture"]).most_common(1)[0][0] if len(df) else "unknown"
emo_mode     = Counter(df["emotion"]).most_common(1)[0][0] if len(df) else "unknown"

print("=== SESSION SUMMARY ===")
print(f"Frames analysed    : {len(df)}")
print(f"Blink rate         : {blink_rate:.1f} blinks/min  ({blink_cnt} total)")
print(f"Dominant posture   : {posture_mode}")
print(f"Dominant emotion   : {emo_mode}")
print(f"Log saved          : {csv_name}")
