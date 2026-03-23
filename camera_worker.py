import cv2
import time
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QImage
from ultralytics import YOLO
import supervision as sv

from swimmer_module import PROFILES, SwimmerMonitor, GhostTracker, RISK_WARNING, RISK_DANGER

# 모델 경로 하드코딩 혹은 외부 주입 가능
MODEL_PATH = "yolo26m_openvino_model_1280/"
CONF_THRESHOLD = 0.25
SKIP_FRAMES = 3
IMG_SIZE = 1280

class CameraWorker(QThread):
    # (카메라 ID, 그려진 QImage, 깨끗한 QImage)
    change_pixmap_signal = pyqtSignal(int, QImage, QImage)
    # (카메라 ID, 알림 메시지, 알림 레벨: 0=Safe, 1=Warning, 2=Danger)
    alert_signal = pyqtSignal(int, str, int)
    
    def __init__(self, cam_id, source, profile_name="KIDS_POOL"):
        super().__init__()
        self.cam_id = cam_id
        self.source = source
        self.profile_name = profile_name
        self._run_flag = True
        self.is_paused = False
        
        # ROI Data
        self.pool_polygon = None
        self.exit_polygons = []
        
    def run(self):
        # 1. 모델 로드 (스레드별 독립 객체)
        try:
            is_pose_model = "pose" in MODEL_PATH.lower()
            if is_pose_model:
                model = YOLO(MODEL_PATH, task="pose")
            else:
                model = YOLO(MODEL_PATH)
        except Exception as e:
            self.alert_signal.emit(self.cam_id, f"모델 로드 오류: {e}", 2)
            return

        tracker = sv.ByteTrack(
            track_activation_threshold=0.2,
            lost_track_buffer=30,
            frame_rate=30,
        )

        ghost_tracker = GhostTracker()
        monitors = {}
        prev_track_ids = set()
        frame_idx = 0
        
        # 2. 미디어 소스 판별 (웹캠 번호 처리)
        src = self.source
        is_file = False
        if isinstance(src, str):
            if src.isdigit():
                src = int(src)
            elif src.endswith(('.mp4', '.avi', '.mkv', '.mov')):
                is_file = True
                
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            self.alert_signal.emit(self.cam_id, f"소스 연결 실패: {src}", 2)
            return
            
        profile = PROFILES.get(self.profile_name, PROFILES["KIDS_POOL"])

        while self._run_flag:
            if self.is_paused:
                time.sleep(0.1)
                continue
                
            ret, frame = cap.read()
            if not ret:
                if is_file:
                    # 파일 끝 도달 시 무한 반복 처리
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    self.alert_signal.emit(self.cam_id, "스트림 종료", 1)
                    break
                    
            frame_idx += 1
            if frame_idx % SKIP_FRAMES != 0:
                continue

            # --- 추론 ---
            results = model(frame, imgsz=IMG_SIZE, verbose=False, conf=CONF_THRESHOLD, device="cpu")
            result = results[0]
            
            detections = sv.Detections.from_ultralytics(result)
            
            has_pose = False
            if result.keypoints is not None:
                kp_xy_all = result.keypoints.xy.cpu().numpy()
                kp_conf_all = (
                    result.keypoints.conf.cpu().numpy()
                    if result.keypoints.conf is not None
                    else np.ones((kp_xy_all.shape[0], kp_xy_all.shape[1]))
                )
                detections.data["kp_xy"] = kp_xy_all
                detections.data["kp_conf"] = kp_conf_all
                has_pose = True
            else:
                detections.data["kp_xy"] = np.empty((len(detections), 0, 2))
                detections.data["kp_conf"] = np.empty((len(detections), 0))

            if hasattr(detections, "class_id") and detections.class_id is not None:
                detections = detections[detections.class_id == 0]

            detections = tracker.update_with_detections(detections)
            current_ids = set()
            
            # --- 상태 업데이트 시각화 그리기 ---
            for xyxy, mask, confidence, class_id, track_id, data in detections:
                if track_id is None: continue
                
                # 중심점 좌표
                cx = (xyxy[0] + xyxy[2]) / 2
                cy = (xyxy[1] + xyxy[3]) / 2
                center = (cx, cy)
                
                # ROI 필터링 로직
                is_in_pool = True
                is_in_exit = False
                
                if self.pool_polygon is not None and len(self.pool_polygon) > 0:
                    is_in_pool = (cv2.pointPolygonTest(self.pool_polygon, center, False) >= 0)
                
                for ep in self.exit_polygons:
                    if len(ep) > 0 and cv2.pointPolygonTest(ep, center, False) >= 0:
                        is_in_exit = True
                        break
                
                # 구역 밖에 있으면 무시
                if not is_in_pool and not is_in_exit:
                    continue

                current_ids.add(track_id)
                ghost_tracker.mark_alive(track_id)

                if track_id not in monitors:
                    monitors[track_id] = SwimmerMonitor(track_id, profile)
                    matched_ghost_id, matched_info = ghost_tracker.try_match_new_detection(xyxy)
                    if matched_ghost_id is not None and matched_ghost_id in monitors:
                        monitors[track_id].risk_history = monitors[matched_ghost_id].risk_history
                        monitors[track_id].risk_score = monitors[matched_ghost_id].risk_score

                kp_xy = data.get("kp_xy", None) if has_pose else None
                kp_conf = data.get("kp_conf", None) if has_pose else None

                # 탈출 영역에 있으면 무조건 안전(SAFE_EXIT) 처리 및 강제 할당
                if is_in_exit:
                    status = "SAFE_EXIT"
                    risk_level = 0
                    monitors[track_id].state = status
                    monitors[track_id].risk_score = 0.0
                    monitors[track_id].risk_history.clear()
                    # 히스토리만 업데이트 목적으로 update 호출 (결과는 무시)
                    monitors[track_id].update(xyxy, kp_xy, kp_conf)
                else:
                    status, risk_level = monitors[track_id].update(xyxy, kp_xy, kp_conf)
                
                # 시각화 BBox 및 상태 표시
                x1, y1, x2, y2 = map(int, xyxy)
                risk_score = monitors[track_id].risk_score
                dbg = monitors[track_id].debug

                if risk_level == 2: color = (0, 0, 255)       # 위험 (빨강)
                elif risk_level == 1: color = (0, 220, 255)   # 경고 (노랑)
                else: color = (0, 200, 0)                     # 안전 (초록)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                bar_w = x2 - x1
                bar_fill = int(bar_w * min(1.0, risk_score))
                cv2.rectangle(frame, (x1, y2 + 2), (x2, y2 + 8), (50, 50, 50), -1)
                if bar_fill > 0:
                    bc = (0, 200, 0) if risk_score < RISK_WARNING else ((0, 220, 255) if risk_score < RISK_DANGER else (0, 0, 255))
                    cv2.rectangle(frame, (x1, y2 + 2), (x1 + bar_fill, y2 + 8), bc, -1)

                label = f"ID:{track_id} {status}"
                sub = f"R:{risk_score:.2f} Spd:{dbg.get('speed', 0):.1f}"
                cv2.putText(frame, label, (x1 + 2, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
                cv2.putText(frame, sub, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

                # 위험 레벨 상향 시 UI로 알림 방출
                if risk_level > 0 and (frame_idx % 30 == 0):
                    self.alert_signal.emit(self.cam_id, f"ID:{track_id} {status} ({risk_score:.2f})", risk_level)

            # --- 소실 감지 (GhostTracker) ---
            disappeared = prev_track_ids - current_ids
            for tid in disappeared:
                if tid in monitors:
                    m = monitors[tid]
                    if m.bbox_history:
                        ghost_tracker.mark_disappeared(tid, list(m.bbox_history)[-1], m.state, m.risk_score)
            prev_track_ids = current_ids.copy()

            # 유령 경고 시각화 및 알림 전송
            for alert in ghost_tracker.get_alerts(profile):
                bx1, by1, bx2, by2 = map(int, alert["bbox"])
                elapsed = alert["elapsed"]
                gc = (0, 180, 255) if alert["level"] == 1 else (0, 0, 255)

                cv2.rectangle(frame, (bx1, by1), (bx2, by2), gc, 2, cv2.LINE_AA)
                gtxt = f"LOST ID:{alert['track_id']} {elapsed:.1f}s"
                cv2.putText(frame, gtxt, (bx1, by1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, gc, 2)

                if frame_idx % 30 == 0:
                    self.alert_signal.emit(self.cam_id, f"유령 소실 경고 ID:{alert['track_id']} ({elapsed:.1f}초)", alert["level"])

            info_txt = f"Detect: {len(current_ids)} | Ghost: {len(ghost_tracker.ghosts)}"
            cv2.putText(frame, info_txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # ROI 설정 외곽선 그리기 전에 깨끗한 원본 복사
            clean_frame = frame.copy()
            
            # ROI 설정 외곽선 그리기 (UI 피드백용)
            if self.pool_polygon is not None and len(self.pool_polygon) > 0:
                cv2.polylines(frame, [self.pool_polygon], True, (255, 150, 0), 2)
            for ep in self.exit_polygons:
                if len(ep) > 0:
                    cv2.polylines(frame, [ep], True, (100, 255, 0), 2)

            # BGR -> RGB & QImage 변환 방출
            rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_img.shape
            bytes_per_line = ch * w
            drawn_qimg = QImage(rgb_img.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()
            
            # 깨끗한 프레임도 변환
            rgb_clean = cv2.cvtColor(clean_frame, cv2.COLOR_BGR2RGB)
            clean_qimg = QImage(rgb_clean.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()
            
            # 여기서 방출하지 않으면 UI가 갱신되지 않으므로 필수적으로 수행
            self.change_pixmap_signal.emit(self.cam_id, drawn_qimg, clean_qimg)

        cap.release()

    def stop(self):
        self._run_flag = False
        self.wait()
