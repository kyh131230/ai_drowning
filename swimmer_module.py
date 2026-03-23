import time
import numpy as np
from collections import deque

# ==========================================
# 1. 환경 프로필 시스템
# ==========================================
PROFILES = {
    "KIDS_POOL": {
        "name": "유아 풀",
        "normal_orientation": "vertical",
        "stationary_warning_sec": 3.0,
        "stationary_danger_sec": 6.0,
        "disappear_warning_sec": 3.0,
        "disappear_danger_sec": 6.0,
        "speed_threshold_ratio": 0.03,
        "ar_variance_threshold": 0.20,
        "w_area_decrease": 0.25,
        "w_stationary": 0.10,
        "w_ar_variance": 0.15,
        "w_direction_variance": 0.10,
        "w_sinking": 0.20,
        "w_disappear": 0.20,
    },
    "LANE_POOL": {
        "name": "레인 수영장",
        "normal_orientation": "horizontal",
        "stationary_warning_sec": 4.0,
        "stationary_danger_sec": 7.0,
        "disappear_warning_sec": 3.0,
        "disappear_danger_sec": 5.0,
        "speed_threshold_ratio": 0.05,
        "ar_variance_threshold": 0.15,
        "w_area_decrease": 0.20,
        "w_stationary": 0.20,
        "w_ar_variance": 0.15,
        "w_direction_variance": 0.10,
        "w_sinking": 0.20,
        "w_disappear": 0.15,
    },
}

HISTORY_LENGTH = 60
RISK_WARNING = 0.35
RISK_DANGER = 0.60
GHOST_EXPIRE_SEC = 30.0


# ==========================================
# 2. 수영자 모니터
# ==========================================
class SwimmerMonitor:
    def __init__(self, track_id, profile):
        self.id = track_id
        self.profile = profile

        self.center_history = deque(maxlen=HISTORY_LENGTH)
        self.ar_history = deque(maxlen=HISTORY_LENGTH)
        self.area_history = deque(maxlen=HISTORY_LENGTH)
        self.bbox_history = deque(maxlen=HISTORY_LENGTH)
        self.time_history = deque(maxlen=HISTORY_LENGTH)

        self.state = "SAFE"
        self.risk_score = 0.0
        self.risk_history = deque(maxlen=10)
        self.last_seen = time.time()
        self.stationary_start_time = None
        self.debug = {}

    def update(self, box_xyxy, kp_xy=None, kp_conf=None):
        now = time.time()
        self.last_seen = now

        x1, y1, x2, y2 = map(float, box_xyxy)
        w = max(x2 - x1, 1)
        h = max(y2 - y1, 1)
        center = ((x1 + x2) / 2, (y1 + y2) / 2)
        area = w * h
        ar = h / w

        self.center_history.append(center)
        self.ar_history.append(ar)
        self.area_history.append(area)
        self.bbox_history.append((x1, y1, x2, y2))
        self.time_history.append(now)

        sig_area = self._signal_area_decrease()
        sig_stop = self._signal_stationary(h)
        sig_ar = self._signal_ar_variance()
        sig_dir = self._signal_direction_variance()
        sig_sink = self._signal_sinking()
        pose_mod = self._pose_bonus(kp_xy, kp_conf)

        p = self.profile
        raw = (
            sig_area * p["w_area_decrease"]
            + sig_stop * p["w_stationary"]
            + sig_ar   * p["w_ar_variance"]
            + sig_dir  * p["w_direction_variance"]
            + sig_sink * p["w_sinking"]
        )
        raw += pose_mod
        raw = max(0.0, min(1.0, raw))

        self.risk_history.append(raw)
        avg_risk = float(np.mean(list(self.risk_history)))
        self.risk_score = avg_risk

        if avg_risk >= RISK_DANGER:
            self.state = "DANGER"
            level = 2
        elif avg_risk >= RISK_WARNING:
            self.state = "WARNING"
            level = 1
        else:
            self.state = "SAFE"
            level = 0

        speed = self.debug.get("speed", 0)
        self.debug = {
            "speed": speed,
            "ar": ar,
            "sig_area": sig_area,
            "sig_stop": sig_stop,
            "sig_ar": sig_ar,
            "sig_dir": sig_dir,
            "sig_sink": sig_sink,
            "pose_mod": pose_mod,
            "raw": raw,
            "risk": avg_risk,
        }
        return self.state, level

    def _signal_area_decrease(self):
        areas = list(self.area_history)
        if len(areas) < 10: return 0.0
        mid = len(areas) // 2
        early_avg = float(np.mean(areas[:mid]))
        late_avg = float(np.mean(areas[mid:]))
        if early_avg <= 0: return 0.0
        ratio = (early_avg - late_avg) / early_avg
        return max(0.0, min(1.0, ratio / 0.30))

    def _signal_stationary(self, box_h):
        centers = list(self.center_history)
        if len(centers) < 3: return 0.0
        speeds = []
        for i in range(1, min(6, len(centers))):
            d = np.linalg.norm(np.array(centers[-i]) - np.array(centers[-(i + 1)]))
            speeds.append(d)
        avg_speed = float(np.mean(speeds)) if speeds else 0.0
        self.debug["speed"] = avg_speed

        threshold = max(2.0, box_h * self.profile["speed_threshold_ratio"])
        is_stationary = avg_speed < threshold

        now = time.time()
        if not is_stationary:
            self.stationary_start_time = None
            return 0.0

        if self.stationary_start_time is None:
            self.stationary_start_time = now

        elapsed = now - self.stationary_start_time
        warn = self.profile["stationary_warning_sec"]
        dang = self.profile["stationary_danger_sec"]

        if elapsed < warn: return 0.0
        elif elapsed < dang: return (elapsed - warn) / (dang - warn) * 0.7
        else: return 1.0

    def _signal_ar_variance(self):
        ars = list(self.ar_history)
        if len(ars) < 5: return 0.0
        recent = ars[-15:]
        std = float(np.std(recent))
        cap = self.profile["ar_variance_threshold"] * 2
        return min(1.0, std / cap)

    def _signal_direction_variance(self):
        centers = list(self.center_history)
        if len(centers) < 5: return 0.0
        pts = centers[-10:]
        angles = []
        for i in range(1, len(pts)):
            dx = pts[i][0] - pts[i - 1][0]
            dy = pts[i][1] - pts[i - 1][1]
            if abs(dx) > 0.5 or abs(dy) > 0.5:
                angles.append(np.arctan2(dy, dx))
        if len(angles) < 2: return 0.0
        return min(1.0, float(np.std(angles)) / np.pi)

    def _signal_sinking(self):
        bboxes = list(self.bbox_history)
        if len(bboxes) < 10: return 0.0
        mid = len(bboxes) // 2
        early_y1 = float(np.mean([b[1] for b in bboxes[:mid]]))
        late_y1 = float(np.mean([b[1] for b in bboxes[mid:]]))
        early_h = float(np.mean([b[3] - b[1] for b in bboxes[:mid]]))
        if early_h <= 0: return 0.0
        sink_ratio = (late_y1 - early_y1) / early_h
        return max(0.0, min(1.0, sink_ratio / 0.50))

    def _pose_bonus(self, kp_xy, kp_conf):
        if kp_xy is None or kp_conf is None: return 0.0
        if len(kp_xy) < 11 or len(kp_conf) < 11: return 0.0
        mod = 0.0
        nose_conf = float(kp_conf[0])
        if nose_conf > 0.5: mod -= 0.05
        l_wr_conf, r_wr_conf = float(kp_conf[9]), float(kp_conf[10])
        if l_wr_conf > 0.4 and r_wr_conf > 0.4:
            shoulder_y = (float(kp_xy[5][1]) + float(kp_xy[6][1])) / 2
            l_wr_y, r_wr_y = float(kp_xy[9][1]), float(kp_xy[10][1])
            if l_wr_y > shoulder_y and r_wr_y > shoulder_y and nose_conf < 0.3:
                mod += 0.08
        return mod


# ==========================================
# 3. 유령 트래커
# ==========================================
class GhostTracker:
    def __init__(self):
        self.ghosts = {}

    def mark_disappeared(self, track_id, last_bbox, last_state, last_risk):
        if track_id not in self.ghosts:
            self.ghosts[track_id] = {
                "bbox": last_bbox,
                "state": last_state,
                "risk_at_disappear": last_risk,
                "time": time.time(),
                "alerted_danger": False,
            }

    def mark_alive(self, track_id):
        self.ghosts.pop(track_id, None)

    @staticmethod
    def _iou(box_a, box_b):
        x1, y1 = max(box_a[0], box_b[0]), max(box_a[1], box_b[1])
        x2, y2 = min(box_a[2], box_b[2]), min(box_a[3], box_b[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    def try_match_new_detection(self, new_bbox, iou_thresh=0.2):
        best_tid, best_iou = None, 0.0
        for tid, info in self.ghosts.items():
            score = self._iou(new_bbox, info["bbox"])
            if score > best_iou:
                best_iou = score
                best_tid = tid
        if best_tid is not None and best_iou >= iou_thresh:
            return best_tid, self.ghosts.pop(best_tid)
        return None, None

    def get_alerts(self, profile):
        alerts, expired = [], []
        now = time.time()
        for tid, info in self.ghosts.items():
            elapsed = now - info["time"]
            if elapsed > GHOST_EXPIRE_SEC:
                expired.append(tid)
                continue
            if elapsed > profile["disappear_danger_sec"]:
                alerts.append({"track_id": tid, "bbox": info["bbox"], "elapsed": elapsed, "level": 2})
                info["alerted_danger"] = True
            elif elapsed > profile["disappear_warning_sec"]:
                alerts.append({"track_id": tid, "bbox": info["bbox"], "elapsed": elapsed, "level": 1})
        for tid in expired:
            del self.ghosts[tid]
        return alerts
