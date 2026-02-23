import cv2
import numpy as np
import time
import threading
from collections import deque
from ultralytics import YOLO
import supervision as sv

# ==========================================
# 1. 설정 및 상수 정의
# ==========================================
MODEL_PATH = "yolo26s_openvino_model/"
CONF_THRESHOLD = 0.25
SKIP_FRAMES = 3
IMG_SIZE = 640


# ---- 슬라이딩 풀 전용 파라미터 ----
SUBMERGE_WARNING_SEC = 3.0      # 수면 아래 경고 (초)
SUBMERGE_DANGER_SEC = 5.0       # 수면 아래 위험 (초)
GHOST_EXPIRE_SEC = 30.0         # 유령 만료 (풀에서 나간 것으로 간주)
EXIT_PROXIMITY_RATIO = 0.15     # 탈출 영역 근처 판정 비율 (대각선 대비)
EXIT_DIRECTION_FRAMES = 8       # 이동 방향 판단에 사용할 최근 프레임 수
HISTORY_LENGTH = 60             # 히스토리 버퍼 크기


# ==========================================
# 2. ROI 설정기 (마우스 클릭으로 영역 지정)
#    풀 영역 1개 + 탈출 영역 N개(여러 곳 가능)
# ==========================================
class ROISelector:
    """
    프로그램 시작 시 마우스 클릭으로 영역(폴리곤)을 설정합니다.
      1. 풀 영역 (Pool Zone) — 미끄럼틀 출구부터 물이 있는 전체 수영 구역
      2. 탈출 영역 (Exit Zone) — 풀에서 밖으로 나가는 통로/계단 (여러 곳 가능)

    [풀 영역 설정 팁]
      미끄럼틀에서 물로 빠지는 출구 지점부터, 아이들이 수영하거나
      걸어다닐 수 있는 물 영역 전체를 포함하도록 그려주세요.

    [탈출 영역 설정 팁]
      풀에서 밖으로 나가는 계단, 사다리, 통로 부분을 그려주세요.
      양쪽에 출구가 있으면 각각 따로 그릴 수 있습니다.
    """

    def __init__(self):
        self.pool_polygon = None           # np.array, shape (N, 2)
        self.exit_polygons = []            # [np.array, ...] — 여러 탈출 영역
        self._current_points = []
        self._done = False
        self._frame = None
        self._original_frame = None

    def select(self, frame):
        """영상 첫 프레임에서 ROI를 설정합니다."""
        self._original_frame = frame.copy()
        self._frame = frame.copy()

        cv2.namedWindow("ROI Setup", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("ROI Setup", 1280, 720)

        #
        # ---- Phase 1: 풀 영역 ----
        #
        self._current_points = []
        self._done = False
        cv2.setMouseCallback("ROI Setup", self._mouse_callback)

        while not self._done:
            display = self._frame.copy()
            guide = "[1] POOL - Left-click: add point / Right-click: finish"
            sub_guide = "R: reset | ESC: cancel"
            cv2.putText(display, guide, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display, sub_guide, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            for i, pt in enumerate(self._current_points):
                cv2.circle(display, pt, 5, (255, 200, 0), -1)
                if i > 0:
                    cv2.line(display, self._current_points[i-1], pt, (255, 200, 0), 2)

            cv2.imshow("ROI Setup", display)
            key = cv2.waitKey(30) & 0xFF
            if key == ord('r') or key == ord('R'):
                self._current_points = []
            elif key == 27:
                cv2.destroyWindow("ROI Setup")
                return False

        self.pool_polygon = np.array(self._current_points, dtype=np.int32)

        #
        # ---- Phase 2+: 탈출 영역 (여러 곳 반복) ----
        #
        exit_idx = 1
        while True:
            self._frame = self._original_frame.copy()
            # 이미 설정된 영역 표시
            cv2.polylines(self._frame, [self.pool_polygon], True, (255, 200, 0), 2)
            for ep in self.exit_polygons:
                cv2.polylines(self._frame, [ep], True, (0, 255, 100), 2)

            self._current_points = []
            self._done = False
            cv2.setMouseCallback("ROI Setup", self._mouse_callback)

            while not self._done:
                display = self._frame.copy()
                guide = f"[EXIT #{exit_idx}] Left-click: add point / Right-click: finish"
                sub_guide = "R: reset | S: skip (no more exits) | ESC: cancel"
                cv2.putText(display, guide, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(display, sub_guide, (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

                for i, pt in enumerate(self._current_points):
                    cv2.circle(display, pt, 5, (0, 255, 100), -1)
                    if i > 0:
                        cv2.line(display, self._current_points[i-1], pt, (0, 255, 100), 2)

                # 기존 Pool/Exit 표시
                overlay = display.copy()
                cv2.fillPoly(overlay, [self.pool_polygon], (255, 200, 0))
                for ep in self.exit_polygons:
                    cv2.fillPoly(overlay, [ep], (0, 200, 100))
                cv2.addWeighted(overlay, 0.2, display, 0.8, 0, display)

                cv2.imshow("ROI Setup", display)
                key = cv2.waitKey(30) & 0xFF
                if key == ord('r') or key == ord('R'):
                    self._current_points = []
                elif key == ord('s') or key == ord('S'):
                    # 더 이상 탈출 영역 없음
                    self._done = True
                    self._current_points = []  # 빈 상태로 종료
                elif key == 27:
                    cv2.destroyWindow("ROI Setup")
                    return False

            # S 키로 종료한 경우 (포인트 없음)
            if len(self._current_points) < 3:
                break

            self.exit_polygons.append(
                np.array(self._current_points, dtype=np.int32)
            )
            exit_idx += 1

        cv2.destroyWindow("ROI Setup")

        if len(self.exit_polygons) == 0:
            print("[WARNING] 탈출 영역이 설정되지 않았습니다. 모든 소실을 위험으로 판단합니다.")

        return True

    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self._current_points.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN:
            if len(self._current_points) >= 3:
                self._done = True

    def is_in_pool(self, point):
        """좌표가 풀 영역 내인지 판별"""
        if self.pool_polygon is None:
            return False
        return cv2.pointPolygonTest(self.pool_polygon, point, False) >= 0

    def is_in_exit(self, point):
        """좌표가 탈출 영역(들) 중 하나 안에 있는지 판별"""
        for ep in self.exit_polygons:
            if cv2.pointPolygonTest(ep, point, False) >= 0:
                return True
        return False

    def is_near_exit(self, point, frame_shape):
        """좌표가 탈출 영역(들) 중 하나 근처인지 판별"""
        diag = np.sqrt(frame_shape[0]**2 + frame_shape[1]**2)
        threshold = diag * EXIT_PROXIMITY_RATIO
        for ep in self.exit_polygons:
            exit_center = np.mean(ep, axis=0)
            dist = np.linalg.norm(np.array(point) - exit_center)
            if dist < threshold:
                return True
        return False

    def get_nearest_exit_center(self, point):
        """주어진 점에서 가장 가까운 탈출 영역의 중심 좌표 반환"""
        if not self.exit_polygons:
            return None
        best_center = None
        best_dist = float('inf')
        for ep in self.exit_polygons:
            center = np.mean(ep, axis=0)
            dist = np.linalg.norm(np.array(point) - center)
            if dist < best_dist:
                best_dist = dist
                best_center = tuple(center.astype(int))
        return best_center

    def get_all_exit_centers(self):
        """모든 탈출 영역의 중심 좌표 리스트 반환"""
        return [tuple(np.mean(ep, axis=0).astype(int)) for ep in self.exit_polygons]


# ==========================================
# 3. 슬라이딩 풀 모니터 (사람별 상태 관리)
#    ★ 시간 계산은 영상 FPS 기반 (프레임 카운트)
# ==========================================
class SlidePoolMonitor:
    """
    슬라이딩 풀에서 각 사람(track_id)의 상태를 관리합니다.
    시간은 time.time()이 아니라, 영상 FPS 기반 프레임 카운트로 계산합니다.

    상태:
      - NEW_IN_POOL : 풀에 방금 나타남 (슬라이드에서 내려온 것으로 추정)
      - IN_POOL     : 풀 안에서 활동 중 (정상)
      - MISSING     : 풀 내에서 감지 안 됨 (아직 경고 전)
      - WARNING     : 3초 이상 미감지 (경고)
      - DANGER      : 5초 이상 미감지 (위험 — 익사 가능)
      - SAFE_EXIT   : 탈출 영역으로 걸어나감 (정상 퇴장)
    """

    def __init__(self, track_id, video_fps, skip_frames, use_wall_clock=False):
        self.id = track_id
        self.video_fps = video_fps
        self.skip_frames = skip_frames
        self.use_wall_clock = use_wall_clock
        # FILE 모드용: 프레임 간의 시간 간격 (초)
        self.sec_per_step = skip_frames / video_fps

        # ---- 상태 ----
        self.state = "NEW_IN_POOL"
        self.last_seen_step = 0
        self.disappeared_step = None
        self.current_step = 0

        # ---- RTSP 모드용: 벽시계 시간 ----
        self.last_seen_wall_time = time.time()
        self.disappeared_wall_time = None

        # ---- 위치 히스토리 ----
        self.center_history = deque(maxlen=HISTORY_LENGTH)
        self.bbox_history = deque(maxlen=HISTORY_LENGTH)
        self.area_history = deque(maxlen=HISTORY_LENGTH)

        # ---- 마지막 감지 정보 ----
        self.last_center = None
        self.last_bbox = None
        self.was_in_exit = False
        self.was_near_exit = False
        self.moving_toward_exit = False

    def update_detected(self, box_xyxy, roi_selector, frame_shape, step, wall_time=None):
        """Detection이 있을 때 호출 — 상태 업데이트"""
        self.current_step = step
        self.last_seen_step = step
        self.disappeared_step = None
        if wall_time is not None:
            self.last_seen_wall_time = wall_time
            self.disappeared_wall_time = None

        x1, y1, x2, y2 = map(float, box_xyxy)
        center = ((x1 + x2) / 2, (y1 + y2) / 2)
        area = (x2 - x1) * (y2 - y1)

        self.center_history.append(center)
        self.bbox_history.append((x1, y1, x2, y2))
        self.area_history.append(area)
        self.last_center = center
        self.last_bbox = (x1, y1, x2, y2)

        # ---- 위치 판별 ----
        in_exit = roi_selector.is_in_exit(center)
        near_exit = roi_selector.is_near_exit(center, frame_shape)

        self.was_in_exit = in_exit
        self.was_near_exit = near_exit
        self.moving_toward_exit = self._check_moving_toward_exit(roi_selector)

        # ---- 상태 전이 ----
        if in_exit:
            self.state = "SAFE_EXIT"
        elif self.state == "SAFE_EXIT":
            # 풀에서 다시 감지됨 → SAFE_EXIT 취소, 풀로 복귀
            self.state = "IN_POOL"
        elif self.state in ("MISSING", "WARNING", "DANGER"):
            self.state = "IN_POOL"
        elif self.state == "NEW_IN_POOL":
            self.state = "IN_POOL"

        return self.state

    def update_missing(self, roi_selector, frame_shape, step, wall_time=None):
        """Detection이 없을 때 호출 — 사라짐 처리"""
        self.current_step = step

        if self.state == "SAFE_EXIT":
            return self.state, 0.0

        # 첫 사라짐
        if self.disappeared_step is None:
            self.disappeared_step = step
            if wall_time is not None:
                self.disappeared_wall_time = wall_time

            if self.was_in_exit:
                self.state = "SAFE_EXIT"
                return self.state, 0.0

            if self.was_near_exit and self.moving_toward_exit:
                self.state = "SAFE_EXIT"
                return self.state, 0.0

        # ★ 경과 시간 계산 (RTSP=벽시계, FILE=FPS기반)
        if self.use_wall_clock and wall_time and self.disappeared_wall_time:
            elapsed_sec = wall_time - self.disappeared_wall_time
        else:
            elapsed_sec = (step - self.disappeared_step) * self.sec_per_step

        if elapsed_sec >= SUBMERGE_DANGER_SEC:
            self.state = "DANGER"
        elif elapsed_sec >= SUBMERGE_WARNING_SEC:
            self.state = "WARNING"
        else:
            self.state = "MISSING"

        return self.state, elapsed_sec

    def _check_moving_toward_exit(self, roi_selector):
        """최근 프레임에서 탈출 영역 방향으로 이동 중인지 판단"""
        if self.last_center is None:
            return False

        exit_center = roi_selector.get_nearest_exit_center(self.last_center)
        if exit_center is None:
            return False

        centers = list(self.center_history)
        if len(centers) < 3:
            return False

        recent = centers[-min(EXIT_DIRECTION_FRAMES, len(centers)):]
        if len(recent) < 3:
            return False

        dx = recent[-1][0] - recent[0][0]
        dy = recent[-1][1] - recent[0][1]
        move_dist = np.sqrt(dx**2 + dy**2)
        if move_dist < 5:
            return False

        ex_dx = exit_center[0] - recent[0][0]
        ex_dy = exit_center[1] - recent[0][1]
        ex_dist = np.sqrt(ex_dx**2 + ex_dy**2)
        if ex_dist < 1:
            return True

        cos_sim = (dx * ex_dx + dy * ex_dy) / (move_dist * ex_dist)
        return cos_sim > 0.5

    def get_missing_elapsed_sec(self, step, wall_time=None):
        """사라진 경과 시간(초) 반환"""
        if self.disappeared_step is None:
            return 0.0
        if self.use_wall_clock and wall_time and self.disappeared_wall_time:
            return wall_time - self.disappeared_wall_time
        return (step - self.disappeared_step) * self.sec_per_step

    def get_elapsed_since_seen(self, step, wall_time=None):
        """마지막 감지 이후 경과 시간(초) 반환"""
        if self.use_wall_clock and wall_time:
            return wall_time - self.last_seen_wall_time
        return (step - self.last_seen_step) * self.sec_per_step

    def get_risk_level(self):
        """위험 레벨 반환 (0=안전, 1=경고, 2=위험)"""
        if self.state == "DANGER":
            return 2
        elif self.state == "WARNING":
            return 1
        else:
            return 0


# ==========================================
# 4. 고스트 매칭 헬퍼
#    새 Detection이 기존 고스트(실종자) 근처에 나타나면
#    같은 사람으로 판단 → 고스트 제거하여 오탐 방지
# ==========================================
def _iou(box_a, box_b):
    """두 bbox의 IoU(겹침 비율) 계산"""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _match_ghost(new_xyxy, new_center, monitors, frame_shape):
    """
    새 Detection(new_xyxy)이 기존 고스트(MISSING/WARNING/DANGER) 중
    가장 적합한 것과 매칭을 시도합니다.

    매칭 조건 (둘 중 하나 충족):
      1. IoU >= 0.15 (bbox 겹침)
      2. 중심 거리 < 화면 대각선 10% 이내

    Returns:
        매칭된 고스트의 track_id, 없으면 None
    """
    ghosts = {}
    for tid, mon in monitors.items():
        if mon.state not in ("MISSING", "WARNING", "DANGER"):
            continue
        if mon.last_bbox is None:
            continue
        ghosts[tid] = mon

    if not ghosts:
        return None

    diag = np.sqrt(frame_shape[0]**2 + frame_shape[1]**2)
    dist_threshold = diag * 0.10

    best_tid = None
    best_score = 0.0

    for tid, mon in ghosts.items():
        # IoU 체크
        iou = _iou(new_xyxy, mon.last_bbox)
        if iou >= 0.15:
            if iou > best_score:
                best_score = iou
                best_tid = tid
            continue

        # 거리 체크
        if mon.last_center is not None:
            dist = np.linalg.norm(
                np.array(new_center) - np.array(mon.last_center)
            )
            if dist < dist_threshold:
                score = 1.0 - (dist / dist_threshold)
                if score > best_score:
                    best_score = score
                    best_tid = tid

    return best_tid


# ==========================================
# 5. RTSP 스레드 프레임 리더
#    RTSP 스트림에서 항상 최신 프레임만 유지하여
#    버퍼 밀림 현상을 완전 해결
# ==========================================
class RTSPFrameReader:
    """
    별도 스레드에서 RTSP 스트림을 계속 읽어
    항상 최신 프레임만 유지합니다.
    메인 스레드는 read()로 최신 프레임을 가져옵니다.
    """

    def __init__(self, source_path):
        self.cap = cv2.VideoCapture(source_path)
        self.lock = threading.Lock()
        self.frame = None
        self.ret = False
        self.running = False
        self._thread = None

    def start(self):
        """백그라운드 스레드 시작"""
        self.running = True
        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()
        # 첫 프레임 대기
        time.sleep(0.5)
        return self

    def _reader_loop(self):
        """계속 프레임을 읽어 최신 것만 유지"""
        while self.running:
            ret, frame = self.cap.read()
            with self.lock:
                self.ret = ret
                self.frame = frame
            if not ret:
                break

    def read(self):
        """최신 프레임 반환 (밀림 없음)"""
        with self.lock:
            return self.ret, self.frame.copy() if self.frame is not None else None

    def get(self, prop):
        return self.cap.get(prop)

    def isOpened(self):
        return self.cap.isOpened()

    def release(self):
        self.running = False
        if self._thread is not None:
            self._thread.join(timeout=2)
        self.cap.release()

    def set(self, prop, val):
        self.cap.set(prop, val)


# ==========================================
# 6. 메인 파이프라인
# ==========================================
def run_sliding_pool_system(source_path):
    # ---- 소스 타입 감지 ----
    is_rtsp = isinstance(source_path, str) and (
        source_path.lower().startswith("rtsp://")
        or source_path.lower().startswith("rtmp://")
        or source_path.lower().startswith("http://")
        or source_path.lower().startswith("https://")
    )
    source_type = "RTSP" if is_rtsp else "FILE"

    print("=" * 55)
    print("  슬라이딩 풀 익사 감지 시스템 v1.2")
    print(f"  소스: {source_type} — {source_path}")
    print(f"  모델: {MODEL_PATH}")
    print(f"  경고: {SUBMERGE_WARNING_SEC}초 | 위험: {SUBMERGE_DANGER_SEC}초")
    print(f"  Conf: {CONF_THRESHOLD} | Skip: {SKIP_FRAMES} | ImgSz: {IMG_SIZE}")
    print("=" * 55)

    # ---- 모델 로드 ----
    is_pose_model = "pose" in MODEL_PATH.lower()
    try:
        if is_pose_model:
            model = YOLO(MODEL_PATH, task="pose")
            print("  Pose 모델 로드 완료")
        else:
            model = YOLO(MODEL_PATH)
            print("  Detection 모델 로드 완료")
    except Exception as e:
        print(f"모델 로드 오류: {e}")
        return

    # ---- 영상/스트림 열기 ----
    if is_rtsp:
        # RTSP: 스레드 프레임 리더 사용 (버퍼 밀림 방지)
        cap = RTSPFrameReader(source_path).start()
        print("  RTSP 스레드 리더 시작 (최신 프레임만 사용)")
    else:
        # 파일: 일반 VideoCapture
        cap = cv2.VideoCapture(source_path)

    if not cap.isOpened():
        print(f"영상 열기 실패: {source_path}")
        return

    # ★ FPS 및 시간 계산 방식 결정
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0:
        video_fps = 30.0

    if is_rtsp:
        # RTSP: 실제 벽시계 사용 (추론 속도와 무관하게 정확)
        sec_per_step = None  # 사용 안 함
        print(f"  시간 계산: time.time() (실시간)")
    else:
        # 파일: FPS 기반 계산 (영상 속 시간 기준)
        sec_per_step = SKIP_FRAMES / video_fps
        print(f"  영상 FPS: {video_fps:.1f}")
        print(f"  시간 계산: FPS 기반 ({sec_per_step:.3f}초/스텝)")

    ret, first_frame = cap.read()
    if not ret:
        print("첫 프레임 읽기 실패")
        return

    # ---- ROI 설정 ----
    print("\n[ROI 설정]")
    print("  1단계: 풀 영역 — 미끄럼틀 출구에서 물 전체 구역을 감싸주세요")
    print("  2단계: 탈출 영역 — 풀에서 나가는 계단/통로 (여러 곳 가능)")
    print("")
    print("  조작법:")
    print("    왼쪽 클릭 = 꼭짓점 추가")
    print("    오른쪽 클릭 = 해당 영역 완료")
    print("    R = 현재 영역 다시 그리기")
    print("    S = 탈출 영역 추가 종료 (더 이상 없을 때)")
    print("    ESC = 취소\n")

    roi_selector = ROISelector()
    if not roi_selector.select(first_frame):
        print("ROI 설정이 취소되었습니다.")
        cap.release()
        return

    print("ROI 설정 완료!")
    print(f"  풀 영역: {len(roi_selector.pool_polygon)} 꼭짓점")
    print(f"  탈출 영역: {len(roi_selector.exit_polygons)}개")
    for i, ep in enumerate(roi_selector.exit_polygons):
        print(f"    EXIT #{i+1}: {len(ep)} 꼭짓점")

    # 프레임 크기
    frame_h, frame_w = first_frame.shape[:2]
    frame_shape = (frame_h, frame_w)

    # ---- 트래커 ----
    tracker = sv.ByteTrack(
        track_activation_threshold=0.2,
        lost_track_buffer=30,
        frame_rate=int(video_fps),
    )

    monitors = {}           # track_id → SlidePoolMonitor
    prev_track_ids = set()
    frame_idx = 0
    step_count = 0          # 실제 처리된 스텝 수
    danger_flash = 0
    last_step_time = time.time()  # RTSP용: 마지막 스텝의 벽시계 시간

    # ---- 영상 파일: 처음으로 되감기 ----
    if not is_rtsp:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            if is_rtsp:
                continue  # RTSP는 일시 끊김 가능 → 재시도
            break

        frame_idx += 1
        # 파일 모드: SKIP_FRAMES마다 처리 / RTSP: 무조건 최신 프레임
        if not is_rtsp and frame_idx % SKIP_FRAMES != 0:
            continue

        step_count += 1
        danger_flash += 1
        current_wall_time = time.time()

        # ---- 1. 추론 ----
        results = model(
            frame, imgsz=IMG_SIZE, verbose=False,
            conf=CONF_THRESHOLD, device="cpu",
        )
        result = results[0]

        # ---- 2. Detection 데이터 ----
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

        # 사람만 필터
        if hasattr(detections, "class_id") and detections.class_id is not None:
            detections = detections[detections.class_id == 0]

        detections = tracker.update_with_detections(detections)

        # ---- 3. 각 Detection 업데이트 ----
        current_ids = set()

        for xyxy, mask, confidence, class_id, track_id, data in detections:
            if track_id is None:
                continue

            current_ids.add(track_id)

            cx = (xyxy[0] + xyxy[2]) / 2
            cy = (xyxy[1] + xyxy[3]) / 2
            center = (cx, cy)

            in_pool = roi_selector.is_in_pool(center)
            in_exit = roi_selector.is_in_exit(center)

            if not in_pool and not in_exit:
                if track_id in monitors:
                    monitors[track_id].update_detected(
                        xyxy, roi_selector, frame_shape, step_count
                    )
                continue

            if track_id not in monitors:
                # ★ 고스트 매칭: 새 ID가 나타났을 때,
                #   기존 MISSING/WARNING/DANGER 고스트 중
                #   가까운 위치에 있던 것과 매칭 → 오탐 방지
                matched_ghost_id = _match_ghost(
                    xyxy, center, monitors, frame_shape
                )
                if matched_ghost_id is not None:
                    del monitors[matched_ghost_id]

                monitors[track_id] = SlidePoolMonitor(
                    track_id, video_fps, SKIP_FRAMES,
                    use_wall_clock=is_rtsp
                )

            state = monitors[track_id].update_detected(
                xyxy, roi_selector, frame_shape,
                step_count, current_wall_time
            )

            # ---- 시각화: 감지된 사람 ----
            x1, y1, x2, y2 = map(int, xyxy)
            mon = monitors[track_id]
            level = mon.get_risk_level()

            if state == "SAFE_EXIT":
                color = (200, 200, 200)
            elif level == 0:
                color = (0, 200, 0)
            else:
                color = (0, 220, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            label = f"ID:{track_id} {state}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(frame, (x1, y1 - 20), (x1 + tw, y1), color, -1)
            cv2.putText(frame, label, (x1 + 2, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        # ---- 4. 소실 감지 ----
        disappeared = prev_track_ids - current_ids
        for tid in disappeared:
            if tid in monitors:
                mon = monitors[tid]
                if mon.state == "SAFE_EXIT":
                    continue
                mon.update_missing(roi_selector, frame_shape,
                                   step_count, current_wall_time)

        # 이미 MISSING/WARNING/DANGER인 모니터도 계속 업데이트
        for tid, mon in list(monitors.items()):
            if tid not in current_ids and tid not in disappeared:
                if mon.state in ("MISSING", "WARNING", "DANGER"):
                    mon.update_missing(roi_selector, frame_shape,
                                       step_count, current_wall_time)

        prev_track_ids = current_ids.copy()

        # ---- 5. 고스트 박스 시각화 (사라진 사람 — MISSING/WARNING/DANGER 모두) ----
        for tid, mon in monitors.items():
            if mon.state in ("MISSING", "WARNING", "DANGER") and mon.last_bbox is not None:
                bx1, by1, bx2, by2 = map(int, mon.last_bbox)
                elapsed = mon.get_missing_elapsed_sec(
                    step_count, current_wall_time
                )
                level = mon.get_risk_level()

                # 상태별 색상 + 텍스트
                if level == 2:  # DANGER (5초+)
                    gc = (0, 0, 255) if danger_flash % 4 < 2 else (0, 0, 180)
                    alert_text = f"!! DROWNING RISK !! ID:{tid} {elapsed:.1f}s"
                elif level == 1:  # WARNING (3초+)
                    gc = (0, 140, 255)  # 주황
                    alert_text = f"SUBMERGED? ID:{tid} {elapsed:.1f}s"
                else:  # MISSING (3초 미만)
                    gc = (200, 200, 200)  # 흰색/회색
                    alert_text = f"LOST ID:{tid} {elapsed:.1f}s"

                # 점선 사각형 (고스트 박스)
                for i in range(bx1, bx2, 10):
                    cv2.line(frame, (i, by1), (min(i + 5, bx2), by1), gc, 2)
                    cv2.line(frame, (i, by2), (min(i + 5, bx2), by2), gc, 2)
                for i in range(by1, by2, 10):
                    cv2.line(frame, (bx1, i), (bx1, min(i + 5, by2)), gc, 2)
                    cv2.line(frame, (bx2, i), (bx2, min(i + 5, by2)), gc, 2)

                # 경과 시간 텍스트
                cv2.putText(frame, alert_text, (bx1, by1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, gc, 2)

                # DANGER일 때 화면 하단 대형 경고
                if level == 2 and danger_flash % 4 < 3:
                    warn_msg = f"[DANGER] ID:{tid} - {elapsed:.1f}s UNDERWATER!"
                    cv2.putText(frame, warn_msg, (10, frame_h - 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)

        # ---- 6. ROI 영역 오버레이 ----
        overlay = frame.copy()

        if roi_selector.pool_polygon is not None:
            cv2.fillPoly(overlay, [roi_selector.pool_polygon], (255, 150, 0))
            cv2.polylines(frame, [roi_selector.pool_polygon], True, (255, 200, 0), 2)

        for ep in roi_selector.exit_polygons:
            cv2.fillPoly(overlay, [ep], (0, 200, 100))
            cv2.polylines(frame, [ep], True, (0, 255, 100), 2)

        cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)

        # ROI 라벨
        if roi_selector.pool_polygon is not None:
            pool_center = tuple(np.mean(roi_selector.pool_polygon, axis=0).astype(int))
            cv2.putText(frame, "POOL", pool_center,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)

        for i, ec in enumerate(roi_selector.get_all_exit_centers()):
            label = f"EXIT#{i+1}" if len(roi_selector.exit_polygons) > 1 else "EXIT"
            cv2.putText(frame, label, ec,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 100), 2)

        # ---- 7. 상단 정보 표시 ----
        if is_rtsp:
            time_label = "LIVE"
        else:
            video_time_sec = frame_idx / video_fps
            video_min = int(video_time_sec // 60)
            video_sec = video_time_sec % 60
            time_label = f"{video_min}:{video_sec:04.1f}"

        active_count = sum(1 for m in monitors.values()
                          if m.state not in ("SAFE_EXIT",))
        warning_count = sum(1 for m in monitors.values()
                           if m.state == "WARNING")
        danger_count = sum(1 for m in monitors.values()
                          if m.state == "DANGER")

        info_txt = (
            f"[Sliding Pool {'RTSP' if is_rtsp else 'FILE'}] "
            f"Time: {time_label} | "
            f"Active: {active_count} | "
            f"Warning: {warning_count} | "
            f"Danger: {danger_count}"
        )
        cv2.putText(frame, info_txt, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        # ---- 8. 만료된 모니터 정리 ----
        expired_ids = []
        for tid, mon in monitors.items():
            if mon.state == "SAFE_EXIT":
                safe_elapsed = mon.get_elapsed_since_seen(
                    step_count, current_wall_time
                )
                if safe_elapsed > 5.0:
                    expired_ids.append(tid)
            elif mon.state in ("MISSING", "WARNING", "DANGER"):
                if mon.get_missing_elapsed_sec(
                    step_count, current_wall_time
                ) > GHOST_EXPIRE_SEC:
                    expired_ids.append(tid)

        for tid in expired_ids:
            del monitors[tid]

        # ---- 출력 ----
        display = cv2.resize(frame, (1280, 720))
        cv2.imshow("Sliding Pool - Drowning Detection", display)

        if cv2.waitKey(1) == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # =============================================
    # 소스 선택: 영상 파일 또는 RTSP 주소
    # =============================================

    # ▼ 영상 파일 테스트
    run_sliding_pool_system("./source/water_slide_test_3.mp4")

    # ▼ RTSP 스트림 (실제 사용 시 아래 주석 해제)
    # run_sliding_pool_system("rtsp://admin:password@192.168.0.100:554/stream1")
