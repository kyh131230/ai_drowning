"""
AI 익사 감지 시스템 - FastAPI 메인 서버 (다중 카메라 버전)
"""
import os
import cv2
import time
import threading
import socket
import sys
import numpy as np
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, Request, Form, Body
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
import uvicorn
import webbrowser
from threading import Timer

import supervision as sv
from ultralytics import YOLO

# ── 중복 실행 방지 ──
try:
    lock_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    lock_socket.bind(("127.0.0.1", 49152))
except socket.error:
    print("\n[경고] 이미 프로그램이 실행 중입니다. 기존 창을 확인해 주세요.")
    sys.exit(1)

import config
from core.video_manager import VideoManager
from core.camera_manager import (
    load_cameras, save_cameras, add_camera, remove_camera,
    update_camera, get_camera
)
from core.event_logger import log_event, get_recent_events
from alert.alert_manager import AlertManager
from swimmer_module import PROFILES, SwimmerMonitor, GhostTracker, RISK_WARNING, RISK_DANGER

# ── 디렉토리 생성 ────────────────────────────────
os.makedirs(config.UPLOAD_DIR, exist_ok=True)

# ── FastAPI 앱 ───────────────────────────────────
app = FastAPI(title="AI 기반 익사 감지 시스템")
app.mount("/static", StaticFiles(directory=os.path.join(config.BASE_DIR, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(config.BASE_DIR, "templates"))

# ── 글로벌 상태 ──────────────────────────────────
cameras_config: list = load_cameras()

# 카메라별 처리 인스턴스
video_managers: dict = {}
frame_locks: dict = {}
current_frames: dict = {}
processing_flags: dict = {}
fps_displays: dict = {}
# 카메라별 SwimmerMonitor 딕셔너리
cam_monitors: dict = {}
# 카메라별 GhostTracker
cam_ghost_trackers: dict = {}
# 카메라별 ByteTrack 트래커
cam_trackers: dict = {}
# 카메라별 알림 상태
active_alerts: dict = {}       # camera_id → set of alerted track IDs
# 카메라별 ROI (numpy polygon)
cam_roi_pool: dict = {}        # camera_id → np.array or None
cam_roi_exits: dict = {}       # camera_id → list of np.array
# 카메라별 스냅샷 (ROI 설정용)
cam_snapshots: dict = {}

# 공유 알림
global_alert = {"active": False, "camera_name": "", "message": ""}
alert_manager = AlertManager()

# ── YOLO 모델 (공유) ─────────────────────────────
print("🔄 YOLO 모델 로딩 중...")
is_pose_model = "pose" in config.MODEL_PATH.lower()
if is_pose_model:
    shared_model = YOLO(config.MODEL_PATH, task="pose")
else:
    shared_model = YOLO(config.MODEL_PATH)
print("✅ YOLO 모델 로드 완료")


def _init_camera_instances(cam_id: str):
    """카메라 ID에 해당하는 처리 인스턴스 초기화"""
    if cam_id not in video_managers:
        video_managers[cam_id] = VideoManager()
        current_frames[cam_id] = None
        frame_locks[cam_id] = threading.Lock()
        processing_flags[cam_id] = False
        fps_displays[cam_id] = 0.0
        cam_monitors[cam_id] = {}
        cam_ghost_trackers[cam_id] = GhostTracker()
        cam_trackers[cam_id] = sv.ByteTrack(
            track_activation_threshold=0.2,
            lost_track_buffer=30,
            frame_rate=30,
        )
        active_alerts[cam_id] = set()
        cam_roi_pool[cam_id] = None
        cam_roi_exits[cam_id] = []
        cam_snapshots[cam_id] = None

    # ROI 복원 (cameras.json에서)
    cam = get_camera(cameras_config, cam_id)
    if cam:
        if cam.get("pool_polygon"):
            cam_roi_pool[cam_id] = np.array(cam["pool_polygon"], dtype=np.int32)
        if cam.get("exit_polygons"):
            cam_roi_exits[cam_id] = [np.array(ep, dtype=np.int32) for ep in cam["exit_polygons"]]


def _init_all_cameras():
    """저장된 카메라 목록 모두 인스턴스 생성 및 자동 연결"""
    for cam in cameras_config:
        cam_id = cam["id"]
        _init_camera_instances(cam_id)
        if cam.get("source_type") == "rtsp" and cam.get("source_path"):
            if video_managers[cam_id].open_rtsp(cam["source_path"]):
                _start_processing(cam_id)
            else:
                print(f"⚠️ [{cam['name']}] RTSP 자동 연결 실패: {cam['source_path']}")


# ═══════════════════════════════════════════════════
#  영상 처리 루프 (카메라별 독립 쓰레드)
# ═══════════════════════════════════════════════════

def process_loop(cam_id: str, cam_name: str):
    global global_alert
    processing_flags[cam_id] = True
    skip_counter = 0
    fps_counter = 0
    fps_time = time.time()

    vm = video_managers[cam_id]
    monitors = cam_monitors[cam_id]
    ghost_tracker = cam_ghost_trackers[cam_id]
    tracker = cam_trackers[cam_id]
    prev_track_ids = set()

    cam = get_camera(cameras_config, cam_id)
    profile_name = cam.get("profile", "KIDS_POOL") if cam else "KIDS_POOL"
    profile = PROFILES.get(profile_name, PROFILES["KIDS_POOL"])

    while processing_flags[cam_id] and vm.is_opened():
        t_start = time.time()
        frame = vm.read_frame()
        if frame is None:
            time.sleep(0.01)
            continue

        skip_counter += 1
        if skip_counter % (config.FRAME_SKIP + 1) != 0:
            # 스킵 프레임에도 최신 프레임을 유지
            with frame_locks[cam_id]:
                if current_frames[cam_id] is not None:
                    pass  # 이전 annotated 프레임 유지
            continue

        # ── YOLO 추론 ──
        results = shared_model(frame, imgsz=config.INPUT_SIZE, verbose=False,
                               conf=config.CONF_THRESHOLD, device="cpu")
        result = results[0]

        detections = sv.Detections.from_ultralytics(result)

        # Pose 데이터 추출
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

        # person 클래스만 필터
        if hasattr(detections, "class_id") and detections.class_id is not None:
            detections = detections[detections.class_id == 0]

        detections = tracker.update_with_detections(detections)
        current_ids = set()

        # ── 상태 업데이트 & 시각화 ──
        pool_poly = cam_roi_pool.get(cam_id)
        exit_polys = cam_roi_exits.get(cam_id, [])

        for xyxy, mask, confidence, class_id, track_id, data in detections:
            if track_id is None:
                continue

            cx = (xyxy[0] + xyxy[2]) / 2
            cy = (xyxy[1] + xyxy[3]) / 2
            center = (cx, cy)

            # ROI 필터링
            is_in_pool = True
            is_in_exit = False

            if pool_poly is not None and len(pool_poly) > 0:
                is_in_pool = (cv2.pointPolygonTest(pool_poly, center, False) >= 0)

            for ep in exit_polys:
                if len(ep) > 0 and cv2.pointPolygonTest(ep, center, False) >= 0:
                    is_in_exit = True
                    break

            if not is_in_pool and not is_in_exit:
                continue

            current_ids.add(track_id)
            ghost_tracker.mark_alive(track_id)

            if track_id not in monitors:
                monitors[track_id] = SwimmerMonitor(track_id, profile)
                matched_ghost_id, matched_info = ghost_tracker.try_match_new_detection(xyxy, frame.shape)
                if matched_ghost_id is not None and matched_ghost_id in monitors:
                    monitors[track_id].risk_history = monitors[matched_ghost_id].risk_history
                    monitors[track_id].risk_score = monitors[matched_ghost_id].risk_score
                    monitors[track_id].state = monitors[matched_ghost_id].state
                    del monitors[matched_ghost_id]

            kp_xy = data.get("kp_xy", None) if has_pose else None
            kp_conf = data.get("kp_conf", None) if has_pose else None

            if is_in_exit:
                status = "SAFE_EXIT"
                risk_level = 0
                monitors[track_id].risk_score = 0.0
                monitors[track_id].risk_history.clear()
                monitors[track_id].update(xyxy, kp_xy, kp_conf)
                monitors[track_id].state = status
            else:
                status, risk_level = monitors[track_id].update(xyxy, kp_xy, kp_conf)

            # 시각화
            x1, y1, x2, y2 = map(int, xyxy)
            risk_score = monitors[track_id].risk_score
            dbg = monitors[track_id].debug

            if risk_level == 2:
                color = (0, 0, 255)
            elif risk_level == 1:
                color = (0, 220, 255)
            else:
                color = (0, 200, 0)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            bar_w = x2 - x1
            bar_fill = int(bar_w * min(1.0, risk_score))
            cv2.rectangle(frame, (x1, y2 + 2), (x2, y2 + 8), (50, 50, 50), -1)
            if bar_fill > 0:
                bc = (0, 200, 0) if risk_score < RISK_WARNING else (
                    (0, 220, 255) if risk_score < RISK_DANGER else (0, 0, 255))
                cv2.rectangle(frame, (x1, y2 + 2), (x1 + bar_fill, y2 + 8), bc, -1)

            label = f"ID:{track_id} {status}"
            sub = f"R:{risk_score:.2f} Spd:{dbg.get('speed', 0):.1f}"
            cv2.putText(frame, label, (x1 + 2, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            cv2.putText(frame, sub, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

            # 위험 알림
            if risk_level > 0 and (skip_counter % 30 == 0):
                active_alerts[cam_id].add(track_id)
                log_event(cam_name, f"ID:{track_id} {status} (위험도:{risk_score:.2f})")
                global_alert["active"] = True
                global_alert["camera_name"] = cam_name
                global_alert["message"] = f"ID:{track_id} {status}"
                alert_manager.trigger()

        # ── 소실 감지 (GhostTracker) ──
        disappeared = prev_track_ids - current_ids
        for tid in disappeared:
            if tid in monitors:
                m = monitors[tid]
                if m.state == "SAFE_EXIT":
                    continue
                if m.bbox_history:
                    ghost_tracker.mark_disappeared(tid, list(m.bbox_history)[-1], m.state, m.risk_score)
        prev_track_ids = current_ids.copy()

        # 유령 경고 시각화
        for alert_info in ghost_tracker.get_alerts(profile):
            bx1, by1, bx2, by2 = map(int, alert_info["bbox"])
            elapsed = alert_info["elapsed"]

            if alert_info["level"] == 2:
                gc = (0, 0, 255)
            elif alert_info["level"] == 1:
                gc = (0, 180, 255)
            else:
                gc = (200, 200, 200)

            for i in range(bx1, bx2, 10):
                cv2.line(frame, (i, by1), (min(i + 5, bx2), by1), gc, 2)
                cv2.line(frame, (i, by2), (min(i + 5, bx2), by2), gc, 2)
            for i in range(by1, by2, 10):
                cv2.line(frame, (bx1, i), (bx1, min(i + 5, by2)), gc, 2)
                cv2.line(frame, (bx2, i), (bx2, min(i + 5, by2)), gc, 2)

            gtxt = f"LOST ID:{alert_info['track_id']} {elapsed:.1f}s"
            cv2.putText(frame, gtxt, (bx1, by1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, gc, 2)

            if skip_counter % 30 == 0 and alert_info["level"] > 0:
                log_event(cam_name, f"유령 소실 경고 ID:{alert_info['track_id']} ({elapsed:.1f}초)")

        # 상단 정보
        info_txt = f"Detect: {len(current_ids)} | Ghost: {len(ghost_tracker.ghosts)}"
        cv2.putText(frame, info_txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # ROI 외곽선 그리기
        if pool_poly is not None and len(pool_poly) > 0:
            cv2.polylines(frame, [pool_poly], True, (255, 150, 0), 2)
        for ep in exit_polys:
            if len(ep) > 0:
                cv2.polylines(frame, [ep], True, (100, 255, 0), 2)

        # 스냅샷 저장 (ROI 설정용 - 원본 깨끗한 프레임)
        cam_snapshots[cam_id] = vm.read_frame()

        # 결과 프레임 저장
        with frame_locks[cam_id]:
            current_frames[cam_id] = frame

        # FPS 계산
        fps_counter += 1
        elapsed_fps = time.time() - fps_time
        if elapsed_fps >= 1.0:
            fps_displays[cam_id] = fps_counter / elapsed_fps
            fps_counter = 0
            fps_time = time.time()

        elapsed = time.time() - t_start
        wait_time = (1 / vm.fps) - elapsed
        if wait_time > 0:
            time.sleep(wait_time)

    processing_flags[cam_id] = False


def generate_mjpeg(cam_id: str):
    """카메라별 독립 MJPEG 스트림"""
    while True:
        with frame_locks.get(cam_id, threading.Lock()):
            frame = current_frames.get(cam_id)
        if frame is not None:
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
        time.sleep(0.033)


def _stop_camera(cam_id: str):
    """카메라 처리 중단"""
    processing_flags[cam_id] = False
    time.sleep(0.3)
    if cam_id in cam_monitors:
        cam_monitors[cam_id].clear()
    if cam_id in cam_ghost_trackers:
        cam_ghost_trackers[cam_id] = GhostTracker()
    if cam_id in cam_trackers:
        cam_trackers[cam_id] = sv.ByteTrack(
            track_activation_threshold=0.2, lost_track_buffer=30, frame_rate=30)
    if cam_id in active_alerts:
        active_alerts[cam_id].clear()


def _start_processing(cam_id: str):
    """카메라 처리 쓰레드 시작"""
    cam = get_camera(cameras_config, cam_id)
    cam_name = cam["name"] if cam else cam_id
    t = threading.Thread(target=process_loop, args=(cam_id, cam_name), daemon=True)
    t.start()


# ═══════════════════════════════════════════════════
#  API 라우트
# ═══════════════════════════════════════════════════

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


@app.get("/video_feed/{cam_id}")
async def video_feed(cam_id: str):
    return StreamingResponse(
        generate_mjpeg(cam_id),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


# ── 카메라 CRUD ──────────────────────────────────

@app.get("/api/cameras")
async def list_cameras():
    result = []
    for cam in cameras_config:
        cid = cam["id"]
        result.append({
            **cam,
            "fps": round(fps_displays.get(cid, 0), 1),
            "processing": processing_flags.get(cid, False),
            "active_alerts": list(active_alerts.get(cid, set())),
        })
    return result


@app.post("/api/cameras")
async def create_camera(name: str = Body(..., embed=True)):
    cam = add_camera(cameras_config, name)
    _init_camera_instances(cam["id"])
    return cam


@app.delete("/api/cameras/{cam_id}")
async def delete_camera(cam_id: str):
    _stop_camera(cam_id)
    if cam_id in video_managers:
        video_managers[cam_id].release()
        del video_managers[cam_id]
    for d in [cam_monitors, cam_ghost_trackers, cam_trackers,
              current_frames, frame_locks, processing_flags,
              fps_displays, active_alerts, cam_roi_pool, cam_roi_exits, cam_snapshots]:
        d.pop(cam_id, None)
    success = remove_camera(cameras_config, cam_id)
    return {"status": "ok" if success else "not_found"}


@app.put("/api/cameras/{cam_id}/name")
async def rename_camera(cam_id: str, name: str = Body(..., embed=True)):
    ok = update_camera(cameras_config, cam_id, name=name)
    return {"status": "ok" if ok else "not_found"}


# ── 프로필 설정 ──────────────────────────────────

@app.put("/api/cameras/{cam_id}/profile")
async def set_profile(cam_id: str, profile: str = Body(..., embed=True)):
    if profile not in PROFILES:
        return {"status": "error", "message": f"존재하지 않는 프로필: {profile}"}
    ok = update_camera(cameras_config, cam_id, profile=profile)
    return {"status": "ok" if ok else "not_found"}


# ── 소스 설정 ────────────────────────────────────

@app.post("/api/cameras/{cam_id}/source/webcam")
async def set_webcam(cam_id: str):
    _stop_camera(cam_id)
    if video_managers[cam_id].open_webcam(0):
        update_camera(cameras_config, cam_id, source_type="webcam", source_path=None)
        _start_processing(cam_id)
        return {"status": "ok", "message": "웹캠 연결됨"}
    return {"status": "error", "message": "웹캠 연결 실패"}


@app.post("/api/cameras/{cam_id}/source/upload")
async def upload_video(cam_id: str, file: UploadFile = File(...)):
    _stop_camera(cam_id)
    file_path = os.path.join(config.UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)
    if video_managers[cam_id].open_file(file_path):
        update_camera(cameras_config, cam_id, source_type="file", source_path=file_path)
        _start_processing(cam_id)
        return {"status": "ok", "message": f"'{file.filename}' 로드 완료"}
    return {"status": "error", "message": "영상 파일 열기 실패"}


@app.post("/api/cameras/{cam_id}/source/rtsp")
async def set_rtsp(cam_id: str, url: str = Form(...)):
    _stop_camera(cam_id)
    if video_managers[cam_id].open_rtsp(url):
        update_camera(cameras_config, cam_id, source_type="rtsp", source_path=url)
        _start_processing(cam_id)
        return {"status": "ok", "message": "RTSP 연결됨"}
    return {"status": "error", "message": "RTSP 연결 실패"}


# ── ROI 설정 ─────────────────────────────────────

@app.get("/api/cameras/{cam_id}/snapshot")
async def get_snapshot(cam_id: str):
    """ROI 설정을 위한 현재 프레임 스냅샷 (JPEG)"""
    frame = cam_snapshots.get(cam_id)
    if frame is None:
        # 스냅샷이 없으면 현재 프레임 사용
        with frame_locks.get(cam_id, threading.Lock()):
            frame = current_frames.get(cam_id)
    if frame is None:
        return JSONResponse({"status": "error", "message": "스냅샷 없음"}, status_code=404)

    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return StreamingResponse(
        iter([buf.tobytes()]),
        media_type="image/jpeg",
    )


@app.get("/api/cameras/{cam_id}/roi")
async def get_roi(cam_id: str):
    """현재 ROI 설정 조회"""
    cam = get_camera(cameras_config, cam_id)
    if not cam:
        return {"status": "not_found"}
    return {
        "pool_polygon": cam.get("pool_polygon"),
        "exit_polygons": cam.get("exit_polygons", []),
    }


@app.post("/api/cameras/{cam_id}/roi")
async def set_roi(cam_id: str, pool_polygon: list = Body(None), exit_polygons: list = Body([])):
    """ROI 설정 저장"""
    # cameras.json에 저장
    update_camera(cameras_config, cam_id,
                  pool_polygon=pool_polygon,
                  exit_polygons=exit_polygons)

    # 런타임 상태 업데이트
    if pool_polygon and len(pool_polygon) > 0:
        cam_roi_pool[cam_id] = np.array(pool_polygon, dtype=np.int32)
    else:
        cam_roi_pool[cam_id] = None

    cam_roi_exits[cam_id] = [np.array(ep, dtype=np.int32) for ep in exit_polygons if ep]

    return {"status": "ok", "message": "ROI 설정 저장됨"}


@app.delete("/api/cameras/{cam_id}/roi")
async def clear_roi(cam_id: str):
    """ROI 설정 초기화"""
    update_camera(cameras_config, cam_id, pool_polygon=None, exit_polygons=[])
    cam_roi_pool[cam_id] = None
    cam_roi_exits[cam_id] = []
    return {"status": "ok"}


# ── 알람 ─────────────────────────────────────────

@app.post("/api/cameras/{cam_id}/alarm/reset")
async def reset_alarm(cam_id: str):
    alert_manager.reset()
    active_alerts.get(cam_id, set()).clear()
    if not any(active_alerts.values()):
        global_alert["active"] = False
        global_alert["camera_name"] = ""
        global_alert["message"] = ""
    return {"status": "ok"}


@app.post("/api/alarm/reset/all")
async def reset_all_alarms():
    alert_manager.reset()
    for cam_id in list(active_alerts.keys()):
        active_alerts.get(cam_id, set()).clear()
    global_alert["active"] = False
    global_alert["camera_name"] = ""
    global_alert["message"] = ""
    return {"status": "ok"}


# ── 상태 조회 ────────────────────────────────────

@app.get("/api/status")
async def get_status():
    camera_statuses = []
    for cam in cameras_config:
        cid = cam["id"]
        camera_statuses.append({
            "id": cid,
            "name": cam["name"],
            "source_type": cam.get("source_type", "none"),
            "profile": cam.get("profile", "KIDS_POOL"),
            "fps": round(fps_displays.get(cid, 0), 1),
            "processing": processing_flags.get(cid, False),
            "active_alerts": list(active_alerts.get(cid, set())),
            "has_roi": cam_roi_pool.get(cid) is not None,
        })
    return {
        "cameras": camera_statuses,
        "global_alert": global_alert,
        "alert": alert_manager.get_status(),
        "profiles": list(PROFILES.keys()),
    }


@app.get("/api/events")
async def get_events():
    return get_recent_events(20)


# ═══════════════════════════════════════════════════
#  엔트리 포인트
# ═══════════════════════════════════════════════════

if __name__ == "__main__":
    try:
        import pyi_splash
        pyi_splash.close()
    except ImportError:
        pass

    _init_all_cameras()

    def open_browser():
        webbrowser.open(f"http://localhost:{config.PORT}")

    Timer(1.5, open_browser).start()

    print("=" * 50)
    print("  🏊 AI 기반 익사 감지 시스템")
    print(f"  http://localhost:{config.PORT}")
    print("=" * 50)
    uvicorn.run(app, host=config.HOST, port=config.PORT)
