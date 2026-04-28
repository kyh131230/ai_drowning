"""
카메라 설정 관리 모듈
- cameras.json에 카메라 목록, 이름, RTSP 주소, ROI 데이터를 영구 저장
- 서버 재시작 시 자동 복원
"""
import json
import os
import uuid
from typing import List, Optional

CAMERAS_FILE = os.path.join(os.getcwd(), "cameras.json")


def _default_camera(name: str = "새 카메라") -> dict:
    return {
        "id": str(uuid.uuid4())[:8],
        "name": name,
        "source_type": "none",
        "source_path": None,
        "profile": "KIDS_POOL",
        "pool_polygon": None,
        "exit_polygons": [],
    }


def load_cameras() -> List[dict]:
    """cameras.json 로드. 없으면 빈 리스트 반환."""
    if os.path.exists(CAMERAS_FILE):
        try:
            with open(CAMERAS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
        except Exception:
            pass
    return []


def save_cameras(cameras: List[dict]):
    """cameras.json에 저장."""
    try:
        with open(CAMERAS_FILE, "w", encoding="utf-8") as f:
            json.dump(cameras, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[설정 저장 오류] {e}")


def get_camera(cameras: List[dict], cam_id: str) -> Optional[dict]:
    return next((c for c in cameras if c["id"] == cam_id), None)


def add_camera(cameras: List[dict], name: str = "새 카메라") -> dict:
    cam = _default_camera(name)
    cameras.append(cam)
    save_cameras(cameras)
    return cam


def remove_camera(cameras: List[dict], cam_id: str) -> bool:
    original = len(cameras)
    cameras[:] = [c for c in cameras if c["id"] != cam_id]
    if len(cameras) < original:
        save_cameras(cameras)
        return True
    return False


def update_camera(cameras: List[dict], cam_id: str, **kwargs) -> bool:
    cam = get_camera(cameras, cam_id)
    if not cam:
        return False
    cam.update(kwargs)
    save_cameras(cameras)
    return True
