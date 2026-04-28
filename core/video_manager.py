"""
영상 소스 관리 모듈
- 웹캠, 영상 파일(무한 반복), RTSP 카메라 지원
- RTSP/웹캠 지연 방지를 위한 쓰레딩 리더 지원
"""
import cv2
import threading
import time
from typing import Optional


class VideoManager:
    """영상 입력 소스를 관리하는 클래스"""

    def __init__(self):
        self.cap: Optional[cv2.VideoCapture] = None
        self.source_type = "none"   # "webcam" | "file" | "rtsp" | "none"
        self.source_path = None
        self.loop = False
        self.fps = 30

        # ── 쓰레딩 리더 관련 ──
        self._frame = None
        self._ret = False
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    # ── 소스 열기 ────────────────────────────────

    def open_webcam(self, index: int = 0) -> bool:
        self._release()
        self.cap = cv2.VideoCapture(index)
        if self.cap.isOpened():
            self.source_type = "webcam"
            self.source_path = index
            self.loop = False
            self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
            self._start_reader()
            return True
        return False

    def open_file(self, path: str) -> bool:
        self._release()
        self.cap = cv2.VideoCapture(path)
        if self.cap.isOpened():
            self.source_type = "file"
            self.source_path = path
            self.loop = True          # 무한 반복 재생
            self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
            self._start_reader()
            return True
        return False

    def open_rtsp(self, url: str) -> bool:
        self._release()
        self.cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        if self.cap.isOpened():
            self.source_type = "rtsp"
            self.source_path = url
            self.loop = False
            self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25
            self._start_reader()
            return True
        return False

    # ── 쓰레딩 리더 ──────────────────────────────

    def _start_reader(self):
        self._running = True
        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()

    def _reader_loop(self):
        """백그라운드에서 버퍼를 계속 비워 항상 최신 프레임을 유지 (RTSP 지연 방지 핵심)"""
        while self._running and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()

            if not ret:
                if self.loop and self.source_type == "file":
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    with self._lock:
                        self._ret = False
                    break

            with self._lock:
                self._frame = frame
                self._ret = True

            # 파일의 경우 너무 빨리 읽지 않도록 제한
            if self.source_type == "file":
                time.sleep(1 / self.fps)

    # ── 프레임 읽기 ──────────────────────────────

    def read_frame(self):
        """최신 프레임 1장 반환 (버퍼 지연 없음)"""
        with self._lock:
            if not self._ret:
                return None
            return self._frame.copy() if self._frame is not None else None

    # ── 유틸리티 ─────────────────────────────────

    def is_opened(self) -> bool:
        return self.cap is not None and self.cap.isOpened()

    def get_info(self) -> dict:
        if self.cap is None or not self.cap.isOpened():
            return {"source_type": "none", "source_path": None, "fps": 0, "width": 0, "height": 0}
        return {
            "source_type": self.source_type,
            "source_path": str(self.source_path),
            "fps": round(self.fps, 1),
            "width": int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        }

    def _release(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=0.1)
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        with self._lock:
            self._frame = None
            self._ret = False
        self.source_type = "none"

    def release(self):
        self._release()
