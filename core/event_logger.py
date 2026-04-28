"""
이벤트 로그 모듈
- 위험 감지 이벤트를 log/events.csv에 영구 저장
- 서버 재시작 시에도 기존 기록 유지 (누적 저장)
"""
import csv
import os
import threading
from datetime import datetime

LOG_DIR = os.path.join(os.getcwd(), "log")
LOG_FILE = os.path.join(LOG_DIR, "events.csv")
HEADERS = ["연번", "날짜", "시간", "카메라이름", "이벤트종류"]

_lock = threading.Lock()


def _ensure_file():
    """log 폴더와 CSV 파일이 없으면 생성, 헤더 포함"""
    os.makedirs(LOG_DIR, exist_ok=True)
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(HEADERS)


def _next_seq() -> int:
    """현재 저장된 이벤트 수 + 1 반환"""
    try:
        with open(LOG_FILE, "r", encoding="utf-8-sig") as f:
            return sum(1 for row in csv.reader(f))
    except Exception:
        return 1


def log_event(camera_name: str, event_type: str):
    """이벤트를 events.csv에 추가 기록."""
    with _lock:
        try:
            _ensure_file()
            seq = _next_seq()
            now = datetime.now()
            row = [
                seq - 1,
                now.strftime("%Y-%m-%d"),
                now.strftime("%H:%M:%S"),
                camera_name,
                event_type,
            ]
            with open(LOG_FILE, "a", newline="", encoding="utf-8-sig") as f:
                writer = csv.writer(f)
                writer.writerow(row)
        except Exception as e:
            print(f"[로그 저장 오류] {e}")


def get_recent_events(n: int = 20) -> list:
    """최근 n건의 이벤트를 dict 리스트로 반환"""
    try:
        _ensure_file()
        with open(LOG_FILE, "r", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        return rows[-n:]
    except Exception:
        return []
