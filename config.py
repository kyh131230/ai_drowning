"""
AI 익사 감지 시스템 - 설정 파일
"""
import os
import sys
import json

# ── 모델 설정 ──────────────────────────────────────
MODEL_PATH = "yolo26m_openvino_model_1280/"
CONF_THRESHOLD = 0.30
PERSON_CLASS_ID = 0
INPUT_SIZE = 640

# ── 영상 설정 ──────────────────────────────────────
DEFAULT_SOURCE = 0
FRAME_SKIP = 0              # 0 = 모든 프레임 처리 (스킵 없음)

# ── 경광등 / 알림 설정 (기본값) ───────────────────
ALERT_MOCK_MODE = True      # True = 테스트(경광등 없이), False = 실제 경광등
ALERT_COM_PORT = "COM3"
ALERT_BAUDRATE = 9600
ALERT_SOUND_ENABLED = True

# ── 서버 설정 ──────────────────────────────────────
HOST = "0.0.0.0"
PORT = 8000

# ── 경로 설정 ──────────────────────────────────────
if getattr(sys, 'frozen', False):
    BASE_DIR = sys._MEIPASS
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_DIR = os.path.join(os.getcwd(), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ── 경광등 설정 영구 저장/로드 ───────────────────
_ALERT_SETTINGS_FILE = os.path.join(os.getcwd(), "alert_settings.json")

def load_alert_settings():
    """저장된 경광등 설정을 불러와 전역 변수에 적용합니다."""
    global ALERT_COM_PORT, ALERT_MOCK_MODE, ALERT_SOUND_ENABLED
    if os.path.exists(_ALERT_SETTINGS_FILE):
        try:
            with open(_ALERT_SETTINGS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            ALERT_COM_PORT = data.get("com_port", ALERT_COM_PORT)
            ALERT_MOCK_MODE = data.get("mock_mode", ALERT_MOCK_MODE)
            ALERT_SOUND_ENABLED = data.get("sound_enabled", ALERT_SOUND_ENABLED)
        except Exception as e:
            print(f"[설정 로드 오류] {e}")

def save_alert_settings():
    """현재 경광등 설정을 파일에 저장합니다."""
    try:
        with open(_ALERT_SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump({
                "com_port": ALERT_COM_PORT,
                "mock_mode": ALERT_MOCK_MODE,
                "sound_enabled": ALERT_SOUND_ENABLED,
            }, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[설정 저장 오류] {e}")

# 시작 시 저장된 설정 자동 로드
load_alert_settings()
