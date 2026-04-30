"""
AI 익사 감지 시스템 - 설정 파일
"""
import os
import sys

# ── 모델 설정 ──────────────────────────────────────
MODEL_PATH = "yolo26m_openvino_model_1280/"
CONF_THRESHOLD = 0.40
PERSON_CLASS_ID = 0
INPUT_SIZE = 640

# ── 영상 설정 ──────────────────────────────────────
DEFAULT_SOURCE = 0
FRAME_SKIP = 3              # N프레임마다 1회 추론

# ── 경광등 / 알림 설정 ─────────────────────────────
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
