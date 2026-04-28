"""
경광등 + PC 알람 사운드 관리 모듈
- ALERT_MOCK_MODE = True  → 경광등 없이 콘솔 로그 + 사운드만
- ALERT_MOCK_MODE = False → USB 경광등(pyserial) + 사운드
"""
import threading
import time
import platform

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class AlertManager:
    """경광등 및 PC 알람 통합 관리"""

    def __init__(self):
        self.mock_mode = config.ALERT_MOCK_MODE
        self.serial_conn = None
        self.is_active = False
        self.last_trigger_time = None
        self._lock = threading.Lock()

        if not self.mock_mode:
            try:
                import serial
                self.serial_conn = serial.Serial(
                    config.ALERT_COM_PORT,
                    config.ALERT_BAUDRATE,
                    timeout=1,
                )
                print(f"✅ 경광등 연결 완료: {config.ALERT_COM_PORT}")
            except Exception as e:
                print(f"⚠️ 경광등 연결 실패: {e} → 모의 모드로 전환")
                self.mock_mode = True

    def trigger(self):
        """경보 발동"""
        with self._lock:
            if self.is_active:
                return
            self.is_active = True
            self.last_trigger_time = time.time()

        print("🚨 [경보] 위험 감지! 알람 작동!")

        if not self.mock_mode and self.serial_conn:
            try:
                self.serial_conn.write(bytes([0xA0, 0x01, 0x01, 0xA2]))
            except Exception as e:
                print(f"경광등 제어 오류: {e}")

        if config.ALERT_SOUND_ENABLED:
            threading.Thread(target=self._play_alarm, daemon=True).start()

    def reset(self):
        """경보 해제"""
        with self._lock:
            self.is_active = False

        print("✅ [경보 해제]")

        if not self.mock_mode and self.serial_conn:
            try:
                self.serial_conn.write(bytes([0xA0, 0x01, 0x00, 0xA1]))
            except Exception as e:
                print(f"경광등 제어 오류: {e}")

    def _play_alarm(self):
        """Windows 비프음 반복"""
        if platform.system() == "Windows":
            try:
                import winsound
                for _ in range(5):
                    if not self.is_active:
                        break
                    winsound.Beep(1000, 800)
                    time.sleep(0.2)
            except Exception:
                pass

    def get_status(self) -> dict:
        return {
            "is_active": self.is_active,
            "mock_mode": self.mock_mode,
            "last_trigger": self.last_trigger_time,
        }

    def close(self):
        self.reset()
        if self.serial_conn:
            self.serial_conn.close()
