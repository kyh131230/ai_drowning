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


def find_ch340_port() -> str | None:
    """CH340 칩셋 기반 USB 장치의 COM 포트를 자동으로 탐색합니다."""
    try:
        import serial.tools.list_ports
        for port in serial.tools.list_ports.comports():
            desc = (port.description or "").lower()
            hwid = (port.hwid or "").lower()
            # CH340 / CH341 칩셋 식별
            if "ch340" in desc or "ch341" in desc or "1a86" in hwid:
                return port.device
    except Exception as e:
        print(f"[포트 탐색 오류] {e}")
    return None


def list_serial_ports() -> list[dict]:
    """현재 시스템에 연결된 모든 시리얼 포트 목록을 반환합니다."""
    try:
        import serial.tools.list_ports
        ports = []
        for port in serial.tools.list_ports.comports():
            desc = (port.description or "").lower()
            hwid = (port.hwid or "").lower()
            is_ch340 = "ch340" in desc or "ch341" in desc or "1a86" in hwid
            ports.append({
                "port": port.device,
                "description": port.description or port.device,
                "is_ch340": is_ch340,
            })
        return ports
    except Exception as e:
        print(f"[포트 목록 오류] {e}")
        return []


class AlertManager:
    """경광등 및 PC 알람 통합 관리"""

    def __init__(self):
        self.mock_mode = config.ALERT_MOCK_MODE
        self.serial_conn = None
        self.is_active = False
        self.last_trigger_time = None
        self._lock = threading.Lock()

        if not self.mock_mode:
            self._connect(config.ALERT_COM_PORT)

    def _connect(self, port: str) -> bool:
        """지정된 포트로 시리얼 연결을 시도합니다."""
        try:
            import serial
            if self.serial_conn and self.serial_conn.is_open:
                self.serial_conn.close()
            self.serial_conn = serial.Serial(port, config.ALERT_BAUDRATE, timeout=1)
            print(f"✅ 경광등 연결 완료: {port}")
            return True
        except Exception as e:
            print(f"⚠️ 경광등 연결 실패 ({port}): {e} → 모의 모드로 전환")
            self.serial_conn = None
            return False

    def reconnect(self, port: str) -> bool:
        """새 포트로 재연결합니다. (UI에서 포트 변경 시 호출)"""
        self.mock_mode = False
        success = self._connect(port)
        if not success:
            self.mock_mode = True
        return success

    def test_light(self, duration: float = 3.0):
        """경광등을 duration초 동안 켰다 끕니다. (테스트용)"""
        def _do_test():
            print(f"🔦 [테스트] 경광등 {duration}초 켜기")
            if not self.mock_mode and self.serial_conn:
                try:
                    self.serial_conn.write(bytes([0xA0, 0x01, 0x01, 0xA2]))
                except Exception as e:
                    print(f"경광등 제어 오류: {e}")
            time.sleep(duration)
            print("🔦 [테스트] 경광등 끄기")
            if not self.mock_mode and self.serial_conn:
                try:
                    self.serial_conn.write(bytes([0xA0, 0x01, 0x00, 0xA1]))
                except Exception as e:
                    print(f"경광등 제어 오류: {e}")

        threading.Thread(target=_do_test, daemon=True).start()

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
            "com_port": config.ALERT_COM_PORT,
            "connected": self.serial_conn is not None and (
                self.serial_conn.is_open if self.serial_conn else False
            ),
        }

    def close(self):
        self.reset()
        if self.serial_conn:
            self.serial_conn.close()
