"""
AI 익사 감지 시스템 - 런처
EXE 실행 시 스플래시 화면을 보여주고, 서버 준비 완료 후 브라우저를 자동으로 엽니다.
"""
import os
import sys
import threading
import time
import webbrowser
import urllib.request
import tkinter as tk
from tkinter import ttk

# ── 경로 설정 ─────────────────────────────────────────────
# EXE로 실행 중이면 작업 디렉토리를 EXE가 있는 폴더로 고정
if getattr(sys, 'frozen', False):
    EXE_DIR = os.path.dirname(sys.executable)
    os.chdir(EXE_DIR)
    # PyInstaller가 압축 해제한 내부 폴더도 sys.path에 추가
    sys.path.insert(0, sys._MEIPASS)
else:
    EXE_DIR = os.path.dirname(os.path.abspath(__file__))

SERVER_URL = "http://localhost:8000"
APP_TITLE  = "AI 익사 감지 시스템"

# ────────────────────────────────────────────────────────
# 스플래시 창
# ────────────────────────────────────────────────────────
class SplashScreen:
    BG      = "#0a0e17"
    ACCENT  = "#06b6d4"
    TEXT1   = "#f1f5f9"
    TEXT2   = "#94a3b8"
    MUTED   = "#475569"
    BORDER  = "#1e3a5f"

    def __init__(self):
        self.root = tk.Tk()
        self.root.title(APP_TITLE)
        self.root.overrideredirect(True)       # 타이틀바 제거
        self.root.attributes("-topmost", True)  # 항상 최상위

        W, H = 480, 300
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        self.root.geometry(f"{W}x{H}+{(sw-W)//2}+{(sh-H)//2}")
        self.root.configure(bg=self.BG)

        # ── 상단 테두리 강조선
        tk.Frame(self.root, bg=self.ACCENT, height=3).pack(fill="x")

        # ── 로고 영역
        tk.Label(
            self.root, text="🏊",
            bg=self.BG, fg=self.ACCENT,
            font=("Segoe UI Emoji", 36)
        ).pack(pady=(28, 0))

        tk.Label(
            self.root, text=APP_TITLE,
            bg=self.BG, fg=self.TEXT1,
            font=("Malgun Gothic", 17, "bold")
        ).pack(pady=(6, 0))

        tk.Label(
            self.root, text="AI Drowning Detection System  v1.0",
            bg=self.BG, fg=self.TEXT2,
            font=("Malgun Gothic", 8)
        ).pack(pady=(2, 0))

        # ── 구분선
        tk.Frame(self.root, bg=self.BORDER, height=1).pack(fill="x", padx=40, pady=18)

        # ── 상태 텍스트
        self.status_var = tk.StringVar(value="초기화 중...")
        tk.Label(
            self.root, textvariable=self.status_var,
            bg=self.BG, fg=self.TEXT2,
            font=("Malgun Gothic", 9)
        ).pack()

        # ── 프로그레스바 (indeterminate)
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure(
            "Splash.Horizontal.TProgressbar",
            troughcolor=self.BG,
            background=self.ACCENT,
            darkcolor=self.ACCENT,
            lightcolor=self.ACCENT,
            bordercolor=self.BG,
            thickness=5,
        )
        self.progress = ttk.Progressbar(
            self.root,
            style="Splash.Horizontal.TProgressbar",
            mode="indeterminate",
            length=380,
        )
        self.progress.pack(pady=14)
        self.progress.start(10)

        # ── 하단 안내
        tk.Label(
            self.root, text="잠시만 기다려 주세요...",
            bg=self.BG, fg=self.MUTED,
            font=("Malgun Gothic", 8)
        ).pack()

    def update_status(self, text: str):
        self.status_var.set(text)
        self.root.update_idletasks()

    def close(self):
        try:
            self.progress.stop()
            self.root.destroy()
        except Exception:
            pass


# ────────────────────────────────────────────────────────
# 실행 중 제어 창
# ────────────────────────────────────────────────────────
class RunningWindow:
    BG     = "#0a0e17"
    ACCENT = "#06b6d4"
    TEXT1  = "#f1f5f9"
    GREEN  = "#22c55e"

    def __init__(self):
        self.root = tk.Tk()
        self.root.title(APP_TITLE)
        self.root.geometry("340x160")
        self.root.resizable(False, False)
        self.root.configure(bg=self.BG)
        self.root.protocol("WM_DELETE_WINDOW", self._on_quit)
        # 최상위 아님 (일반 창)
        self.root.attributes("-topmost", False)

        # ── 상단 강조선
        tk.Frame(self.root, bg=self.ACCENT, height=3).pack(fill="x")

        tk.Label(
            self.root, text=f"🏊  {APP_TITLE}",
            bg=self.BG, fg=self.ACCENT,
            font=("Malgun Gothic", 12, "bold")
        ).pack(pady=(16, 2))

        tk.Label(
            self.root, text="● 서버 실행 중  |  localhost:8000",
            bg=self.BG, fg=self.GREEN,
            font=("Malgun Gothic", 9)
        ).pack()

        # ── 버튼 행
        btn_frame = tk.Frame(self.root, bg=self.BG)
        btn_frame.pack(pady=18)

        tk.Button(
            btn_frame, text="🌐 브라우저 열기",
            command=lambda: webbrowser.open(SERVER_URL),
            bg=self.ACCENT, fg="#fff",
            font=("Malgun Gothic", 9, "bold"),
            relief="flat", padx=14, pady=7, cursor="hand2",
            activebackground="#0891b2", activeforeground="#fff",
        ).pack(side="left", padx=8)

        tk.Button(
            btn_frame, text="  종료  ",
            command=self._on_quit,
            bg="#ef4444", fg="#fff",
            font=("Malgun Gothic", 9, "bold"),
            relief="flat", padx=14, pady=7, cursor="hand2",
            activebackground="#b91c1c", activeforeground="#fff",
        ).pack(side="left", padx=8)

    def _on_quit(self):
        os._exit(0)

    def run(self):
        self.root.mainloop()


# ────────────────────────────────────────────────────────
# 서버 시작
# ────────────────────────────────────────────────────────
def _start_server():
    try:
        import uvicorn
        import main as app_main
        uvicorn.run(
            app_main.app,
            host="0.0.0.0",
            port=8000,
            log_level="warning",
        )
    except Exception as e:
        print(f"[서버 오류] {e}")


def _wait_for_server(splash: SplashScreen, timeout: int = 120) -> bool:
    """서버가 응답할 때까지 최대 timeout초 대기"""
    for i in range(timeout * 5):
        try:
            urllib.request.urlopen(f"{SERVER_URL}/api/cameras", timeout=1)
            return True
        except Exception:
            pass
        time.sleep(0.2)
        elapsed = int(i * 0.2)
        splash.update_status(f"AI 모델 로딩 중... ({elapsed}초 경과)")
    return False


# ────────────────────────────────────────────────────────
# 메인 진입점
# ────────────────────────────────────────────────────────
def main():
    # 1. 스플래시 창 표시
    splash = SplashScreen()

    # 2. 서버 대기 및 상태 업데이트용 스레드 정의
    def _bg_wait():
        start_t = time.time()
        # 서버 스레드 시작
        server_thread = threading.Thread(target=_start_server, daemon=True)
        server_thread.start()

        # 서버가 응답할 때까지 루프 (최대 120초)
        ok = False
        for i in range(600): # 0.2s * 600 = 120s
            elapsed = int(time.time() - start_t)
            splash.update_status(f"AI 모델 로딩 중... ({elapsed}초 경과)")
            
            try:
                # 서버가 켜졌는지 확인
                urllib.request.urlopen(f"{SERVER_URL}/api/cameras", timeout=0.5)
                ok = True
                break
            except Exception:
                pass
            time.sleep(0.2)

        if ok:
            splash.update_status("✅ 준비 완료! 브라우저를 여는 중...")
            splash.root.after(700, lambda: _finish(splash))
        else:
            splash.update_status("❌ 서버 시작 실패. 콘솔 로그를 확인하세요.")
            splash.root.after(4000, lambda: os._exit(1))

    def _finish(splash: SplashScreen):
        splash.close()
        print("✅ 서버가 준비되었습니다. 브라우저를 엽니다.")
        webbrowser.open(SERVER_URL)
        # 이제 별도의 창(RunningWindow)을 띄우지 않고, 
        # 메인 스레드는 터미널이 닫힐 때까지 대기합니다.

    # 3. 상태 업데이트 스레드 시작
    threading.Thread(target=_bg_wait, daemon=True).start()

    # 4. 스플래시 이벤트 루프 실행
    splash.root.mainloop()

    # 스플래시 창이 닫힌 후, 서버 스레드가 살아있도록 메인 스레드를 유지합니다.
    while True:
        time.sleep(1)


if __name__ == "__main__":
    main()
