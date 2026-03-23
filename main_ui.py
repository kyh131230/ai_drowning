import sys
import math
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QGridLayout, QPushButton, QLineEdit, QLabel, QComboBox, QListWidget, QInputDialog, QFrame, QMessageBox
)
from PyQt5.QtCore import Qt, pyqtSlot
from PyQt5.QtGui import QPixmap, QImage
from camera_worker import CameraWorker
from roi_dialog import ROISetupDialog

class CameraWidget(QFrame):
    def __init__(self, cam_id, name, worker, parent=None):
        super().__init__(parent)
        self.cam_id = cam_id
        self.name = name
        self.worker = worker
        self.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        self.setLineWidth(1)
        
        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(5, 5, 5, 5)
        
        # 상단 컨트롤 바
        self.control_layout = QHBoxLayout()
        self.name_label = QLabel(self.name)
        self.name_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        
        self.roi_btn = QPushButton("영역 설정")
        self.roi_btn.setStyleSheet("background-color: #e0e0e0;")
        self.rename_btn = QPushButton("이름 변경")
        self.delete_btn = QPushButton("카메라 삭제")
        self.delete_btn.setStyleSheet("background-color: #ffcccc;")
        
        self.control_layout.addWidget(self.name_label)
        self.control_layout.addStretch()
        self.control_layout.addWidget(self.roi_btn)
        self.control_layout.addWidget(self.rename_btn)
        self.control_layout.addWidget(self.delete_btn)
        
        # 비디오 표시 라벨
        self.video_label = QLabel("영상 로딩 중 또는 스트림 대기...")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: black; color: white;")
        self.video_label.setMinimumSize(320, 240)
        
        self.layout.addLayout(self.control_layout)
        self.layout.addWidget(self.video_label, 1)
        
        self.setLayout(self.layout)
        
        self.last_qimage = None

    @pyqtSlot(int, QImage, QImage)
    def update_image(self, cam_id, drawn_qimg, clean_qimg):
        self.last_qimage = clean_qimg.copy()
        pixmap = QPixmap.fromImage(drawn_qimg)
        # 라벨 크기에 맞게 화면 스케일링 (비율 유지)
        scaled_pixmap = pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.video_label.setPixmap(scaled_pixmap)


class DrowningMonitorUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI 익사 감지 시스템 v4.0 (다중 카메라 지원 플랫폼)")
        self.setGeometry(100, 100, 1280, 768)
        
        self.cameras = {}   # cam_id -> CameraWidget 매핑
        self.next_cam_id = 0
        
        self.init_ui()
        
    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # 좌측: 제어 패널 (카메라 추가 및 알림 로그)
        left_panel = QVBoxLayout()
        left_panel.setContentsMargins(10, 10, 10, 10)
        
        control_group = QFrame()
        control_group.setFrameStyle(QFrame.StyledPanel | QFrame.Sunken)
        cg_layout = QVBoxLayout(control_group)
        
        cg_layout.addWidget(QLabel("<b>[ 새 카메라 추가 ]</b>"))
        
        self.src_input = QLineEdit()
        self.src_input.setPlaceholderText("소스 (숫자:웹캠, URL:IP카메라, 경로:영상)")
        self.src_input.setText("./source/water_slide_test_3.mp4") # 테스트용 기본 소스
        cg_layout.addWidget(self.src_input)
        
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("카메라 이름 (예: 1번 유아풀)")
        cg_layout.addWidget(self.name_input)
        
        self.profile_combo = QComboBox()
        self.profile_combo.addItems(["KIDS_POOL", "LANE_POOL"]) # 프로필 메뉴
        cg_layout.addWidget(self.profile_combo)
        
        self.add_btn = QPushButton("카메라 연결 시작")
        self.add_btn.setStyleSheet("background-color: #cceeff; font-weight: bold; padding: 10px;")
        self.add_btn.clicked.connect(self.add_camera)
        cg_layout.addWidget(self.add_btn)
        
        left_panel.addWidget(control_group)
        
        # 실시간 알림 패널
        left_panel.addWidget(QLabel("<b>[ 실시간 알림 로그 ]</b>"))
        self.log_list = QListWidget()
        left_panel.addWidget(self.log_list)
        
        # 우측: 비디오 그리드 영역
        self.grid_widget = QWidget()
        self.grid_layout = QGridLayout(self.grid_widget)
        
        # 화면 분할 비율 설정 (좌측 제어부: 1, 우측 영상부: 4)
        main_layout.addLayout(left_panel, 1)
        main_layout.addWidget(self.grid_widget, 4)
        
    def add_camera(self):
        src = self.src_input.text().strip()
        name = self.name_input.text().strip()
        profile = self.profile_combo.currentText()
        
        if not src:
            QMessageBox.warning(self, "오류", "소스 경로를 입력해주세요.")
            return
            
        if not name:
            name = f"Camera {self.next_cam_id + 1} ({src[:10]})"
            
        cam_id = self.next_cam_id
        self.next_cam_id += 1
        
        # 스레드 워커 생성 및 시작
        worker = CameraWorker(cam_id, src, profile)
        widget = CameraWidget(cam_id, name, worker)
        
        # 시그널 연결
        worker.change_pixmap_signal.connect(widget.update_image)
        worker.alert_signal.connect(self.handle_alert)
        
        # 버튼 이벤트
        widget.roi_btn.clicked.connect(lambda: self.set_camera_roi(cam_id))
        widget.rename_btn.clicked.connect(lambda: self.rename_camera(cam_id))
        widget.delete_btn.clicked.connect(lambda: self.delete_camera(cam_id))
        
        self.cameras[cam_id] = widget
        self.update_grid_layout()
        
        worker.start()
        self.log_list.insertItem(0, f"[시스템] {name} 연결 시작 ({profile})")
        
    def rename_camera(self, cam_id):
        if cam_id not in self.cameras: return
        widget = self.cameras[cam_id]
        new_name, ok = QInputDialog.getText(self, "이름 변경", "새 카메라 이름:", text=widget.name)
        if ok and new_name:
            widget.name = new_name
            widget.name_label.setText(new_name)
            self.log_list.insertItem(0, f"[시스템] 카메라 이름 변경: {new_name}")
            
    def delete_camera(self, cam_id):
        if cam_id not in self.cameras: return
        
        # 삭제 확인 대화상자
        reply = QMessageBox.question(self, '삭제 확인', '정말 카메라를 삭제하시겠습니까?', 
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.No: return
        
        widget = self.cameras.pop(cam_id)
        
        # 스레드 안전 종료 및 메모리 해제
        widget.worker.stop()
        
        self.grid_layout.removeWidget(widget)
        widget.deleteLater()
        
        self.update_grid_layout()
        self.log_list.insertItem(0, f"[시스템] {widget.name} 작동이 중지되었습니다.")
        
    def set_camera_roi(self, cam_id):
        if cam_id not in self.cameras: return
        widget = self.cameras[cam_id]
        
        if widget.last_qimage is None:
            QMessageBox.warning(self, "대기", "아직 영상 프레임이 수신되지 않았습니다. 잠시 후 다시 시도해주세요.")
            return
            
        # 스레드 일시 정지 (UI 그리기만 중지, 모델 추론 생략)
        widget.worker.is_paused = True
        
        # 다이얼로그 호출
        dlg = ROISetupDialog(widget.last_qimage, self)
        if dlg.exec_():
            pool, exits = dlg.get_polygons()
            # 저장된 폴리곤 좌표를 워커에 넘김
            if pool is not None:
                widget.worker.pool_polygon = pool
            widget.worker.exit_polygons = exits
            self.log_list.insertItem(0, f"[시스템] {widget.name} 감지 구역(ROI) 설정 적용됨.")
            
        # 작업 재개
        widget.worker.is_paused = False
    
    def update_grid_layout(self):
        n = len(self.cameras)
        if n == 0: return
        
        # 카메라 수에 따른 스마트 열/행 계산
        cols = math.ceil(math.sqrt(n))
        
        idx = 0
        for cam_id, widget in self.cameras.items():
            self.grid_layout.removeWidget(widget)
            row = idx // cols
            col = idx % cols
            self.grid_layout.addWidget(widget, row, col)
            idx += 1
            
    @pyqtSlot(int, str, int)
    def handle_alert(self, cam_id, msg, level):
        if cam_id not in self.cameras: return
        name = self.cameras[cam_id].name
        
        log_msg = f"[{name}] {msg}"
        self.log_list.insertItem(0, log_msg)
        
        # 알림 레벨별 색상 (리스트 아이템) 2: 빨강, 1: 노랑
        item = self.log_list.item(0)
        if level == 2:
            item.setBackground(Qt.red)
            item.setForeground(Qt.white)
        elif level == 1:
            item.setBackground(Qt.yellow)
            item.setForeground(Qt.black)

    def closeEvent(self, event):
        """앱 종료 시 활성된 모든 카메라 스레드를 안전하게 정리"""
        self.log_list.insertItem(0, "[시스템] 모든 스트림을 종료하고 앱을 닫습니다...")
        QApplication.processEvents()
        
        for widget in self.cameras.values():
            widget.worker.stop()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
    window = DrowningMonitorUI()
    window.show()
    sys.exit(app.exec_())
