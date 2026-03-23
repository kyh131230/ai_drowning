import cv2
import numpy as np
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel, QPushButton, QHBoxLayout, QMessageBox
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen, QColor, QPolygon
from PyQt5.QtCore import Qt, QPoint

class ROISetupDialog(QDialog):
    def __init__(self, q_img, parent=None):
        super().__init__(parent)
        self.setWindowTitle("안전/감지 구역 설정 (ROI)")
        self.setWindowFlag(Qt.WindowMaximizeButtonHint)
        
        # 원본 이미지
        self.original_img = q_img.copy()
        
        # 설정될 데이터
        self.pool_polygon = []   # 감지 대상 구역 (물)
        self.exit_polygons = []  # 안전 구역 (여러 개 가능, 현재는 1개만 지원 형태로 단순화)
        
        self.current_points = []
        self.current_mode = "POOL" # "POOL" or "EXIT"
        
        self.init_ui()
        self.showMaximized()

    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # 가이드 라벨
        self.guide_label = QLabel(
            "<b>[감지 구역 설정]</b> 영상을 클릭하여 물 영역(감지 구역)의 테두리를 그리세요. "
            "<br>- 왼쪽 클릭: 점 추가 <br>- 우클릭 또는 '구역 완료' 버튼: 그리기 완료"
        )
        layout.addWidget(self.guide_label)
        
        # 캔버스 라벨 (이미지 클릭 이벤트를 위해 서브클래싱 적용)
        self.canvas = ClickableLabel()
        self.canvas.setAlignment(Qt.AlignCenter)
        self.canvas.mousePressSignal.connect(self.on_canvas_click)
        layout.addWidget(self.canvas)
        
        # 하단 컨트롤
        control_layout = QHBoxLayout()
        self.finish_poly_btn = QPushButton("현재 구역 완료 (우클릭)")
        self.finish_poly_btn.clicked.connect(self.finish_current_polygon)
        
        self.reset_btn = QPushButton("초기화 (다시 그리기)")
        self.reset_btn.clicked.connect(self.reset_all)
        
        self.save_btn = QPushButton("모든 설정 저장 및 닫기")
        self.save_btn.setStyleSheet("background-color: #cceeff; font-weight: bold;")
        self.save_btn.clicked.connect(self.accept)
        
        control_layout.addWidget(self.finish_poly_btn)
        control_layout.addWidget(self.reset_btn)
        control_layout.addStretch()
        control_layout.addWidget(self.save_btn)
        
        layout.addLayout(control_layout)
        
        # 첫 렌더링
        self.update_canvas()

    def on_canvas_click(self, x, y):
        """캔버스 내부 좌표(x, y) 클릭 시"""
        # 이미지의 스케일 비율 계산
        pixmap = self.canvas.pixmap()
        if not pixmap: return
        
        # 클릭된 좌표는 라벨 기준이므로 여백(Alignment Center) 보정
        label_w = self.canvas.width()
        label_h = self.canvas.height()
        pix_w = pixmap.width()
        pix_h = pixmap.height()
        
        off_x = (label_w - pix_w) // 2
        off_y = (label_h - pix_h) // 2
        
        if x < off_x or x > off_x + pix_w or y < off_y or y > off_y + pix_h:
            return # 이미지 바깥 클릭
            
        real_x = x - off_x
        real_y = y - off_y
        
        # 원본 해상도로 변환
        scale_x = self.original_img.width() / pix_w
        scale_y = self.original_img.height() / pix_h
        
        orig_x = int(real_x * scale_x)
        orig_y = int(real_y * scale_y)
        
        self.current_points.append((orig_x, orig_y))
        self.update_canvas()

    def finish_current_polygon(self):
        if len(self.current_points) < 3:
            QMessageBox.warning(self, "오류", "최소 3개 이상의 점을 찍어야 영역이 완성됩니다.")
            return
            
        if self.current_mode == "POOL":
            self.pool_polygon = list(self.current_points)
            self.current_points = []
            self.current_mode = "EXIT"
            self.guide_label.setText(
                "<b>[안전 구역 설정]</b> <font color='green'>안전 구역(탈출구, 계단 등)</font>을 그리세요. "
                "완료 시 '구역 완료' 버튼을 누르거나 저장하세요."
            )
        elif self.current_mode == "EXIT":
            self.exit_polygons.append(list(self.current_points))
            self.current_points = []
            self.guide_label.setText("<b>[설정 완료]</b> 더 추가하려면 안전구역을 계속 그리거나 [저장 및 닫기]를 누르세요.")
            
        self.update_canvas()

    def reset_all(self):
        self.pool_polygon = []
        self.exit_polygons = []
        self.current_points = []
        self.current_mode = "POOL"
        self.guide_label.setText("<b>[감지 구역 설정]</b> 영상을 클릭하여 물 영역(감지 구역)의 테두리를 그리세요.")
        self.update_canvas()

    def update_canvas(self):
        # QImage를 복사해서 Painter로 그림
        display_img = self.original_img.copy()
        
        painter = QPainter(display_img)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # 풀(감지) 영역 그리기 (파란색)
        if self.pool_polygon:
            painter.setPen(QPen(QColor(0, 150, 255), 3))
            painter.setBrush(QColor(0, 150, 255, 60))
            poly = QPolygon([QPoint(x, y) for x, y in self.pool_polygon])
            painter.drawPolygon(poly)
            
        # 안전 영역 그리기 (초록색)
        for ep in self.exit_polygons:
            painter.setPen(QPen(QColor(0, 255, 100), 3))
            painter.setBrush(QColor(0, 255, 100, 60))
            poly = QPolygon([QPoint(x, y) for x, y in ep])
            painter.drawPolygon(poly)
            
        # 현재 그리고 있는 선명한 점/선
        if self.current_points:
            color = QColor(0, 150, 255) if self.current_mode == "POOL" else QColor(0, 255, 100)
            painter.setPen(QPen(color, 2))
            painter.setBrush(color)
            for i, pt in enumerate(self.current_points):
                painter.drawEllipse(QPoint(pt[0], pt[1]), 5, 5)
                if i > 0:
                    painter.drawLine(QPoint(self.current_points[i-1][0], self.current_points[i-1][1]), 
                                     QPoint(pt[0], pt[1]))
        painter.end()
        
        # 다이얼로그 창 크기에 맞춰 영상 최대 확대 비율 계산
        max_w = max(800, self.width() - 40)
        max_h = max(600, self.height() - 130)
        
        pixmap = QPixmap.fromImage(display_img)
        scaled = pixmap.scaled(max_w, max_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.canvas.setPixmap(scaled)

    def resizeEvent(self, event):
        self.update_canvas()
        super().resizeEvent(event)

    def mousePressEvent(self, event):
        # 다이얼로그 전역 우클릭 캐치 (구역 그리기 완료)
        if event.button() == Qt.RightButton:
            self.finish_current_polygon()
        super().mousePressEvent(event)

    def get_polygons(self):
        """다이얼로그 종료 시 반환값"""
        pool = np.array(self.pool_polygon, dtype=np.int32) if self.pool_polygon else None
        exits = [np.array(ep, dtype=np.int32) for ep in self.exit_polygons]
        return pool, exits


from PyQt5.QtCore import pyqtSignal

class ClickableLabel(QLabel):
    mousePressSignal = pyqtSignal(int, int)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.mousePressSignal.emit(event.pos().x(), event.pos().y())
        super().mousePressEvent(event)
