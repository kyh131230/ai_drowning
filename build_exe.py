import PyInstaller.__main__
import os
import shutil

# 1. 빌드 전 정리
print("🧹 이전 빌드 기록 정리 중...")
for folder in ['build', 'dist']:
    if os.path.exists(folder):
        shutil.rmtree(folder)

# 2. 빌드 설정
# --onedir: 폴더 형태로 생성 (OpenVINO 모델 로딩 및 설정 파일 유지 관리에 유리)
# --name: 최종 실행 파일 이름
# --add-data: 필수 리소스 포함
params = [
    'launcher.py',              # 진입점 (스플래시 런처)
    '--onedir',                 # 폴더 모드 (권장)
    '--name=AI_Drowning_System',
    '--clean',
    
    # [데이터 포함]
    '--add-data=templates;templates',
    '--add-data=static;static',
    '--add-data=alert;alert',
    '--add-data=core;core',
    '--add-data=yolo26m_openvino_model_1280;yolo26m_openvino_model_1280',
    '--add-data=cameras.json;.',
    
    # [의존성 강제 포함]
    '--collect-all=ultralytics',
    '--collect-all=openvino',
    '--collect-all=supervision',
    '--collect-submodules=uvicorn',
    '--collect-submodules=serial',
    
    # [기타 설정]
    '--console',                # 오류 확인을 위해 터미널 창 표시
]

print("🚀 빌드 시작 (OpenVINO/Ultralytics 포함으로 인해 수 분이 소요될 수 있습니다)...")
PyInstaller.__main__.run(params)

# 3. 빌드 후 추가 파일 복사
print("📂 설정 파일 복사 중...")
dist_path = os.path.join('dist', 'AI_Drowning_System')
for f in ['alert_settings.json', 'profile_settings.json']:
    if os.path.exists(f):
        shutil.copy(f, dist_path)

# 필수 폴더 생성
for d in ['log', 'uploads']:
    os.makedirs(os.path.join(dist_path, d), exist_ok=True)

print("\n✨ 빌드가 완료되었습니다!")
print(f"👉 위치: {os.path.abspath(dist_path)}")
print("   'AI_Drowning_System.exe'를 실행하세요.")
