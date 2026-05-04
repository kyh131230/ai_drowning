# -*- mode: python ; coding: utf-8 -*-
# ─────────────────────────────────────────────────────────────
# AI 익사 감지 시스템 - PyInstaller 빌드 스펙
# 실행: pyinstaller drowning.spec
# ─────────────────────────────────────────────────────────────

block_cipher = None

# ── 포함할 데이터 파일/폴더 ──────────────────────────────────
added_datas = [
    ('templates',                    'templates'),
    ('static',                       'static'),
    ('alert',                        'alert'),
    ('core',                         'core'),
    ('yolo26m_openvino_model_1280',  'yolo26m_openvino_model_1280'),
    ('config.py',                    '.'),
    ('main.py',                      '.'),
    ('swimmer_module.py',            '.'),
]

# ── 숨겨진 임포트 (PyInstaller가 자동 감지 못하는 것들) ───────
hidden = [
    # FastAPI / Uvicorn
    'uvicorn',
    'uvicorn.lifespan.on',
    'uvicorn.loops',
    'uvicorn.loops.asyncio',
    'uvicorn.loops.auto',
    'uvicorn.protocols',
    'uvicorn.protocols.http.auto',
    'uvicorn.protocols.http.h11_impl',
    'uvicorn.protocols.websockets.auto',
    'uvicorn.protocols.websockets.websockets_impl',
    'fastapi',
    'starlette',
    'starlette.middleware',
    'starlette.routing',
    'starlette.staticfiles',
    'starlette.templating',
    'jinja2',
    'python_multipart',
    'multipart',
    # AI / 영상
    'ultralytics',
    'ultralytics.nn.tasks',
    'ultralytics.models',
    'openvino',
    'openvino.runtime',
    'supervision',
    'supervision.tracker.byte_tracker',
    'cv2',
    'numpy',
    # 시리얼 (경광등)
    'serial',
    'serial.tools',
    'serial.tools.list_ports',
    'serial.tools.list_ports_windows',
    # 기타
    'aiofiles',
    'h11',
    'anyio',
    'asyncio',
    'tkinter',
    'tkinter.ttk',
]

a = Analysis(
    ['launcher.py'],          # 진입점
    pathex=['.'],
    binaries=[],
    datas=added_datas,
    hiddenimports=hidden,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'matplotlib',
        'IPython',
        'jupyter',
        'notebook',
        'pandas',
        'scipy',
        'PIL',         # Pillow (미사용 시)
        'pytest',
        'setuptools',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(
    a.pure,
    a.zipped_data,
    cipher=block_cipher,
)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,       # onedir 방식
    name='AI익사감지시스템',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,                   # UPX 압축 끄기 (OpenVINO 호환성)
    console=True,               # 콘솔창 숨김
    # icon='static/favicon.ico', # 아이콘 파일이 있을 경우 주석 해제
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='AI익사감지시스템',
)
