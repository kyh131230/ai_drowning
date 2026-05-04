// ── 전역 상태 ──────────────────────────────────
let cameras = [];

// ── ROI 관련 상태 ──────────────────────────────
let roiCamId = null;
let roiImage = null;
let roiMode = "POOL"; // "POOL" | "EXIT"
let roiCurrentPoints = [];
let roiPoolPolygons = [];   // 복수 파란 구역 배열
let roiExitPolygons = [];
let roiScale = 1;
let roiOffsetX = 0;
let roiOffsetY = 0;

// ── 초기화 ──────────────────────────────────────
document.addEventListener("DOMContentLoaded", async () => {
    await loadCameras();
    startPolling();
    setupROICanvas();
    await loadAlertStatus();  // 경광등 상태 초기 로드
});

// ── 카메라 목록 로드 및 그리드 렌더링 ─────────
async function loadCameras() {
    const res = await fetch("/api/cameras");
    cameras = await res.json();
    renderGrid(cameras);
    await loadRecentEvents();
}

function renderGrid(cams) {
    const grid = document.getElementById("camera-grid");
    const template = document.getElementById("camera-card-template");

    const existingIds = new Set([...grid.querySelectorAll(".camera-card")].map(c => c.dataset.camId));
    const newIds = new Set(cams.map(c => c.id));

    // 삭제된 카메라 제거
    grid.querySelectorAll(".camera-card").forEach(card => {
        if (!newIds.has(card.dataset.camId)) card.remove();
    });

    // 새 카메라 추가
    cams.forEach(cam => {
        if (!existingIds.has(cam.id)) {
            const clone = template.content.cloneNode(true);
            const card = clone.querySelector(".camera-card");
            card.dataset.camId = cam.id;
            card.querySelector(".cam-name-input").value = cam.name;
            card.querySelector(".video-feed").src = `/video_feed/${cam.id}`;

            // RTSP 주소 자동 복원
            if (cam.source_path && cam.source_type === "rtsp") {
                card.querySelector(".rtsp-input").value = cam.source_path;
            }

            grid.appendChild(clone);
        }
    });
}

// ── 폴링 (상태 갱신) ────────────────────────────
function startPolling() {
    setInterval(async () => {
        const res = await fetch("/api/status");
        const data = await res.json();
        updateStatus(data);
    }, 1500);

    setInterval(loadRecentEvents, 8000);
}

function updateStatus(data) {
    // 전역 알림 배너
    const banner = document.getElementById("global-alert-banner");
    const bannerText = document.getElementById("alert-banner-text");
    if (data.global_alert && data.global_alert.active) {
        banner.classList.remove("hidden");
        bannerText.textContent = `⚠️ ${data.global_alert.camera_name}에서 위험이 감지되었습니다! ${data.global_alert.message || ""}`;
    } else {
        banner.classList.add("hidden");
    }

    // 각 카메라 카드 상태 갱신
    if (data.cameras) {
        data.cameras.forEach(cam => {
            const card = document.querySelector(`.camera-card[data-cam-id="${cam.id}"]`);
            if (!card) return;

            const dot = card.querySelector(".cam-status-dot");
            const infoText = card.querySelector(".cam-info-text");
            const dangerOverlay = card.querySelector(".danger-overlay");
            const alarmBtn = card.querySelector(".alarm-reset-btn");

            const hasAlert = cam.active_alerts && cam.active_alerts.length > 0;
            dot.className = "cam-status-dot" + (hasAlert ? " alert" : cam.processing ? " active" : "");
            card.classList.toggle("alert", hasAlert);
            dangerOverlay.classList.toggle("hidden", !hasAlert);
            alarmBtn.classList.toggle("hidden", !hasAlert);

            const roiStatus = cam.has_roi ? " | ROI ✅" : "";
            const pauseStatus = cam.paused ? " | ⏸️ 정지" : "";
            infoText.textContent = cam.processing
                ? `${cam.source_type.toUpperCase()} | ${cam.fps} FPS${roiStatus}${pauseStatus}`
                : "영상 없음";

            // 토글 버튼 상태 동기화
            const toggleBtn = card.querySelector(".analysis-toggle-btn");
            if (toggleBtn) updateToggleBtn(toggleBtn, cam.paused);
        });
    }

    // 입력 크기 드롭다운 동기화
    if (data.input_size) {
        const sizeSelect = document.getElementById("input-size-select");
        if (sizeSelect && sizeSelect.value !== String(data.input_size)) {
            sizeSelect.value = String(data.input_size);
        }
    }
}

// ── 최근 이벤트 로드 ────────────────────────────
async function loadRecentEvents() {
    const res = await fetch("/api/events");
    const events = await res.json();
    const list = document.getElementById("event-log-list");
    list.innerHTML = "";
    events.slice(-5).reverse().forEach(ev => {
        const li = document.createElement("li");
        li.innerHTML = `<strong>${ev["카메라이름"] || ""}</strong> ${ev["이벤트종류"] || ""} <br><span>${ev["날짜"]} ${ev["시간"]}</span>`;
        list.appendChild(li);
    });
}

// ── 카메라 추가 ─────────────────────────────────
async function addCamera() {
    const input = document.getElementById("new-camera-name");
    const name = input.value.trim() || "새 카메라";
    await fetch("/api/cameras", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name }),
    });
    input.value = "";
    await loadCameras();
}

// ── 카메라 삭제 ─────────────────────────────────
async function deleteCamera(btn) {
    const card = btn.closest(".camera-card");
    const camId = card.dataset.camId;
    if (!confirm("이 카메라를 삭제하시겠습니까?")) return;
    await fetch(`/api/cameras/${camId}`, { method: "DELETE" });
    card.remove();
}

// ── 이름 변경 ───────────────────────────────────
async function renameCamera(btn) {
    const card = btn.closest(".camera-card");
    const camId = card.dataset.camId;
    const input = card.querySelector(".cam-name-input");
    input.focus();
    input.addEventListener("blur", async () => {
        await fetch(`/api/cameras/${camId}/name`, {
            method: "PUT",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ name: input.value }),
        });
    }, { once: true });
}

// ── 프로필 변경 ─────────────────────────────────
async function changeProfile(select) {
    const camId = select.closest(".camera-card").dataset.camId;
    const profile = select.value;
    await fetch(`/api/cameras/${camId}/profile`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ profile }),
    });
}

// ── 소스 연결 ───────────────────────────────────
async function connectWebcam(btn) {
    const camId = btn.closest(".camera-card").dataset.camId;
    const res = await fetch(`/api/cameras/${camId}/source/webcam`, { method: "POST" });
    const data = await res.json();
    alert(data.message);
}

async function uploadVideo(input) {
    const camId = input.closest(".camera-card").dataset.camId;
    const file = input.files[0];
    if (!file) return;
    const formData = new FormData();
    formData.append("file", file);
    const res = await fetch(`/api/cameras/${camId}/source/upload`, { method: "POST", body: formData });
    const data = await res.json();
    alert(data.message);
}

async function connectRTSP(btn) {
    const card = btn.closest(".camera-card");
    const camId = card.dataset.camId;
    const url = card.querySelector(".rtsp-input").value.trim();
    if (!url) { alert("RTSP 주소를 입력하세요."); return; }
    const formData = new FormData();
    formData.append("url", url);
    const res = await fetch(`/api/cameras/${camId}/source/rtsp`, { method: "POST", body: formData });
    const data = await res.json();
    alert(data.message);
}

// ── 알람 해제 ───────────────────────────────────
async function resetAlarm(btn) {
    const camId = btn.closest(".camera-card").dataset.camId;
    await fetch(`/api/cameras/${camId}/alarm/reset`, { method: "POST" });
}

async function resetAllAlarms() {
    await fetch("/api/alarm/reset/all", { method: "POST" });
}

// ── 설정 변경 ───────────────────────────────────
async function changeInputSize(select) {
    const value = parseInt(select.value);
    const res = await fetch("/api/settings/input_size", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ value }),
    });
    const data = await res.json();
    if (data.status === "ok") {
        console.log(`입력 크기 변경: ${data.input_size}`);
    }
}

// ── 분석 일시정지 토글 ──────────────────────────
async function toggleAnalysis(btn) {
    const camId = btn.closest(".camera-card").dataset.camId;
    const res = await fetch(`/api/cameras/${camId}/analysis/toggle`, { method: "POST" });
    const data = await res.json();
    if (data.status === "ok") {
        updateToggleBtn(btn, data.paused);
    }
}

function updateToggleBtn(btn, paused) {
    if (paused) {
        btn.textContent = "▶️ 분석 재개";
        btn.classList.remove("btn-secondary");
        btn.classList.add("btn-primary");
    } else {
        btn.textContent = "⏸️ 분석 정지";
        btn.classList.remove("btn-primary");
        btn.classList.add("btn-secondary");
    }
}


// ═══════════════════════════════════════════════════
//  ROI 설정 모달
// ═══════════════════════════════════════════════════

function setupROICanvas() {
    const canvas = document.getElementById("roi-canvas");

    canvas.addEventListener("click", (e) => {
        if (!roiImage) return;
        const rect = canvas.getBoundingClientRect();
        const clickX = e.offsetX;
        const clickY = e.offsetY;

        // 캔버스 좌표 → 원본 이미지 좌표
        const origX = Math.round((clickX - roiOffsetX) / roiScale);
        const origY = Math.round((clickY - roiOffsetY) / roiScale);

        if (origX < 0 || origX > roiImage.width || origY < 0 || origY > roiImage.height) return;

        roiCurrentPoints.push([origX, origY]);
        drawROI();
    });

    canvas.addEventListener("contextmenu", (e) => {
        e.preventDefault();
        finishCurrentPolygon();
    });
}

async function openROISetup(btn) {
    const camId = btn.closest(".camera-card").dataset.camId;
    roiCamId = camId;
    roiMode = "POOL";
    roiCurrentPoints = [];
    roiPoolPolygons = [];
    roiExitPolygons = [];

    // 기존 ROI 로드
    try {
        const roiRes = await fetch(`/api/cameras/${camId}/roi`);
        const roiData = await roiRes.json();
        if (roiData.pool_polygons && roiData.pool_polygons.length > 0) {
            roiPoolPolygons = roiData.pool_polygons;
            roiMode = "EXIT";  // 이미 설정된 경우 EXIT 모드로
        }
        if (roiData.exit_polygons) roiExitPolygons = roiData.exit_polygons;
    } catch (e) { /* 무시 */ }

    // 스냅샷 가져오기
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => {
        roiImage = img;
        const modal = document.getElementById("roi-modal");
        modal.classList.remove("hidden");
        updateROIGuide();

        // 캔버스 크기 조정 (풀스크린 모달이므로 래퍼 실제 크기 사용)
        requestAnimationFrame(() => {
            const wrapper = document.querySelector(".roi-canvas-wrapper");
            const canvas = document.getElementById("roi-canvas");
            const maxW = wrapper.clientWidth - 16;
            const maxH = wrapper.clientHeight - 8;

            const scaleW = maxW / img.width;
            const scaleH = maxH / img.height;
            roiScale = Math.min(scaleW, scaleH); // 업스케일 허용

            canvas.width = Math.round(img.width * roiScale);
            canvas.height = Math.round(img.height * roiScale);
            roiOffsetX = 0;
            roiOffsetY = 0;

            drawROI();
        });
    };
    img.onerror = () => {
        alert("스냅샷을 가져올 수 없습니다. 영상 소스를 먼저 연결하세요.");
    };
    img.src = `/api/cameras/${camId}/snapshot?t=${Date.now()}`;
}

function drawROI() {
    const canvas = document.getElementById("roi-canvas");
    const ctx = canvas.getContext("2d");

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (roiImage) {
        ctx.drawImage(roiImage, roiOffsetX, roiOffsetY,
            roiImage.width * roiScale, roiImage.height * roiScale);
    }

    // 감지(풀) 영역들 (파란색 – 복수 가능)
    roiPoolPolygons.forEach((poly, idx) => {
        drawPolygon(ctx, poly, "rgba(0, 150, 255, 0.25)", "rgba(0, 150, 255, 0.9)", 2);
        // 구역 번호 표시
        if (poly.length > 0) {
            const sx = poly[0][0] * roiScale + roiOffsetX;
            const sy = poly[0][1] * roiScale + roiOffsetY;
            ctx.fillStyle = "rgba(0,150,255,0.9)";
            ctx.font = "bold 14px Inter";
            ctx.fillText(`풀${idx + 1}`, sx + 4, sy + 16);
        }
    });

    // 안전 영역들 (초록색)
    roiExitPolygons.forEach(ep => {
        drawPolygon(ctx, ep, "rgba(0, 255, 100, 0.3)", "rgba(0, 255, 100, 0.9)", 2);
    });

    // 현재 그리고 있는 점/선
    if (roiCurrentPoints.length > 0) {
        const color = roiMode === "POOL" ? "rgba(0, 150, 255, 0.9)" : "rgba(0, 255, 100, 0.9)";
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.beginPath();
        roiCurrentPoints.forEach((pt, i) => {
            const sx = pt[0] * roiScale + roiOffsetX;
            const sy = pt[1] * roiScale + roiOffsetY;
            if (i === 0) ctx.moveTo(sx, sy);
            else ctx.lineTo(sx, sy);
        });
        ctx.stroke();

        ctx.fillStyle = color;
        roiCurrentPoints.forEach(pt => {
            const sx = pt[0] * roiScale + roiOffsetX;
            const sy = pt[1] * roiScale + roiOffsetY;
            ctx.beginPath();
            ctx.arc(sx, sy, 5, 0, Math.PI * 2);
            ctx.fill();
        });
    }
}

function drawPolygon(ctx, points, fillColor, strokeColor, lineWidth) {
    if (points.length < 3) return;
    ctx.beginPath();
    points.forEach((pt, i) => {
        const sx = pt[0] * roiScale + roiOffsetX;
        const sy = pt[1] * roiScale + roiOffsetY;
        if (i === 0) ctx.moveTo(sx, sy);
        else ctx.lineTo(sx, sy);
    });
    ctx.closePath();
    ctx.fillStyle = fillColor;
    ctx.fill();
    ctx.strokeStyle = strokeColor;
    ctx.lineWidth = lineWidth;
    ctx.stroke();
}

function finishCurrentPolygon() {
    if (roiCurrentPoints.length < 3) {
        alert("최소 3개 이상의 점을 찍어야 영역이 완성됩니다.");
        return;
    }

    if (roiMode === "POOL") {
        // 파란 구역은 여러 개 추가 가능 → 리스트에 저장 후 POOL 모드 유지
        roiPoolPolygons.push([...roiCurrentPoints]);
        roiCurrentPoints = [];
    } else {
        roiExitPolygons.push([...roiCurrentPoints]);
        roiCurrentPoints = [];
    }

    updateROIGuide();
    drawROI();
}

function switchToExitMode() {
    if (roiCurrentPoints.length >= 3) {
        // 그리던 중인 구역이 있으면 자동 완료
        roiPoolPolygons.push([...roiCurrentPoints]);
        roiCurrentPoints = [];
    }
    roiMode = "EXIT";
    updateROIGuide();
    drawROI();
}

function updateROIGuide() {
    const guide = document.getElementById("roi-guide");
    const switchBtn = document.getElementById("switch-to-exit-btn");

    if (roiMode === "POOL") {
        const count = roiPoolPolygons.length;
        const countTxt = count > 0 ? ` (${count}개 완료)` : "";
        guide.innerHTML = `<strong>[감지 구역 설정${countTxt}]</strong> 클릭하여 풀(수영장 물 영역)의 테두리를 그리세요.<br>• 좌클릭: 점 추가 &nbsp; • 우클릭: 구역 완료 &nbsp; • 구역이 여러 개면 반복 추가 가능`;
        if (switchBtn) switchBtn.style.display = "";
    } else {
        const poolCnt = roiPoolPolygons.length;
        const exitCnt = roiExitPolygons.length;
        guide.innerHTML = `<strong>[안전 구역 설정]</strong> <span style="color:#22c55e">안전 구역(탈출구, 계단 등)</span>을 그리세요.<br>• 풀 구역 ${poolCnt}개 설정됨 • 안전 구역 ${exitCnt}개`;
        if (switchBtn) switchBtn.style.display = "none";
    }
}

function resetROI() {
    roiPoolPolygons = [];
    roiExitPolygons = [];
    roiCurrentPoints = [];
    roiMode = "POOL";
    updateROIGuide();
    drawROI();
}

async function saveROI() {
    if (!roiCamId) return;

    // 현재 그리고 있는 점이 남아있으면 완료 처리
    if (roiCurrentPoints.length >= 3) {
        finishCurrentPolygon();
    }

    const body = {
        pool_polygons: roiPoolPolygons,
        exit_polygons: roiExitPolygons,
    };

    await fetch(`/api/cameras/${roiCamId}/roi`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
    });

    closeROIModal();
}

function closeROIModal() {
    document.getElementById("roi-modal").classList.add("hidden");
    roiCamId = null;
    roiImage = null;
    roiCurrentPoints = [];
}

// ═══════════════════════════════════════════════════
//  풀스크린 뷰어 (더블클릭으로 카메라 확대)
// ═══════════════════════════════════════════════════

let fullscreenCamId = null;
let fullscreenPollTimer = null;

function openFullscreenViewer(videoWrapper) {
    const card = videoWrapper.closest(".camera-card");
    const camId = card.dataset.camId;
    const camName = card.querySelector(".cam-name-input").value || "카메라";
    const srcFeed = card.querySelector(".video-feed").src;
    const hasAlert = card.classList.contains("alert");

    fullscreenCamId = camId;

    const modal = document.getElementById("fullscreen-modal");
    const feedImg = document.getElementById("fullscreen-video-feed");
    const overlay = document.getElementById("fullscreen-danger-overlay");
    const nameEl = document.getElementById("fullscreen-cam-name");

    nameEl.textContent = `📹 ${camName}`;
    feedImg.src = srcFeed;
    overlay.classList.toggle("hidden", !hasAlert);

    modal.classList.remove("hidden");

    fullscreenPollTimer = setInterval(() => {
        const liveCard = document.querySelector(`.camera-card[data-cam-id="${camId}"]`);
        if (liveCard) {
            overlay.classList.toggle("hidden", !liveCard.classList.contains("alert"));
        }
    }, 1500);
}

function closeFullscreenViewer() {
    document.getElementById("fullscreen-modal").classList.add("hidden");
    document.getElementById("fullscreen-video-feed").src = "";
    clearInterval(fullscreenPollTimer);
    fullscreenPollTimer = null;
    fullscreenCamId = null;
}

document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
        const modal = document.getElementById("fullscreen-modal");
        if (!modal.classList.contains("hidden")) {
            closeFullscreenViewer();
        }
    }
});

// ═══════════════════════════════════════════════════
//  경광등 COM 포트 설정
// ═══════════════════════════════════════════════════

async function loadAlertStatus() {
    try {
        const res = await fetch("/api/status");
        const data = await res.json();
        if (data.alert) updateAlertPortUI(data.alert);
    } catch (e) { /* 무시 */ }
}

function updateAlertPortUI(alertStatus) {
    const badge = document.getElementById("alert-port-status");
    const label = document.getElementById("alert-port-label");
    if (!badge || !label) return;

    const port = alertStatus.com_port || "--";
    label.textContent = port;

    badge.className = "alert-port-badge";
    if (alertStatus.mock_mode) {
        badge.classList.add("mock");
        badge.textContent = "모의모드";
    } else if (alertStatus.connected) {
        badge.classList.add("connected");
        badge.textContent = "연결됨";
    } else {
        badge.classList.add("disconnected");
        badge.textContent = "미연결";
    }

    // 수동 입력칸에 현재 포트 예시 표시
    const manualInput = document.getElementById("manual-port-input");
    if (manualInput && port !== "--") manualInput.placeholder = port;
}

async function scanAlertPorts() {
    const btn = document.getElementById("scan-ports-btn");
    btn.textContent = "⏳ 탐색 중...";
    btn.disabled = true;

    try {
        const res = await fetch("/api/alert/ports");
        const data = await res.json();
        const ports = data.ports || [];

        const row = document.getElementById("port-select-row");
        const sel = document.getElementById("port-select-dropdown");
        sel.innerHTML = '<option value="">― 포트 선택 ―</option>';

        if (ports.length === 0) {
            alert("연결된 시리얼 포트를 찾지 못했습니다.");
        } else {
            ports.forEach(p => {
                const opt = document.createElement("option");
                opt.value = p.port;
                opt.textContent = p.is_ch340
                    ? `★ ${p.port}  [CH340 감지!]`
                    : `${p.port}  ${p.description}`;
                sel.appendChild(opt);
            });

            // CH340이 있으면 자동 선택
            const ch340 = ports.find(p => p.is_ch340);
            if (ch340) {
                sel.value = ch340.port;
                document.getElementById("manual-port-input").value = ch340.port;
            }

            row.style.display = "";

            // 드롭다운 변경 시 수동 입력칸도 동기화
            sel.onchange = () => {
                if (sel.value) document.getElementById("manual-port-input").value = sel.value;
            };
        }
    } catch (e) {
        alert(`포트 탐색 오류: ${e.message}`);
    } finally {
        btn.textContent = "🔍 CH340 자동 탐색";
        btn.disabled = false;
    }
}

async function applyAlertPort(mockMode) {
    const manualInput = document.getElementById("manual-port-input").value.trim();
    const dropdownVal = document.getElementById("port-select-dropdown")?.value || "";
    const port = manualInput || dropdownVal;

    if (!port && !mockMode) {
        alert("포트를 선택하거나 직접 입력해 주세요.");
        return;
    }

    const res = await fetch("/api/alert/port", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ com_port: port || "MOCK", mock_mode: mockMode }),
    });
    const data = await res.json();

    if (data.status === "ok") {
        const msg = mockMode
            ? "화면 모의 모드로 설정되었습니다."
            : `${port} 연결 성공!`;
        alert(msg);
    } else {
        alert(`연결 실패: ${port}\n\uc9c1접 입력한 포트를 확인해 주세요.`);
    }
    await loadAlertStatus();
}

async function testAlertLight() {
    const btn = document.getElementById("alert-test-btn");
    btn.disabled = true;
    btn.classList.add("testing");
    btn.textContent = "🔦 테스트 중... (3초)";

    try {
        await fetch("/api/alert/test", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ duration: 3.0 }),
        });
    } catch (e) {
        alert(`테스트 실패: ${e.message}`);
    }

    // 3초 후 자동 복원
    setTimeout(() => {
        btn.disabled = false;
        btn.classList.remove("testing");
        btn.textContent = "🔦 경광등 3초 테스트";
    }, 3200);
}
