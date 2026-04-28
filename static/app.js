// ── 전역 상태 ──────────────────────────────────
let cameras = [];

// ── ROI 관련 상태 ──────────────────────────────
let roiCamId = null;
let roiImage = null;
let roiMode = "POOL"; // "POOL" | "EXIT"
let roiCurrentPoints = [];
let roiPoolPolygon = [];
let roiExitPolygons = [];
let roiScale = 1;
let roiOffsetX = 0;
let roiOffsetY = 0;

// ── 초기화 ──────────────────────────────────────
document.addEventListener("DOMContentLoaded", async () => {
    await loadCameras();
    startPolling();
    setupROICanvas();
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

            // 프로필 복원
            const profileSelect = card.querySelector(".profile-select");
            if (cam.profile) {
                profileSelect.value = cam.profile;
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
            infoText.textContent = cam.processing
                ? `${cam.source_type.toUpperCase()} | ${cam.fps} FPS${roiStatus}`
                : "영상 없음";
        });
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
    roiPoolPolygon = [];
    roiExitPolygons = [];

    // 기존 ROI 로드
    try {
        const roiRes = await fetch(`/api/cameras/${camId}/roi`);
        const roiData = await roiRes.json();
        if (roiData.pool_polygon) roiPoolPolygon = roiData.pool_polygon;
        if (roiData.exit_polygons) roiExitPolygons = roiData.exit_polygons;
        if (roiPoolPolygon.length > 0) {
            roiMode = "EXIT";
        }
    } catch (e) { /* 무시 */ }

    // 스냅샷 가져오기
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => {
        roiImage = img;
        const modal = document.getElementById("roi-modal");
        modal.classList.remove("hidden");
        updateROIGuide();

        // 캔버스 크기 조정
        requestAnimationFrame(() => {
            const wrapper = document.querySelector(".roi-canvas-wrapper");
            const canvas = document.getElementById("roi-canvas");
            const maxW = wrapper.clientWidth - 40;
            const maxH = Math.min(window.innerHeight * 0.6, wrapper.clientHeight);

            const scaleW = maxW / img.width;
            const scaleH = maxH / img.height;
            roiScale = Math.min(scaleW, scaleH, 1);

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

    // 배경 이미지
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (roiImage) {
        ctx.drawImage(roiImage, roiOffsetX, roiOffsetY,
            roiImage.width * roiScale, roiImage.height * roiScale);
    }

    // 감지(풀) 영역 (파란색)
    if (roiPoolPolygon.length > 0) {
        drawPolygon(ctx, roiPoolPolygon, "rgba(0, 150, 255, 0.3)", "rgba(0, 150, 255, 0.9)", 2);
    }

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

        // 점 그리기
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
        roiPoolPolygon = [...roiCurrentPoints];
        roiCurrentPoints = [];
        roiMode = "EXIT";
    } else {
        roiExitPolygons.push([...roiCurrentPoints]);
        roiCurrentPoints = [];
    }

    updateROIGuide();
    drawROI();
}

function updateROIGuide() {
    const guide = document.getElementById("roi-guide");
    if (roiMode === "POOL") {
        guide.innerHTML = '<strong>[감지 구역 설정]</strong> 영상을 클릭하여 물 영역(감지 구역)의 테두리를 그리세요.<br>- 왼쪽 클릭: 점 추가 &nbsp; - 우클릭: 그리기 완료';
    } else if (roiExitPolygons.length === 0) {
        guide.innerHTML = '<strong>[안전 구역 설정]</strong> <span style="color:#22c55e">안전 구역(탈출구, 계단 등)</span>을 그리세요. 완료 시 "구역 완료" 버튼을 누르거나 우클릭하세요.';
    } else {
        guide.innerHTML = '<strong>[설정 완료]</strong> 더 추가하려면 안전구역을 계속 그리거나 "저장 및 닫기"를 누르세요.';
    }
}

function resetROI() {
    roiPoolPolygon = [];
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
        pool_polygon: roiPoolPolygon.length > 0 ? roiPoolPolygon : null,
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
