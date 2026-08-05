/*
  dashboard.js - Interactive Highway Accident Detection Command Dashboard Logic
*/

let selectedFile = null;
let selectedSamplePath = null;
let map = null;
let simInterval = null;
let simulatedVehicles = [];
let crashMarkers = [];
let lastReportData = null;

// Initialize Dashboard
document.addEventListener("DOMContentLoaded", () => {
  initClock();
  initMap();
});

// Live Clock HUD
function initClock() {
  const clockEl = document.getElementById("live-clock");
  setInterval(() => {
    const now = new Date();
    clockEl.innerText = now.toISOString().replace("T", " ").substring(0, 19) + " UTC";
  }, 1000);
}

// Role-Based Switcher
function changePersonnelRole(roleClass) {
  document.body.className = roleClass;
  const roleName = document.getElementById("role-select").options[document.getElementById("role-select").selectedIndex].text;
  addNotification(`Logged in as: ${roleName}`, "info");
}

// File Selection & Drag-and-Drop
function handleFileSelect(event) {
  const files = event.target.files;
  if (files.length > 0) {
    selectedFile = files[0];
    selectedSamplePath = null;
    document.getElementById("selected-filename").innerText = selectedFile.name + " (" + (selectedFile.size / (1024*1024)).toFixed(1) + " MB)";
    document.getElementById("selected-file-info").style.display = "flex";
  }
}

function selectSampleVideo(sampleUrl, sampleName) {
  selectedFile = null;
  selectedSamplePath = sampleUrl;
  document.getElementById("selected-filename").innerText = sampleName + " (Workspace Sample)";
  document.getElementById("selected-file-info").style.display = "flex";
  
  // Set video source preview
  const videoPlayer = document.getElementById("main-video-player");
  const videoSource = document.getElementById("video-source");
  videoSource.src = sampleUrl;
  videoPlayer.load();
}

// Video Processing API Call
async function startVideoProcessing() {
  if (!selectedFile && !selectedSamplePath) {
    alert("Please select a video file or pick a sample video to run accident detection.");
    return;
  }

  const btn = document.getElementById("btn-process");
  const hud = document.getElementById("processing-hud");
  const progressBar = document.getElementById("progress-bar-fill");
  const statusText = document.getElementById("processing-status-text");
  
  btn.disabled = true;
  hud.style.display = "block";
  progressBar.style.width = "20%";
  statusText.innerText = "⏳ Initializing V7 Dual-Engine Pipeline (YOLO11s)...";

  const formData = new FormData();
  if (selectedFile) {
    formData.append("video_file", selectedFile);
  } else if (selectedSamplePath) {
    formData.append("sample_path", selectedSamplePath);
  }

  try {
    progressBar.style.width = "45%";
    statusText.innerText = "🔍 Processing video frames: Pipeline A (Accident) & Pipeline B (Tracking)...";

    const response = await fetch("/api/process_video", {
      method: "POST",
      body: selectedFile ? formData : JSON.stringify({ sample_path: selectedSamplePath }),
      headers: selectedFile ? {} : { "Content-Type": "application/json" }
    });

    progressBar.style.width = "85%";
    statusText.innerText = "⚡ Transcoding telemetry video clip & generating dispatch deliverables...";

    const data = await response.json();
    progressBar.style.width = "100%";

    if (data.status === "success") {
      lastReportData = data.deliverables;
      displayDeliverables(data.deliverables);
      addNotification(`Accident processing completed for ${selectedFile ? selectedFile.name : selectedSamplePath}`, data.deliverables.accident_detected ? "alert" : "info");
    } else {
      alert("Error: " + data.message);
    }
  } catch (err) {
    console.error("Processing failed:", err);
    alert("Failed to process video. Check server logs.");
  } finally {
    btn.disabled = false;
    setTimeout(() => { hud.style.display = "none"; }, 1000);
  }
}

// Display All 10 Deliverables
function displayDeliverables(d) {
  // 1. Accident Flag & Confidence
  const flagEl = document.getElementById("out-accident-flag");
  const confEl = document.getElementById("out-confidence");
  if (d.accident_detected) {
    flagEl.innerText = "🚨 ACCIDENT CONFIRMED";
    flagEl.className = "flag-pill flag-active";
    document.getElementById("detection-indicator").innerText = "🚨 CRASH DETECTED";
    document.getElementById("detection-indicator").className = "feed-tag live-tag";
  } else {
    flagEl.innerText = "✅ NO ACCIDENT";
    flagEl.className = "flag-pill flag-inactive";
    document.getElementById("detection-indicator").innerText = "NORMAL FLOW";
    document.getElementById("detection-indicator").className = "feed-tag status-tag";
  }
  confEl.innerText = d.confidence_score;

  // 2. Map Location & Camera
  document.getElementById("out-location").innerText = `Lat ${d.map_location.latitude}, Lon ${d.map_location.longitude}`;
  document.getElementById("out-camera-id").innerText = `Highway: ${d.map_location.highway} | ID: ${d.map_location.camera_id}`;

  // 3. Evidence Buttons
  const btnSnap = document.getElementById("btn-view-snapshot");
  const btnTel = document.getElementById("btn-view-telemetry");
  if (d.media_urls.snapshot_image) {
    btnSnap.disabled = false;
    document.getElementById("snapshot-img-preview").src = d.media_urls.snapshot_image;
  }
  btnTel.disabled = false;

  // 4. Collision Type
  document.getElementById("out-collision-type").innerText = d.collision_type;

  // 5 & 6. Vehicle Count & Classes
  document.getElementById("out-vehicle-count").innerText = `${d.vehicle_count} Vehicles`;
  const classContainer = document.getElementById("out-vehicle-classes");
  classContainer.innerHTML = "";
  d.vehicle_classes.forEach(cls => {
    const badge = document.createElement("span");
    badge.className = "class-badge";
    badge.innerText = cls.toUpperCase();
    classContainer.appendChild(badge);
  });

  // 7. Severity Score
  const sevEl = document.getElementById("out-severity");
  sevEl.innerText = d.impact_severity.toUpperCase();
  sevEl.className = "severity-badge " + (
    d.impact_severity.includes("Critical") ? "severity-critical" :
    d.impact_severity.includes("Severe") ? "severity-severe" :
    d.impact_severity.includes("Moderate") ? "severity-moderate" : "severity-normal"
  );

  // 8, 9, 10 Smart Dynamics
  document.getElementById("out-hit-run").innerText = d.smart_dynamics.hit_and_run_suspect;
  document.getElementById("out-traffic-status").innerText = d.smart_dynamics.post_crash_traffic;
  document.getElementById("out-secondary-warning").innerText = d.smart_dynamics.secondary_collision_warning;

  // Update Authority Dispatch Cards
  const auth = d.authority_dispatches;
  document.getElementById("auth-ems-status").innerText = auth.ems_ambulance.status;
  document.getElementById("auth-police-status").innerText = auth.traffic_police.status;
  document.getElementById("auth-fire-status").innerText = auth.fire_rescue.status;
  document.getElementById("auth-highway-status").innerText = auth.highway_control.status;

  // Update Main Video Player
  if (d.media_urls.processed_video) {
    const player = document.getElementById("main-video-player");
    document.getElementById("video-source").src = d.media_urls.processed_video;
    player.load();
    player.play().catch(() => {});
  }
}

// Leaflet Interactive Map & Simulator
function initMap() {
  map = L.map('sim-map').setView([12.9716, 77.5946], 13);
  L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
    attribution: '&copy; OpenStreetMap &copy; CARTO',
    maxZoom: 18
  }).addTo(map);
}

function toggleHighwaySimulation() {
  const btn = document.getElementById("btn-sim-traffic");
  if (simInterval) {
    clearInterval(simInterval);
    simInterval = null;
    btn.innerHTML = '<i class="fa-solid fa-play"></i> START HIGHWAY TRAFFIC SIMULATION';
    btn.className = "btn btn-secondary glow-blue";
    addNotification("Highway simulation paused.", "info");
  } else {
    simInterval = setInterval(updateSimulatedTraffic, 1500);
    btn.innerHTML = '<i class="fa-solid fa-pause"></i> PAUSE HIGHWAY SIMULATION';
    btn.className = "btn btn-outline";
    addNotification("Highway traffic simulation active. Vehicles monitoring NH-44 Expressway corridor.", "info");
  }
}

function updateSimulatedTraffic() {
  if (simulatedVehicles.length < 8) {
    const lat = 12.9716 + (Math.random() - 0.5) * 0.04;
    const lng = 77.5946 + (Math.random() - 0.5) * 0.04;
    const marker = L.circleMarker([lat, lng], {
      radius: 5,
      color: '#00f2fe',
      fillColor: '#00f2fe',
      fillOpacity: 0.8
    }).addTo(map);
    simulatedVehicles.push({ marker: marker, lat: lat, lng: lng });
  } else {
    simulatedVehicles.forEach(v => {
      v.lat += (Math.random() - 0.48) * 0.002;
      v.lng += (Math.random() - 0.48) * 0.002;
      v.marker.setLatLng([v.lat, v.lng]);
    });
  }
}

async function triggerRandomSimulatedCrash() {
  try {
    const res = await fetch("/api/simulated_accidents");
    const data = await res.json();
    const crashes = data.crashes;
    const crash = crashes[Math.floor(Math.random() * crashes.length)];
    
    // Add pulsing radar crash marker
    const crashIcon = L.divIcon({
      className: 'pulse-crash-icon',
      html: `<div style="background:#ff0055; width:24px; height:24px; border-radius:50%; border:3px solid #fff; box-shadow:0 0 15px #ff0055; cursor:pointer;"></div>`,
      iconSize: [24, 24]
    });
    
    const marker = L.marker([crash.lat, crash.lng], { icon: crashIcon }).addTo(map);
    
    const popupContent = `
      <div style="color:#000; padding:6px; font-family:sans-serif;">
        <h4 style="color:#ff0055; margin-bottom:4px;">🚨 ${crash.name}</h4>
        <p><strong>Collision:</strong> ${crash.collision_type}</p>
        <p><strong>Severity:</strong> ${crash.severity}</p>
        <p><strong>Vehicles:</strong> ${crash.vehicles}</p>
        <button onclick="openSimulatedCrashDetail('${crash.collision_type}', ${crash.vehicles}, '${crash.video}')" 
                style="margin-top:8px; padding:6px 12px; background:#00f2fe; color:#000; border:none; border-radius:6px; font-weight:bold; cursor:pointer;">
          <i class="fa-solid fa-diagram-project"></i> View 2D Trajectory Diagram
        </button>
      </div>
    `;
    
    marker.bindPopup(popupContent).openPopup();
    map.panTo([crash.lat, crash.lng]);
    
    addNotification(`🚨 SIMULATED CRASH ALERT triggered at ${crash.name} (${crash.collision_type})`, "alert");
  } catch (e) {
    console.error("Crash simulation failed:", e);
  }
}

function openSimulatedCrashDetail(collisionType, count, videoUrl) {
  openReconstructionModalWithData(collisionType, count);
}

// 2D Collision Trajectory Reconstruction Canvas Renderer
function draw2DReconstructionCanvas(collisionType, vehicleCount) {
  const canvas = document.getElementById("reconstruction-canvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;

  // Clear Canvas
  ctx.fillStyle = "#050811";
  ctx.fillRect(0, 0, w, h);

  // Draw Highway Road & Lane Markings
  ctx.strokeStyle = "rgba(255, 255, 255, 0.2)";
  ctx.lineWidth = 4;
  ctx.strokeRect(50, 40, w - 100, h - 80);

  // Dashed Center Lanes
  ctx.strokeStyle = "rgba(245, 158, 11, 0.6)";
  ctx.setLineDash([15, 15]);
  ctx.beginPath();
  ctx.moveTo(50, h / 2);
  ctx.lineTo(w - 50, h / 2);
  ctx.stroke();
  ctx.setLineDash([]);

  // Impact Point Starburst
  const ix = w / 2;
  const iy = h / 2;

  ctx.fillStyle = "rgba(255, 0, 85, 0.3)";
  ctx.beginPath();
  ctx.arc(ix, iy, 45, 0, Math.PI * 2);
  ctx.fill();

  ctx.fillStyle = "#ff0055";
  ctx.beginPath();
  ctx.arc(ix, iy, 12, 0, Math.PI * 2);
  ctx.fill();

  ctx.fillStyle = "#fff";
  ctx.font = "bold 12px sans-serif";
  ctx.fillText("IMPACT POINT (0,0)", ix - 50, iy - 20);

  // Draw Vehicles & Trajectory Vectors
  const colors = ["#00f2fe", "#10b981", "#f59e0b", "#8b5cf6"];
  for (let i = 0; i < Math.max(2, vehicleCount); i++) {
    const color = colors[i % colors.length];
    const isV1 = i === 0;

    // Pre-impact trajectory line
    const startX = isV1 ? ix - 200 : ix + 180;
    const startY = isV1 ? iy + 40 : iy - 50;

    ctx.strokeStyle = color;
    ctx.lineWidth = 3;
    ctx.setLineDash([6, 6]);
    ctx.beginPath();
    ctx.moveTo(startX, startY);
    ctx.lineTo(ix, iy);
    ctx.stroke();
    ctx.setLineDash([]);

    // Vehicle Box
    ctx.fillStyle = color;
    ctx.fillRect(startX - 18, startY - 10, 36, 20);
    ctx.fillStyle = "#000";
    ctx.font = "bold 11px sans-serif";
    ctx.fillText(`V-${i+1}`, startX - 10, startY + 4);

    // Speed Arrow
    ctx.fillStyle = color;
    ctx.font = "11px monospace";
    ctx.fillText(`${(65 + i*12).toFixed(1)} km/h`, startX - 25, startY - 15);
  }
}

// Modal Handlers
function openSnapshotModal() {
  document.getElementById("snapshot-modal").style.display = "flex";
}
function closeSnapshotModal() {
  document.getElementById("snapshot-modal").style.display = "none";
}

function openTelemetryModal() {
  if (lastReportData) {
    openReconstructionModalWithData(lastReportData.collision_type, lastReportData.vehicle_count);
  } else {
    openReconstructionModalWithData("Rear-end Collision", 2);
  }
}

function openReconstructionModalWithData(type, count) {
  document.getElementById("reconstruction-modal").style.display = "flex";
  draw2DReconstructionCanvas(type, count);
  
  const detailsEl = document.getElementById("reconstruction-details");
  detailsEl.innerHTML = `
    <div class="deliverable-box">
      <div class="box-label">COLLISION GEOMETRY</div>
      <div class="box-main-text">${type}</div>
    </div>
    <div class="deliverable-box">
      <div class="box-label">PRE-IMPACT VELOCITY</div>
      <div class="box-main-text text-amber">V1: 74.2 km/h | V2: 61.5 km/h</div>
    </div>
    <div class="deliverable-box">
      <div class="box-label">ESTIMATED IMPACT ANGLE</div>
      <div class="box-main-text">14.8° Rear Offset</div>
    </div>
  `;
}

function closeReconstructionModal() {
  document.getElementById("reconstruction-modal").style.display = "none";
}

// Notification Feed Helper
function addNotification(msg, type = "info") {
  const list = document.getElementById("notifications-stream");
  if (!list) return;
  
  const notif = document.createElement("div");
  notif.className = `notif-item ${type === "alert" ? "notif-alert" : ""}`;
  
  const time = new Date().toLocaleTimeString();
  notif.innerHTML = `<span class="notif-time">${time}</span><p>${msg}</p>`;
  
  list.insertBefore(notif, list.firstChild);
}
