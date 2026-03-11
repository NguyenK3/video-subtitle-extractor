/**
 * VSE noVNC Connection Lifecycle Manager
 *
 * Implements a proper state machine for RFB connection management.
 * Directly imports core/rfb.js — no monkey-patching of the noVNC UI layer.
 *
 * States: READY → CONNECTING → CONNECTED → DISCONNECTED
 *                                        → RECONNECTING → …
 *                            → ERROR
 */
import RFB from "./core/rfb.js";

/* =====================================================================
 *  Constants
 * ===================================================================== */

const State = Object.freeze({
    READY:        "READY",
    CONNECTING:   "CONNECTING",
    CONNECTED:    "CONNECTED",
    DISCONNECTED: "DISCONNECTED",
    RECONNECTING: "RECONNECTING",
    ERROR:        "ERROR",
});

/** Which transitions are legal from each state. */
const TRANSITIONS = Object.freeze({
    [State.READY]:        [State.CONNECTING],
    [State.CONNECTING]:   [State.CONNECTED, State.ERROR, State.DISCONNECTED],
    [State.CONNECTED]:    [State.DISCONNECTED, State.RECONNECTING],
    [State.DISCONNECTED]: [State.CONNECTING],
    [State.RECONNECTING]: [State.CONNECTING, State.DISCONNECTED, State.ERROR],
    [State.ERROR]:        [State.CONNECTING],
});

const RECONNECT_BASE_MS  = 1000;
const RECONNECT_MAX_MS   = 30000;
const RECONNECT_FACTOR   = 2;
const MAX_RECONNECT_ATTEMPTS = 10;

/* =====================================================================
 *  Module State
 * ===================================================================== */

let currentState       = State.READY;
let rfb                = null;
let reconnectAttempts  = 0;
let reconnectTimer     = null;
let reconnectCountdown = null;
let resizeTimer        = null;
let debugInterval      = null;

/* =====================================================================
 *  Debug
 * ===================================================================== */

const DEBUG = (() => {
    const p = new URLSearchParams(window.location.search);
    const f = (p.get("debug") || "").toLowerCase();
    return f === "1" || f === "true";
})();

function log(...args) {
    if (!DEBUG) return;
    console.debug("[VSE]", ...args);
}

/* =====================================================================
 *  Query Parameters
 * ===================================================================== */

function param(name, fallback) {
    const re = new RegExp("[?&#]" + name + "=([^&#]*)");
    const m = String(document.location.href + window.location.hash).match(re);
    return m ? decodeURIComponent(m[1]) : fallback;
}

const CFG = Object.freeze({
    autoConnect:   param("autoconnect", "false").toLowerCase() === "true",
    autoReconnect: param("reconnect", "true").toLowerCase() === "true",
    host:          param("host", window.location.hostname),
    port:          param("port", window.location.port),
    path:          param("path", "websockify"),
    password:      param("password", ""),
    viewOnly:      param("view_only", "false").toLowerCase() === "true",
});

function buildWsUrl() {
    const scheme = window.location.protocol === "https:" ? "wss" : "ws";
    let url = scheme + "://" + CFG.host;
    if (CFG.port) url += ":" + CFG.port;
    url += "/" + CFG.path;
    return url;
}

/* =====================================================================
 *  DOM References  (initialized in bootstrap)
 * ===================================================================== */

let $container, $overlay, $label, $sublabel;
let $connectBtn, $reconnectBtn, $retryBtn, $cancelBtn;
let $fullscreenBtn, $ctrlaltdelBtn, $disconnectBtn;
let $toolbarTrigger, $debugBar;

function initDOM() {
    $container      = document.getElementById("noVNC_container");
    $overlay        = document.getElementById("vse_conn_overlay");
    $label          = $overlay.querySelector(".vse_label");
    $sublabel       = $overlay.querySelector(".vse_sublabel");
    $connectBtn     = document.getElementById("vse_connect_btn");
    $reconnectBtn   = document.getElementById("vse_reconnect_btn");
    $retryBtn       = document.getElementById("vse_retry_btn");
    $cancelBtn      = document.getElementById("vse_cancel_btn");
    $fullscreenBtn  = document.getElementById("vse_fullscreen_btn");
    $ctrlaltdelBtn  = document.getElementById("vse_ctrlaltdel_btn");
    $disconnectBtn  = document.getElementById("vse_disconnect_btn");
    $toolbarTrigger = document.getElementById("vse_toolbar_trigger");
    $debugBar       = document.getElementById("vse_debug_bar");

    // Button handlers
    $connectBtn.addEventListener("click",    () => connectVNC());
    $reconnectBtn.addEventListener("click",  () => connectVNC());
    $retryBtn.addEventListener("click",      () => connectVNC());
    $cancelBtn.addEventListener("click",     () => cancelReconnect());
    $disconnectBtn.addEventListener("click", () => userDisconnect());
    $fullscreenBtn.addEventListener("click", () => toggleFullscreen());
    $ctrlaltdelBtn.addEventListener("click", () => { if (rfb) rfb.sendCtrlAltDel(); });

    // Debug mode
    if (DEBUG) {
        document.body.classList.add("vse_debug_active");
        startDebugTicker();
    }
}

/* =====================================================================
 *  State Machine
 * ===================================================================== */

function transition(newState, detail) {
    const allowed = TRANSITIONS[currentState];
    if (!allowed || !allowed.includes(newState)) {
        log("BLOCKED transition:", currentState, "→", newState);
        return false;
    }
    const prev = currentState;
    currentState = newState;
    log("STATE:", prev, "→", newState, detail || "");
    syncUI(detail);
    return true;
}

/* =====================================================================
 *  UI Synchronization
 * ===================================================================== */

function syncUI(detail) {
    // Clear all state classes
    $overlay.className = "";
    $overlay.classList.add("vse_state_" + currentState.toLowerCase());

    // Hide toolbar trigger unless connected
    $toolbarTrigger.style.display = currentState === State.CONNECTED ? "" : "none";

    switch (currentState) {
        case State.READY:
            $label.textContent    = "Remote Desktop Viewer";
            $sublabel.textContent = "Click to connect to the VNC server";
            break;

        case State.CONNECTING:
            $label.textContent    = "Connecting\u2026";
            $sublabel.textContent = "";
            break;

        case State.CONNECTED:
            $overlay.classList.add("vse_hidden");
            break;

        case State.DISCONNECTED:
            $label.textContent    = "Disconnected";
            $sublabel.textContent = "The connection was closed cleanly.";
            break;

        case State.RECONNECTING: {
            const delay = getReconnectDelay();
            $label.textContent    = "Connection lost";
            $sublabel.textContent = "Reconnecting in " +
                Math.ceil(delay / 1000) + "s\u2026  (attempt " +
                (reconnectAttempts + 1) + "/" + MAX_RECONNECT_ATTEMPTS + ")";
            break;
        }

        case State.ERROR:
            $label.textContent    = detail || "Connection failed";
            $sublabel.textContent = "Check that the VNC server is running.";
            break;
    }
}

/* =====================================================================
 *  RFB Lifecycle
 * ===================================================================== */

function destroyRFB() {
    if (!rfb) return;
    try {
        rfb.removeEventListener("connect",             onRfbConnect);
        rfb.removeEventListener("disconnect",          onRfbDisconnect);
        rfb.removeEventListener("credentialsrequired", onRfbCredentials);
        rfb.removeEventListener("securityfailure",     onRfbSecurityFailure);
        rfb.removeEventListener("desktopname",         onRfbDesktopName);
        rfb.disconnect();
    } catch (_) { /* already torn down */ }
    rfb = null;
}

function connectVNC() {
    if (currentState === State.CONNECTING || currentState === State.CONNECTED) return;

    clearTimeout(reconnectTimer);
    clearInterval(reconnectCountdown);
    destroyRFB();

    if (!transition(State.CONNECTING)) return;

    const wsUrl = buildWsUrl();
    log("Opening WebSocket:", wsUrl);

    // Defer RFB creation until the container has a valid layout size.
    requestAnimationFrame(function waitForLayout() {
        // Guard: state may have changed if user clicked cancel/etc.
        if (currentState !== State.CONNECTING) return;

        const rect = $container.getBoundingClientRect();
        if (rect.width < 2 || rect.height < 2) {
            requestAnimationFrame(waitForLayout);
            return;
        }

        try {
            rfb = new RFB($container, wsUrl, {
                credentials: CFG.password ? { password: CFG.password } : undefined,
            });

            rfb.addEventListener("connect",             onRfbConnect);
            rfb.addEventListener("disconnect",          onRfbDisconnect);
            rfb.addEventListener("credentialsrequired", onRfbCredentials);
            rfb.addEventListener("securityfailure",     onRfbSecurityFailure);
            rfb.addEventListener("desktopname",         onRfbDesktopName);

            rfb.viewOnly       = CFG.viewOnly;
            rfb.scaleViewport  = true;
            rfb.resizeSession  = true;

            log("RFB instance created, waiting for connect event");
        } catch (err) {
            log("RFB constructor threw:", err);
            transition(State.ERROR, "Failed to create connection: " + err.message);
        }
    });
}

/* =====================================================================
 *  RFB Event Handlers
 * ===================================================================== */

function onRfbConnect() {
    log("RFB event: connect");
    reconnectAttempts = 0;
    transition(State.CONNECTED);
    applyViewportScaling();
}

function onRfbDisconnect(e) {
    const clean = !!(e.detail && e.detail.clean);
    log("RFB event: disconnect, clean=" + clean);
    rfb = null;

    if (currentState === State.CONNECTING) {
        // Failed during connection attempt
        transition(State.ERROR, "Unable to reach VNC server");
        return;
    }

    if (clean) {
        transition(State.DISCONNECTED);
    } else if (CFG.autoReconnect && reconnectAttempts < MAX_RECONNECT_ATTEMPTS) {
        scheduleReconnect();
    } else {
        const msg = reconnectAttempts >= MAX_RECONNECT_ATTEMPTS
            ? "Failed after " + MAX_RECONNECT_ATTEMPTS + " attempts"
            : "Connection lost";
        transition(State.ERROR, msg);
    }
}

function onRfbCredentials() {
    const pw = prompt("VNC Password Required:");
    if (pw && rfb) {
        rfb.sendCredentials({ password: pw });
    }
}

function onRfbSecurityFailure(e) {
    const reason = (e.detail && e.detail.reason) || "unknown";
    log("RFB event: securityfailure —", reason);
    destroyRFB();
    transition(State.ERROR, "Security failure: " + reason);
}

function onRfbDesktopName(e) {
    const name = (e.detail && e.detail.name) || "";
    log("Desktop name:", name);
    document.title = name ? name + " \u2013 VSE Remote" : "VSE Remote Desktop";
}

/* =====================================================================
 *  User Actions
 * ===================================================================== */

function userDisconnect() {
    clearTimeout(reconnectTimer);
    clearInterval(reconnectCountdown);
    reconnectAttempts = 0;
    destroyRFB();
    transition(State.DISCONNECTED);
}

function cancelReconnect() {
    clearTimeout(reconnectTimer);
    clearInterval(reconnectCountdown);
    reconnectAttempts = 0;
    transition(State.DISCONNECTED);
}

/* =====================================================================
 *  Reconnect with Exponential Backoff
 * ===================================================================== */

function getReconnectDelay() {
    return Math.min(
        RECONNECT_BASE_MS * Math.pow(RECONNECT_FACTOR, reconnectAttempts),
        RECONNECT_MAX_MS
    );
}

function scheduleReconnect() {
    if (!transition(State.RECONNECTING)) return;

    const delay = getReconnectDelay();
    reconnectAttempts++;
    log("Scheduling reconnect #" + reconnectAttempts + " in " + delay + "ms");

    // Live countdown in the sublabel
    let remaining = Math.ceil(delay / 1000);
    reconnectCountdown = setInterval(() => {
        remaining--;
        if (remaining > 0 && currentState === State.RECONNECTING) {
            $sublabel.textContent = "Reconnecting in " + remaining +
                "s\u2026  (attempt " + reconnectAttempts + "/" +
                MAX_RECONNECT_ATTEMPTS + ")";
        }
    }, 1000);

    reconnectTimer = setTimeout(() => {
        clearInterval(reconnectCountdown);
        if (currentState === State.RECONNECTING) {
            connectVNC();
        }
    }, delay);
}

/* =====================================================================
 *  Fullscreen
 * ===================================================================== */

function toggleFullscreen() {
    if (!document.fullscreenElement) {
        $container.requestFullscreen().then(() => {
            applyViewportScaling();
        }).catch((err) => log("Fullscreen error:", err));
    } else {
        document.exitFullscreen();
    }
}

document.addEventListener("fullscreenchange", () => {
    log("Fullscreen:", !!document.fullscreenElement);
    // Allow the browser a frame to finalize the layout change
    requestAnimationFrame(() => applyViewportScaling());
});

/* =====================================================================
 *  Resize Handling
 * ===================================================================== */

function applyViewportScaling() {
    if (!rfb) return;
    rfb.scaleViewport = true;
    rfb.resizeSession = true;
    if (typeof rfb._updateClip === "function") rfb._updateClip();
}

window.addEventListener("resize", () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(applyViewportScaling, 150);
}, { passive: true });

window.addEventListener("orientationchange", () => {
    setTimeout(applyViewportScaling, 250);
}, { passive: true });

/* =====================================================================
 *  Debug Ticker
 * ===================================================================== */

function startDebugTicker() {
    debugInterval = setInterval(() => {
        if (!$debugBar) return;
        const parts = ["STATE: " + currentState];

        if (rfb) {
            const canvas = $container.querySelector("canvas");
            if (canvas) {
                parts.push("Canvas: " + canvas.width + "x" + canvas.height);
            }
        }
        parts.push("Viewport: " + window.innerWidth + "x" + window.innerHeight);
        parts.push("Reconnects: " + reconnectAttempts);

        $debugBar.textContent = parts.join("  |  ");
    }, 1000);
}

/* =====================================================================
 *  Bootstrap
 * ===================================================================== */

document.addEventListener("DOMContentLoaded", () => {
    initDOM();
    syncUI();

    log("Config:", JSON.stringify(CFG));
    log("Initial state: READY");

    if (CFG.autoConnect) {
        // Small delay to ensure the DOM is fully laid out
        setTimeout(() => connectVNC(), 80);
    }
});
