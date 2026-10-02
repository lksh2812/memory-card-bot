// Memory Lane web client.
//
// Two channels to the backend:
//   1. WebRTC via the Pipecat client: mic audio up, bot audio down, and live
//      game_state messages pushed by the bot (RTVI server messages).
//   2. Plain REST: create the session, poll the cached state, end the game,
//      and load the leaderboard / recent games.
//
// Rebuild after editing:  cd frontend && npm install && npm run build

import { PipecatClient } from "@pipecat-ai/client-js";
import { SmallWebRTCTransport, WavMediaManager } from "@pipecat-ai/small-webrtc-transport";

const $ = (id) => document.getElementById(id);

// The transport's default media manager loads a helper bundle from Daily's CDN.
// Open the page with ?media=native to use the browser's own audio APIs instead
// (useful on networks that block that CDN).
const useNativeMedia = new URLSearchParams(location.search).get("media") === "native";

const PHASE_LABEL = {
  waiting_for_call: "Connecting…",
  connecting: "Connecting…",
  presenting: "Listen closely…",
  listening: "Your turn: say the cards",
  evaluating: "Checking…",
  finished: "Game over",
  playing: "Playing",
};

let client = null;
let sessionId = null;
let pollTimer = null;
let botLine = null; // current bot message in the conversation log
let botSentences = [];

// ---- REST helpers ---------------------------------------------------------------

async function api(path, options = {}) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `${res.status} ${res.statusText}`);
  }
  return { data: await res.json(), cache: res.headers.get("X-Cache") };
}

// ---- rendering --------------------------------------------------------------------

function renderState(s) {
  const phase = s.status === "completed" || s.status === "abandoned" ? "finished" : s.phase;
  const pill = $("phase");
  pill.textContent = PHASE_LABEL[phase] || phase;
  pill.className = `pill ${phase}`;

  $("round").textContent = s.round_number || "–";
  $("score").textContent = s.score;
  const max = s.max_lives || 3;
  $("lives").innerHTML =
    "♥".repeat(Math.max(s.lives, 0)) + `<span class="lost">${"♥".repeat(Math.max(max - s.lives, 0))}</span>`;
  $("lives").setAttribute("aria-label", `${s.lives} of ${max} lives`);

  const cards = $("cards");
  cards.innerHTML = "";
  for (let i = 0; i < (s.sequence_length || 0); i++) {
    const c = document.createElement("div");
    c.className = "card";
    cards.appendChild(c);
  }
  cards.classList.toggle("reading", phase === "presenting");
  cards.setAttribute("aria-label", `${s.sequence_length} cards this round`);

  if (s.last_result) renderResult(s.last_result);

  if (phase === "finished") {
    const reason = {
      won: "You cleared every level!",
      out_of_lives: "Out of lives.",
      player_quit: "You ended the game.",
      ended_by_player: "You ended the game.",
      disconnected: "Call disconnected.",
    }[s.end_reason] || "";
    const final = $("final");
    final.hidden = false;
    final.textContent = `${reason} Final score: ${s.score} · Rounds cleared: ${s.rounds_cleared}`;
    $("end").hidden = true;
    $("again").hidden = false;
  }
}

function renderResult(r) {
  $("result").hidden = false;
  const exp = $("expected");
  const got = $("heard");
  exp.innerHTML = "";
  got.innerHTML = "";
  r.expected.forEach((card, i) => {
    exp.appendChild(chip(card, r.heard[i] === card ? "ok" : "bad"));
  });
  if (r.heard.length === 0) got.appendChild(chip("no cards heard", "none"));
  r.heard.forEach((card, i) => got.appendChild(chip(card, r.expected[i] === card ? "ok" : "bad")));
}

function chip(text, cls) {
  const el = document.createElement("span");
  el.className = `chip ${cls}`;
  el.textContent = text;
  return el;
}

function log(kind, text) {
  const li = document.createElement("li");
  li.className = kind;
  li.textContent = text;
  $("log").appendChild(li);
  li.scrollIntoView({ block: "nearest" });
  return li;
}

function setTalking(who, on) {
  $(who === "bot" ? "bot-talking" : "user-talking").classList.toggle("on", on);
}

async function loadBoards() {
  try {
    const [{ data: board }, { data: recent }] = await Promise.all([
      api("/api/leaderboard?limit=10"),
      api("/api/scores/recent?limit=8"),
    ]);
    $("leaderboard").innerHTML = board.length
      ? board.map((e) => `<li><span><span class="rank">${e.rank}</span>${escapeHtml(e.player_name)}</span><strong>${e.best_score}</strong></li>`).join("")
      : '<li class="empty">No games yet</li>';
    $("recent").innerHTML = recent.length
      ? recent.map((g) => `<li><span>${escapeHtml(g.player_name)}<br><span class="meta">${g.rounds_cleared} rounds · ${g.status}</span></span><strong>${g.score}</strong></li>`).join("")
      : '<li class="empty">No games yet</li>';
  } catch (e) {
    console.warn("could not load scores", e);
  }
}

function escapeHtml(s) {
  return s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
}

// ---- polling the cached state (shows the REST + Redis path) ------------------------

function startPolling() {
  stopPolling();
  const tick = async () => {
    if (!sessionId) return;
    try {
      const { data, cache } = await api(`/api/sessions/${sessionId}`);
      $("api-state").textContent = JSON.stringify(data, null, 2);
      const flag = $("cache-flag");
      flag.textContent = cache || "–";
      flag.className = cache || "";
      // If the WebRTC channel is gone (e.g. game ended), keep the UI in sync from REST.
      if (!client) renderState(data);
    } catch (e) {
      console.warn("poll failed", e);
    }
  };
  tick();
  pollTimer = setInterval(tick, 2000);
}

function stopPolling() {
  if (pollTimer) clearInterval(pollTimer);
  pollTimer = null;
}

// ---- the voice call ------------------------------------------------------------------

async function startGame(playerName) {
  const { data: session } = await api("/api/sessions", {
    method: "POST",
    body: JSON.stringify({ player_name: playerName }),
  });
  sessionId = session.session_id;
  $("lobby").hidden = true;
  $("live").hidden = false;
  renderState(session);
  startPolling();

  client = new PipecatClient({
    transport: new SmallWebRTCTransport(useNativeMedia ? { mediaManager: new WavMediaManager() } : {}),
    enableMic: true,
    enableCam: false,
    callbacks: {
      onBotReady: () => log("sys", "Connected. Listen for the cards."),
      onServerMessage: (msg) => {
        if (msg && msg.type === "game_state") renderState(msg.state);
      },
      onTrackStarted: (track, participant) => {
        // Play the bot's audio. SmallWebRTC reports remote tracks with no
        // participant, and our own mic with participant.local = true.
        if (track.kind === "audio" && !participant?.local) {
          $("bot-audio").srcObject = new MediaStream([track]);
        }
      },
      onBotStartedSpeaking: () => {
        setTalking("bot", true);
        botLine = log("bot", "…");
        botSentences = [];
      },
      onBotStoppedSpeaking: () => {
        setTalking("bot", false);
        if (botLine && botLine.textContent === "…") botLine.remove();
        botLine = null;
      },
      onBotOutput: (data) => {
        if (!botLine || !data || !data.text) return;
        if (data.aggregated_by === "word") {
          if (botSentences.length) return; // sentence-level text already shown
          botLine.textContent = (botLine.textContent === "…" ? "" : botLine.textContent + " ") + data.text;
        } else {
          botSentences.push(data.text);
          botLine.textContent = botSentences.join(" ");
        }
      },
      onUserStartedSpeaking: () => setTalking("user", true),
      onUserStoppedSpeaking: () => setTalking("user", false),
      onUserTranscript: (data) => {
        if (data.final && data.text.trim()) log("user", data.text);
      },
      onDisconnected: () => {
        log("sys", "Call ended");
        setTalking("bot", false);
        setTalking("user", false);
        client = null;
        loadBoards();
      },
      onError: (err) => log("sys", `Error: ${err?.data?.error || err?.data?.message || "unknown"}`),
    },
  });

  log("sys", "Connecting…");
  await client.connect({
    webrtcRequestParams: { endpoint: `/api/offer?session_id=${encodeURIComponent(sessionId)}` },
  });
}

async function endGame() {
  if (!sessionId) return;
  $("end").disabled = true;
  try {
    const { data } = await api(`/api/sessions/${sessionId}/end`, { method: "POST" });
    renderState(data);
  } catch (e) {
    log("sys", `Could not end the game: ${e.message}`);
  }
  // The bot says goodbye and hangs up on its own. If it's already gone, disconnect here.
  setTimeout(() => client && client.disconnect(), 8000);
  $("end").disabled = false;
}

function resetToLobby() {
  stopPolling();
  if (client) client.disconnect();
  client = null;
  sessionId = null;
  $("live").hidden = true;
  $("lobby").hidden = false;
  $("final").hidden = true;
  $("result").hidden = true;
  $("log").innerHTML = "";
  $("end").hidden = false;
  $("again").hidden = true;
  $("start").disabled = false;
  $("api-state").textContent = "No session yet";
  loadBoards();
}

// ---- wiring ------------------------------------------------------------------------------

$("lobby").addEventListener("submit", async (e) => {
  e.preventDefault();
  const name = $("name").value.trim();
  const err = $("lobby-error");
  err.hidden = true;
  if (!name) {
    err.textContent = "Enter your name first.";
    err.hidden = false;
    return;
  }
  $("start").disabled = true;
  try {
    await startGame(name);
  } catch (e2) {
    console.error(e2);
    resetToLobby();
    err.textContent = `Could not start: ${e2.message || e2}`;
    err.hidden = false;
  }
});
$("name").addEventListener("input", () => ($("lobby-error").hidden = true));
$("end").addEventListener("click", endGame);
$("again").addEventListener("click", resetToLobby);

loadBoards();
