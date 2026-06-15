const HISTORY_KEY = "persona-rag-history-v1";
const MAX_HISTORY_ITEMS = 12;

function loadHistory() {
  try {
    const value = JSON.parse(localStorage.getItem(HISTORY_KEY) || "[]");
    if (!Array.isArray(value)) return [];
    return value
      .filter((item) => item && ["user", "assistant"].includes(item.role) && item.content)
      .slice(-MAX_HISTORY_ITEMS);
  } catch {
    return [];
  }
}

const state = {
  apiKeyAvailable: false,
  canAnswer: false,
  lastOutput: "",
  history: loadHistory(),
};

const $ = (id) => document.getElementById(id);

function setStatus(text, isError = false) {
  const el = $("statusText");
  el.textContent = text;
  el.className = isError ? "error" : "muted";
}

function saveHistory() {
  localStorage.setItem(HISTORY_KEY, JSON.stringify(state.history.slice(-MAX_HISTORY_ITEMS)));
}

function renderMemoryStatus() {
  const el = $("memoryState");
  if (!el) return;
  const turns = Math.floor(state.history.length / 2);
  el.textContent = `${turns} turns in context`;
  const clearBtn = $("clearMemoryBtn");
  if (clearBtn) clearBtn.disabled = state.history.length === 0;
}

function appendHistory(query, answer) {
  state.history.push({ role: "user", content: query });
  state.history.push({ role: "assistant", content: answer });
  state.history = state.history.slice(-MAX_HISTORY_ITEMS);
  saveHistory();
  renderMemoryStatus();
}

function clearHistory() {
  state.history = [];
  saveHistory();
  renderMemoryStatus();
  setStatus("Context cleared");
}

async function fetchJson(url, options = {}) {
  const response = await fetch(url, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.error || `HTTP ${response.status}`);
  }
  return data;
}

function renderPersona(data) {
  const persona = data.persona;
  $("targetName").textContent = persona.target_name;
  $("statMessages").textContent = persona.target_text_messages;
  $("statAvg").textContent = persona.avg_chars;
  $("statShort").textContent = persona.short_reply_rate;
  $("statLaugh").textContent = persona.laugh_rate;
  $("statQuestion").textContent = persona.question_rate;
  $("statEmoji").textContent = persona.emoji_rate;
  const vectorState = data.vector_index_available ? "FAISS vector ready" : "BM25 only";
  const loraState = data.local_lora_available ? "Local LoRA ready" : "No local LoRA";
  $("runtimeLine").textContent = `${data.doc_count} retrieval chunks · ${data.message_count} cleaned messages · ${vectorState} · ${loraState}`;

  state.apiKeyAvailable = data.api_key_available;
  state.canAnswer = state.apiKeyAvailable || data.local_lora_available;
  const apiState = $("apiState");
  apiState.textContent = state.apiKeyAvailable ? `${data.provider} · ${data.model}` : "Prompt mode";
  apiState.className = `status-pill ${state.apiKeyAvailable ? "ready" : "missing"}`;
  const answerBackend = $("answerBackend");
  answerBackend.querySelector('option[value="api"]').disabled = !state.apiKeyAvailable;
  answerBackend.querySelector('option[value="local_lora"]').disabled = !data.local_lora_available;
  if (!state.apiKeyAvailable && data.local_lora_available) {
    answerBackend.value = "local_lora";
  }
  $("answerBtn").disabled = !state.canAnswer;
}

function renderSources(sources) {
  const list = $("sourcesList");
  list.innerHTML = "";
  $("sourceCount").textContent = String(sources.length);
  for (const [index, source] of sources.entries()) {
    const details = document.createElement("details");
    const summary = document.createElement("summary");
    const meta = source.metadata || {};
    summary.textContent = `#${index + 1} · score ${source.score} · ${meta.start_time || "?"} → ${meta.end_time || "?"}`;
    const pre = document.createElement("pre");
    pre.textContent = source.text;
    details.append(summary, pre);
    list.appendChild(details);
  }
}

async function loadPersona() {
  const data = await fetchJson("/api/persona");
  renderPersona(data);
}

async function generate(wantAnswer) {
  const query = $("queryInput").value.trim();
  if (!query) {
    setStatus("Input is empty", true);
    return;
  }
  if (wantAnswer && !state.canAnswer) {
    setStatus("No answer backend is available", true);
    return;
  }

  $("promptBtn").disabled = true;
  $("answerBtn").disabled = true;
  setStatus(wantAnswer ? "Generating answer" : "Generating prompt");

  try {
    const data = await fetchJson("/api/chat", {
      method: "POST",
      body: JSON.stringify({
        query,
        top_k: Number($("topK").value || 5),
        retriever: $("retriever").value,
        answer_backend: $("answerBackend").value,
        history: state.history.slice(-MAX_HISTORY_ITEMS),
        save: $("saveRun").checked,
        answer: wantAnswer,
      }),
    });
    const output = wantAnswer && data.answer ? data.answer : data.prompt;
    state.lastOutput = output;
    $("outputBox").textContent = output;
    renderSources(data.sources || []);
    if (wantAnswer && data.answer) {
      appendHistory(query, data.answer);
    }
    setStatus(data.saved ? `Saved · ${data.saved.prompt_path}` : "Ready");
  } catch (error) {
    $("outputBox").textContent = "";
    setStatus(error.message, true);
  } finally {
    $("promptBtn").disabled = false;
    $("answerBtn").disabled = !state.canAnswer;
  }
}

async function copyOutput() {
  if (!state.lastOutput) return;
  await navigator.clipboard.writeText(state.lastOutput);
  setStatus("Copied");
}

$("promptBtn").addEventListener("click", () => generate(false));
$("answerBtn").addEventListener("click", () => generate(true));
$("copyBtn").addEventListener("click", copyOutput);
$("clearMemoryBtn").addEventListener("click", clearHistory);
$("reloadBtn").addEventListener("click", async () => {
  setStatus("Reloading");
  await fetchJson("/api/reload", { method: "POST", body: "{}" });
  await loadPersona();
  setStatus("Ready");
});

renderMemoryStatus();
loadPersona().catch((error) => setStatus(error.message, true));
