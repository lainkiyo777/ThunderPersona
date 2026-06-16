const HISTORY_KEY = "persona-rag-history-v1";
const CHAT_KEY = "persona-rag-current-chat-v1";
const MAX_HISTORY_ITEMS = 12;
const DISTILL_LEVELS = {
  light: {
    limit: 5000,
    buildVector: false,
    text: "轻量：最多导出 5,000 条消息，适合先试水和快速看风格，速度最快，但细节覆盖较少。",
  },
  medium: {
    limit: 20000,
    buildVector: false,
    text: "中量：最多导出 20,000 条消息，推荐默认。通常已经足够像，速度和质量比较平衡。",
  },
  full: {
    limit: 200000,
    buildVector: true,
    text: "全量级：最多导出 200,000 条消息，并建议建立 FAISS。最接近完整画像，但耗时和磁盘占用最高。",
  },
};

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

function loadChatTranscript() {
  try {
    const value = JSON.parse(localStorage.getItem(CHAT_KEY) || "[]");
    if (!Array.isArray(value)) return [];
    return value.filter((item) => item && ["user", "assistant"].includes(item.role) && item.content);
  } catch {
    return [];
  }
}

const state = {
  apiKeyAvailable: false,
  canAnswer: false,
  lastOutput: "",
  lastDnaText: "",
  history: loadHistory(),
  chatTranscript: loadChatTranscript(),
  selectedContact: null,
  activeSlug: "",
  distillPollTimer: null,
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

function saveChatTranscript() {
  localStorage.setItem(CHAT_KEY, JSON.stringify(state.chatTranscript));
}

function renderMemoryStatus() {
  const el = $("memoryState");
  if (!el) return;
  const turns = Math.floor(state.history.length / 2);
  el.textContent = `${turns} turns in context`;
  const clearBtn = $("clearMemoryBtn");
  if (clearBtn) clearBtn.disabled = state.history.length === 0;
  const saveBtn = $("saveChatBtn");
  if (saveBtn) saveBtn.disabled = state.chatTranscript.length === 0;
}

function appendHistory(query, answer) {
  const now = new Date().toISOString();
  const userTurn = { role: "user", content: query, timestamp: now };
  const answerTurn = { role: "assistant", content: answer, timestamp: new Date().toISOString() };
  state.history.push(userTurn);
  state.history.push(answerTurn);
  state.history = state.history.slice(-MAX_HISTORY_ITEMS);
  state.chatTranscript.push(userTurn, answerTurn);
  saveHistory();
  saveChatTranscript();
  renderMemoryStatus();
}

function clearHistory() {
  state.history = [];
  saveHistory();
  renderMemoryStatus();
  setStatus("Context cleared");
}

function resetChatSession() {
  state.history = [];
  state.chatTranscript = [];
  state.lastOutput = "";
  saveHistory();
  saveChatTranscript();
  renderMemoryStatus();
  renderSources([]);
  $("queryInput").value = "";
  $("outputBox").textContent = "";
  setStatus("New chat started");
}

function applyDistillLevel(levelName) {
  const level = DISTILL_LEVELS[levelName] || DISTILL_LEVELS.medium;
  $("exportLimit").value = String(level.limit);
  $("buildVector").checked = level.buildVector;
  $("distillLevelHelp").textContent = level.text;
}

function formatElapsed(seconds) {
  const total = Math.max(0, Math.floor(Number(seconds) || 0));
  const minutes = Math.floor(total / 60);
  const rest = total % 60;
  if (minutes <= 0) return `${rest}s`;
  return `${minutes}m ${String(rest).padStart(2, "0")}s`;
}

function renderDistillJob(job) {
  const panel = $("distillProgress");
  panel.classList.remove("hidden");
  $("distillProgressTitle").textContent = `${job.progress || 0}% · ${job.stage || job.status}`;
  $("distillElapsed").textContent = formatElapsed(job.elapsed_seconds);
  $("distillProgressBar").style.width = `${Math.max(0, Math.min(Number(job.progress) || 0, 100))}%`;
  $("distillProgressMessage").textContent = job.error || job.message || "Working";
}

function stopDistillPolling() {
  if (state.distillPollTimer) {
    window.clearTimeout(state.distillPollTimer);
    state.distillPollTimer = null;
  }
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

function renderList(id, items) {
  const list = $(id);
  list.innerHTML = "";
  for (const item of items || []) {
    const li = document.createElement("li");
    li.textContent = item;
    list.appendChild(li);
  }
}

function renderTags(id, items) {
  const list = $(id);
  list.innerHTML = "";
  const values = items && items.length ? items : ["No stable phrase"];
  for (const item of values) {
    const tag = document.createElement("span");
    tag.className = "tag";
    tag.textContent = item;
    list.appendChild(tag);
  }
}

function buildDnaText(persona) {
  const dna = persona.persona_dna || {};
  const confidence = dna.confidence || {};
  const rhythm = dna.rhythm || {};
  const voice = dna.voice || {};
  const interaction = dna.interaction || {};
  const phrases = dna.phrase_bank || persona.common_short_phrases || [];
  const boundaries = dna.honest_boundaries || [];
  const antiPatterns = dna.anti_patterns || [];
  const protocol = dna.response_protocol || [];
  return [
    `Target: ${persona.target_name}`,
    `Confidence: ${confidence.label || "Unknown"} - ${confidence.summary || ""}`,
    `Rhythm: ${rhythm.label || "-"} - ${rhythm.summary || ""}`,
    `Voice: ${voice.label || "-"} - ${voice.summary || ""}`,
    `Interaction: ${interaction.label || "-"} - ${interaction.summary || ""}`,
    `Phrase Bank: ${phrases.join("、") || "No stable phrase"}`,
    "",
    "Response Protocol:",
    ...protocol.map((item) => `- ${item}`),
    "",
    "Anti-patterns:",
    ...antiPatterns.map((item) => `- ${item}`),
    "",
    "Honest Boundaries:",
    ...boundaries.map((item) => `- ${item}`),
  ].join("\n");
}

function renderPersonaDna(persona) {
  const dna = persona.persona_dna || {};
  const confidence = dna.confidence || {};
  const rhythm = dna.rhythm || {};
  const voice = dna.voice || {};
  const interaction = dna.interaction || {};

  const confidenceEl = $("dnaConfidence");
  confidenceEl.textContent = confidence.label || "Unknown";
  confidenceEl.className = `status-pill confidence-${confidence.level || "unknown"}`;

  $("dnaRhythmLabel").textContent = rhythm.label || "-";
  $("dnaRhythmSummary").textContent = rhythm.summary || "-";
  renderList("dnaRhythmSignals", rhythm.signals || []);

  $("dnaVoiceLabel").textContent = voice.label || "-";
  $("dnaVoiceSummary").textContent = voice.summary || "-";
  renderList("dnaVoiceSignals", voice.signals || []);

  $("dnaInteractionLabel").textContent = interaction.label || "-";
  $("dnaInteractionSummary").textContent = interaction.summary || "-";
  renderList("dnaInteractionSignals", interaction.signals || []);

  renderTags("phraseBank", dna.phrase_bank || persona.common_short_phrases || []);
  renderList("dnaBoundaries", dna.honest_boundaries || []);
  renderList("dnaAntiPatterns", dna.anti_patterns || []);
  state.lastDnaText = buildDnaText(persona);
}

function renderPersona(data) {
  const persona = data.persona;
  state.activeSlug = data.active_slug || state.activeSlug;
  $("targetName").textContent = persona.target_name;
  $("statMessages").textContent = persona.target_text_messages;
  $("statAvg").textContent = persona.avg_chars;
  $("statShort").textContent = persona.short_reply_rate;
  $("statLaugh").textContent = persona.laugh_rate;
  $("statQuestion").textContent = persona.question_rate;
  $("statEmoji").textContent = persona.emoji_rate;
  renderPersonaDna(persona);

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

function contactTitle(contact) {
  return contact.display_name || contact.remark || contact.nick_name || contact.username || "";
}

function contactSubtitle(contact) {
  const parts = [];
  if (contact.remark && contact.nick_name && contact.remark !== contact.nick_name) {
    parts.push(contact.nick_name);
  }
  if (contact.username) parts.push(contact.username);
  return parts.join(" · ");
}

function renderContactResults(contacts) {
  const list = $("contactResults");
  list.innerHTML = "";
  if (!contacts.length) {
    const empty = document.createElement("div");
    empty.className = "muted small-text";
    empty.textContent = "No matching contacts";
    list.appendChild(empty);
    return;
  }

  for (const contact of contacts) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "contact-result";
    if (state.selectedContact && state.selectedContact.username === contact.username) {
      button.classList.add("selected");
    }
    const title = document.createElement("strong");
    title.textContent = contactTitle(contact);
    const sub = document.createElement("span");
    sub.textContent = contactSubtitle(contact);
    button.append(title, sub);
    button.addEventListener("click", () => {
      state.selectedContact = contact;
      $("selectedContact").textContent = `Selected: ${contactTitle(contact)}`;
      $("distillBtn").disabled = false;
      renderContactResults(contacts);
    });
    list.appendChild(button);
  }
}

async function searchContacts() {
  const query = $("contactSearch").value.trim();
  const data = await fetchJson(`/api/contacts?query=${encodeURIComponent(query)}&limit=30`);
  $("contactIndexState").textContent = data.updated_at
    ? `${data.total} contacts indexed · ${data.updated_at}`
    : "Contact index not built";
  renderContactResults(data.contacts || []);
}

function renderPersonas(personas) {
  const list = $("personaList");
  list.innerHTML = "";
  if (!personas.length) {
    const empty = document.createElement("div");
    empty.className = "muted small-text";
    empty.textContent = "No distilled personas";
    list.appendChild(empty);
    return;
  }

  for (const persona of personas) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `persona-item ${persona.active ? "active" : ""}`;
    const title = document.createElement("strong");
    title.textContent = persona.target_name || persona.slug;
    const meta = document.createElement("span");
    meta.textContent = `${persona.doc_count || 0} chunks · ${persona.message_count || 0} messages`;
    button.append(title, meta);
    button.addEventListener("click", () => switchPersona(persona.slug));
    list.appendChild(button);
  }
}

async function loadPersonas() {
  const data = await fetchJson("/api/personas");
  renderPersonas(data.personas || []);
}

async function switchPersona(slug) {
  if (!slug || slug === state.activeSlug) return;
  setStatus("Switching persona");
  const data = await fetchJson("/api/personas/switch", {
    method: "POST",
    body: JSON.stringify({ slug }),
  });
  state.history = [];
  state.chatTranscript = [];
  saveHistory();
  saveChatTranscript();
  renderMemoryStatus();
  renderPersona(data.runtime);
  await loadPersonas();
  renderSources([]);
  $("outputBox").textContent = "";
  state.lastOutput = "";
  setStatus("Ready");
}

async function refreshContacts() {
  $("refreshContactsBtn").disabled = true;
  $("contactIndexState").textContent = "Indexing contacts";
  try {
    const data = await fetchJson("/api/contacts/refresh", {
      method: "POST",
      body: JSON.stringify({ limit: 100000 }),
    });
    $("contactIndexState").textContent = `${data.total} contacts indexed · ${data.updated_at}`;
    await searchContacts();
  } catch (error) {
    $("contactIndexState").textContent = error.message;
  } finally {
    $("refreshContactsBtn").disabled = false;
  }
}

async function distillSelectedContact() {
  if (!state.selectedContact) return;
  stopDistillPolling();
  $("distillBtn").disabled = true;
  $("distillBtn").textContent = "Starting";
  setStatus(`Distilling ${contactTitle(state.selectedContact)}`);
  try {
    const data = await fetchJson("/api/distill/start", {
      method: "POST",
      body: JSON.stringify({
        contact: state.selectedContact,
        export_limit: Number($("exportLimit").value || 20000),
        build_vector: $("buildVector").checked,
      }),
    });
    renderDistillJob(data.job);
    $("distillBtn").textContent = "Running";
    pollDistillJob(data.job.job_id);
  } catch (error) {
    setStatus(error.message, true);
    $("distillBtn").textContent = "Distill & Switch";
    $("distillBtn").disabled = !state.selectedContact;
  }
}

async function pollDistillJob(jobId) {
  try {
    const job = await fetchJson(`/api/distill/job?job_id=${encodeURIComponent(jobId)}`);
    renderDistillJob(job);
    if (job.status === "completed") {
      stopDistillPolling();
      const runtime = job.result && job.result.runtime;
      if (runtime) {
        state.history = [];
        state.chatTranscript = [];
        saveHistory();
        saveChatTranscript();
        renderMemoryStatus();
        renderPersona(runtime);
        await loadPersonas();
        renderSources([]);
        $("outputBox").textContent = "";
        state.lastOutput = "";
        setStatus(`Distilled ${runtime.persona.target_name}`);
      } else {
        setStatus("Distillation completed");
      }
      $("distillBtn").textContent = "Distill & Switch";
      $("distillBtn").disabled = !state.selectedContact;
      return;
    }
    if (job.status === "failed") {
      stopDistillPolling();
      setStatus(job.error || "Distillation failed", true);
      $("distillBtn").textContent = "Distill & Switch";
      $("distillBtn").disabled = !state.selectedContact;
      return;
    }
    state.distillPollTimer = window.setTimeout(() => pollDistillJob(jobId), 1000);
  } catch (error) {
    stopDistillPolling();
    setStatus(error.message, true);
    $("distillBtn").textContent = "Distill & Switch";
    $("distillBtn").disabled = !state.selectedContact;
  }
}

function renderSources(sources) {
  const list = $("sourcesList");
  list.innerHTML = "";
  $("sourceCount").textContent = String(sources.length);
  for (const [index, source] of sources.entries()) {
    const details = document.createElement("details");
    const summary = document.createElement("summary");
    const meta = source.metadata || {};
    summary.textContent = `#${index + 1} · score ${source.score} · ${meta.start_time || "?"} -> ${meta.end_time || "?"}`;
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
    if (data.persona) {
      renderPersonaDna(data.persona);
    }
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
  setStatus("Copied output");
}

async function copyDna() {
  if (!state.lastDnaText) return;
  await navigator.clipboard.writeText(state.lastDnaText);
  setStatus("Copied DNA");
}

async function saveChat() {
  if (!state.chatTranscript.length) {
    setStatus("No chat to save", true);
    return;
  }
  $("saveChatBtn").disabled = true;
  setStatus("Saving chat");
  try {
    const data = await fetchJson("/api/chat-session/save", {
      method: "POST",
      body: JSON.stringify({
        messages: state.chatTranscript,
        active_slug: state.activeSlug,
      }),
    });
    setStatus(`Chat saved · ${data.markdown_path}`);
  } catch (error) {
    setStatus(error.message, true);
  } finally {
    renderMemoryStatus();
  }
}

$("promptBtn").addEventListener("click", () => generate(false));
$("answerBtn").addEventListener("click", () => generate(true));
$("copyBtn").addEventListener("click", copyOutput);
$("copyDnaBtn").addEventListener("click", copyDna);
$("clearMemoryBtn").addEventListener("click", clearHistory);
$("newChatBtn").addEventListener("click", resetChatSession);
$("saveChatBtn").addEventListener("click", saveChat);
$("refreshPersonasBtn").addEventListener("click", loadPersonas);
$("refreshContactsBtn").addEventListener("click", refreshContacts);
$("distillBtn").addEventListener("click", distillSelectedContact);
$("distillLevel").addEventListener("change", () => applyDistillLevel($("distillLevel").value));
let contactSearchTimer = null;
$("contactSearch").addEventListener("input", () => {
  window.clearTimeout(contactSearchTimer);
  contactSearchTimer = window.setTimeout(() => {
    searchContacts().catch((error) => {
      $("contactIndexState").textContent = error.message;
    });
  }, 180);
});
$("reloadBtn").addEventListener("click", async () => {
  setStatus("Reloading");
  await fetchJson("/api/reload", { method: "POST", body: "{}" });
  await loadPersona();
  await loadPersonas();
  setStatus("Ready");
});

renderMemoryStatus();
applyDistillLevel($("distillLevel").value);
Promise.all([loadPersona(), loadPersonas(), searchContacts()]).catch((error) => setStatus(error.message, true));
