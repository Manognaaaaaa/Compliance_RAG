import { API_BASE_URL } from "./config.js";

const AVATAR_ICON_SVG =
  '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#C6A75E" stroke-width="1.8">' +
  '<path d="M12 3v18M5 7l-3 6a4 4 0 0 0 8 0l-3-6M19 7l-3 6a4 4 0 0 0 8 0l-3-6M5 7h14M5 7l7-4 7 4"/></svg>';

const state = {
  messages: [], // { role: "user" | "assistant", text: string, sources?: {doc, page}[], isError?: boolean }
  inputValue: "",
  isThinking: false,
};

const conversationEl = document.getElementById("conversation");
const emptyStateEl = document.getElementById("empty-state");
const inputEl = document.getElementById("input");
const sendBtn = document.getElementById("send-btn");

function createAvatar() {
  const avatar = document.createElement("div");
  avatar.className = "avatar";
  avatar.innerHTML = AVATAR_ICON_SVG; // static, trusted markup - no interpolated data
  return avatar;
}

function createUserRow(msg) {
  const row = document.createElement("div");
  row.className = "msg-row user";

  const bubble = document.createElement("div");
  bubble.className = "bubble user";
  bubble.textContent = msg.text;

  row.appendChild(bubble);
  return row;
}

const NO_ANSWER_PATTERN = /i don't have enough information/i;

// The LLM marks emphasis with Markdown **bold**. Render just that, building
// <strong> nodes directly rather than via innerHTML, so model output can
// never inject markup.
function appendFormatted(el, text) {
  text.split(/\*\*(.+?)\*\*/gs).forEach((part, i) => {
    if (!part) return;
    if (i % 2 === 1) {
      const strong = document.createElement("strong");
      strong.textContent = part;
      el.appendChild(strong);
    } else {
      el.appendChild(document.createTextNode(part));
    }
  });
}

function createAssistantRow(msg) {
  const row = document.createElement("div");
  row.className = "msg-row assistant";
  row.appendChild(createAvatar());

  const column = document.createElement("div");
  column.className = "assistant-column";

  const bubble = document.createElement("div");
  bubble.className = "bubble assistant" + (msg.isError ? " error" : "");
  appendFormatted(bubble, msg.text);
  column.appendChild(bubble);

  const isNoAnswer = NO_ANSWER_PATTERN.test(msg.text);
  if (msg.sources && msg.sources.length > 0 && !isNoAnswer) {
    const sourcesRow = document.createElement("div");
    sourcesRow.className = "sources";
    for (const source of msg.sources) {
      const chip = document.createElement("span");
      chip.className = "source-chip";
      chip.textContent = `${source.doc} — p. ${source.page}`;
      sourcesRow.appendChild(chip);
    }
    column.appendChild(sourcesRow);
  }

  row.appendChild(column);
  return row;
}

function createThinkingRow() {
  const row = document.createElement("div");
  row.className = "msg-row assistant";
  row.appendChild(createAvatar());

  const bubble = document.createElement("div");
  bubble.className = "thinking-bubble";
  for (let i = 0; i < 3; i++) {
    const dot = document.createElement("span");
    dot.className = "thinking-dot";
    bubble.appendChild(dot);
  }

  row.appendChild(bubble);
  return row;
}

function render() {
  const hasMessages = state.messages.length > 0;
  emptyStateEl.hidden = hasMessages;
  conversationEl.hidden = !hasMessages;

  if (hasMessages) {
    conversationEl.innerHTML = "";
    for (const msg of state.messages) {
      conversationEl.appendChild(
        msg.role === "user" ? createUserRow(msg) : createAssistantRow(msg)
      );
    }
    if (state.isThinking) {
      conversationEl.appendChild(createThinkingRow());
    }
    const anchor = document.createElement("div");
    anchor.className = "scroll-anchor";
    conversationEl.appendChild(anchor);
    anchor.scrollIntoView({ block: "end" });
  }

  sendBtn.disabled = state.isThinking || !state.inputValue.trim();
}

function autoGrowTextarea() {
  inputEl.style.height = "auto";
  inputEl.style.height = Math.min(inputEl.scrollHeight, 140) + "px";
}

async function sendMessage(text) {
  const trimmed = (text || "").trim();
  if (!trimmed || state.isThinking) return;

  state.messages.push({ role: "user", text: trimmed });
  state.inputValue = "";
  inputEl.value = "";
  autoGrowTextarea();
  state.isThinking = true;
  render();

  try {
    const response = await fetch(`${API_BASE_URL}/query`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: trimmed }),
    });

    if (!response.ok) {
      throw new Error(`Request failed with status ${response.status}`);
    }

    const data = await response.json();
    state.messages.push({
      role: "assistant",
      text: data.answer,
      sources: data.sources || [],
    });
  } catch (err) {
    state.messages.push({
      role: "assistant",
      text: "Something went wrong — try again.",
      isError: true,
    });
  } finally {
    state.isThinking = false;
    render();
  }
}

inputEl.addEventListener("input", () => {
  state.inputValue = inputEl.value;
  autoGrowTextarea();
  render();
});

inputEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendMessage(inputEl.value);
  }
});

sendBtn.addEventListener("click", () => sendMessage(inputEl.value));

document.querySelectorAll(".starter-card").forEach((card) => {
  card.addEventListener("click", () => {
    const text = card.querySelector(".starter-card-text").textContent;
    sendMessage(text);
  });
});

render();
