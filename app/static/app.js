(() => {
  const SUGGESTIONS = [
    "¿Por qué lo apodan Taladro?",
    "¿Quién fundó el club?",
    "¿Qué colores usaba antes de adoptar el verde y el blanco?",
    "¿Cuál es la capacidad del estadio?",
    "¿Qué apodo le puso el público por Eva Perón?",
    "¿Cuál es el máximo goleador histórico?",
  ];
  const MAX_CHARS = 1000;
  const WAKE_MS = 8000;
  const TIMEOUT_MS = 90000;
  const NOT_FOUND = "No encuentro esa información en los documentos.";

  const app = document.querySelector(".app");
  const log = document.getElementById("log");
  const form = document.getElementById("composer");
  const input = document.getElementById("q");
  const sendBtn = document.getElementById("send");
  const count = document.getElementById("count");
  const resetBtn = document.getElementById("reset");
  const suggestionList = document.querySelector(".suggestions");
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)");

  let pending = false;
  let counter = 0;
  let epoch = 0;

  function el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  }

  function markIcon() {
    const ns = "http://www.w3.org/2000/svg";
    const svg = document.createElementNS(ns, "svg");
    svg.setAttribute("class", "mark");
    svg.setAttribute("aria-hidden", "true");
    svg.setAttribute("focusable", "false");
    const use = document.createElementNS(ns, "use");
    use.setAttribute("href", "#mark");
    svg.appendChild(use);
    return svg;
  }

  function stripes() {
    const wrap = el("span", "stripes");
    wrap.setAttribute("aria-hidden", "true");
    for (let i = 0; i < 9; i += 1) {
      const bar = document.createElement("i");
      bar.style.setProperty("--i", String(i));
      wrap.appendChild(bar);
    }
    return wrap;
  }

  function citeButton(ref) {
    const button = el("button", "cite", String(ref));
    button.type = "button";
    button.dataset.ref = String(ref);
    button.setAttribute("aria-label", `Ver la fuente ${ref}`);
    return button;
  }

  function renderInline(text, known) {
    const fragment = document.createDocumentFragment();
    for (const part of text.split(/(\*\*[^*\n]+?\*\*|\[\d+\])/g)) {
      if (!part) continue;
      const bold = part.match(/^\*\*(.+)\*\*$/);
      const cite = part.match(/^\[(\d+)\]$/);
      if (bold) {
        fragment.appendChild(el("strong", "", bold[1]));
      } else if (cite && known.has(Number(cite[1]))) {
        fragment.appendChild(citeButton(Number(cite[1])));
      } else {
        fragment.appendChild(document.createTextNode(part));
      }
    }
    return fragment;
  }

  function renderBlocks(text, sources) {
    const known = new Set(sources.map((source) => source.ref));
    const prose = el("div", "prose");
    for (const block of text.trim().split(/\n{2,}/)) {
      const lines = block.split("\n").filter((line) => line.trim());
      if (!lines.length) continue;
      const bullets = lines.every((line) => /^\s*[-*•]\s+/.test(line));
      const numbered = lines.every((line) => /^\s*\d+[.)]\s+/.test(line));
      if (bullets || numbered) {
        const list = el(numbered ? "ol" : "ul");
        if (numbered) list.start = Number(lines[0].match(/\d+/)[0]);
        for (const line of lines) {
          const item = el("li");
          item.appendChild(renderInline(line.replace(/^\s*([-*•]|\d+[.)])\s+/, ""), known));
          list.appendChild(item);
        }
        prose.appendChild(list);
      } else {
        const paragraph = el("p");
        lines.forEach((line, index) => {
          if (index) paragraph.appendChild(document.createElement("br"));
          paragraph.appendChild(renderInline(line, known));
        });
        prose.appendChild(paragraph);
      }
    }
    return prose;
  }

  function wordsOf(text) {
    const plain = text.toLowerCase().normalize("NFD").replace(/[\u0300-\u036f]/g, "");
    return new Set(plain.match(/[a-z0-9]{5,}/g) || []);
  }

  function excerptOf(fragment, answer) {
    const sentences = fragment.replace(/\s+/g, " ").trim().match(/[^.!?]+[.!?]*\s*/g) || [fragment];
    const target = wordsOf(answer);
    let best = 0;
    let bestScore = 0;
    sentences.forEach((sentence, index) => {
      let score = 0;
      for (const word of wordsOf(sentence)) if (target.has(word)) score += 1;
      if (score > bestScore) {
        bestScore = score;
        best = index;
      }
    });
    return sentences.slice(best, best + 2).join("").trim();
  }

  function toggleSource(item, button, text, excerpt, full) {
    const open = item.classList.toggle("open");
    text.textContent = open ? full : excerpt;
    button.textContent = open ? "Ver menos" : "Ver todo";
    button.setAttribute("aria-expanded", String(open));
  }

  function renderSources(sources, id, answer) {
    const details = el("details", "sources");
    details.appendChild(el("summary", "", sources.length === 1 ? "Fuente" : `Fuentes (${sources.length})`));
    const list = el("ol", "sources-list");
    for (const source of sources) {
      const item = el("li", "src");
      item.id = `src-${id}-${source.ref}`;
      item.appendChild(el("span", "src-ref", String(source.ref)));
      item.appendChild(el("p", "src-title", `${source.source}, fragmento ${source.chunk + 1}`));
      const full = source.text.trim();
      const excerpt = excerptOf(full, answer);
      const text = el("p", "src-text", excerpt);
      item.appendChild(text);
      if (excerpt.length < full.length) {
        const more = el("button", "src-more", "Ver todo");
        more.type = "button";
        more.setAttribute("aria-expanded", "false");
        more.addEventListener("click", () => toggleSource(item, more, text, excerpt, full));
        item.appendChild(more);
      }
      list.appendChild(item);
    }
    details.appendChild(list);
    return details;
  }

  function appendMessage(node) {
    node.classList.add("msg", "enter");
    log.appendChild(node);
    requestAnimationFrame(() => requestAnimationFrame(() => node.classList.remove("enter")));
    return node;
  }

  function scrollTo(node, block) {
    node.scrollIntoView({ block, behavior: reduceMotion.matches ? "auto" : "smooth" });
  }

  function botMessage(content) {
    const item = el("li", "msg-bot");
    item.dataset.id = String(++counter);
    const body = el("div", "body");
    body.appendChild(content);
    item.append(markIcon(), body);
    return item;
  }

  function fillBot(item, content) {
    const body = item.querySelector(".body");
    body.replaceChildren(...content);
  }

  function pendingContent() {
    const row = el("div", "pending-row");
    row.setAttribute("role", "status");
    row.append(stripes(), el("span", "pending-text", "Buscando en los documentos"));
    const note = el("p", "hint");
    note.hidden = true;
    return { row, note };
  }

  function setState(state) {
    app.dataset.state = state;
    resetBtn.hidden = state === "empty";
  }

  function refreshSend() {
    const empty = !input.value.trim();
    sendBtn.setAttribute("aria-disabled", String(empty || pending));
    sendBtn.textContent = pending ? "Buscando" : "Preguntar";
  }

  function updateCount() {
    const length = input.value.length;
    count.textContent = length > 800 ? `${length}/${MAX_CHARS}` : "";
    count.classList.toggle("over", length >= MAX_CHARS);
  }

  function autoGrow() {
    input.style.height = "auto";
    input.style.height = `${Math.min(input.scrollHeight, 152)}px`;
  }

  async function readResponse(response) {
    if (response.ok) return { data: await response.json() };
    if (response.status === 429) {
      const wait = Number(response.headers.get("Retry-After")) || 30;
      return { error: `Preguntaste muchas veces seguidas. Esperá ${wait} segundos y probá de nuevo.`, retry: true };
    }
    if (response.status === 422) {
      return { error: "No pude leer esa pregunta. Probá con una más corta.", retry: false };
    }
    return { error: "Se me trabó algo del lado del servidor. Probá de nuevo en un rato.", retry: true };
  }

  function networkError(error) {
    if (error && error.name === "AbortError") {
      return { error: "Tardó demasiado en responder. Probá de nuevo.", retry: true };
    }
    return { error: "No me pude conectar. Revisá tu conexión y probá de nuevo.", retry: true };
  }

  function answerContent(data, id) {
    const parts = [renderBlocks(data.answer, data.sources)];
    if (data.sources.length) {
      parts.push(renderSources(data.sources, id, data.answer));
    } else if (data.answer.startsWith(NOT_FOUND)) {
      parts.push(el("p", "hint", "Probá con otra pregunta sobre la historia, los ídolos, los récords o el estadio."));
    }
    return parts;
  }

  function errorContent(result, question, item) {
    const parts = [el("p", "", result.error)];
    if (result.retry) {
      const retry = el("button", "retry", "Probar de nuevo");
      retry.type = "button";
      retry.addEventListener("click", () => {
        item.remove();
        ask(question, true);
      });
      parts.push(retry);
    }
    return parts;
  }

  async function ask(question, isRetry = false) {
    if (pending) return;
    pending = true;
    const mine = epoch;
    refreshSend();
    setState("chat");

    if (!isRetry) {
      const userItem = el("li", "msg-user");
      const who = el("span", "sr-only", "Vos: ");
      userItem.append(who, document.createTextNode(question));
      appendMessage(userItem);
    }

    const { row, note } = pendingContent();
    const item = botMessage(row);
    item.querySelector(".body").appendChild(note);
    appendMessage(item);
    window.scrollTo({ top: document.documentElement.scrollHeight, behavior: reduceMotion.matches ? "auto" : "smooth" });

    const wake = setTimeout(() => {
      note.textContent = "Taladro estaba dormido y la primera respuesta tarda un poco más. Ya casi.";
      note.hidden = false;
    }, WAKE_MS);
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), TIMEOUT_MS);

    let result;
    try {
      const response = await fetch("/question", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, n_results: 4 }),
        signal: controller.signal,
      });
      result = await readResponse(response);
    } catch (error) {
      result = networkError(error);
    } finally {
      clearTimeout(wake);
      clearTimeout(timeout);
    }

    pending = false;
    refreshSend();
    if (mine !== epoch) return;

    fillBot(item, result.data ? answerContent(result.data, item.dataset.id) : errorContent(result, question, item));
    scrollTo(item, "start");
  }

  function submit() {
    const question = input.value.trim();
    if (!question || pending) return;
    input.value = "";
    autoGrow();
    updateCount();
    refreshSend();
    ask(question);
  }

  function reset() {
    epoch += 1;
    pending = false;
    log.replaceChildren();
    setState("empty");
    refreshSend();
    window.scrollTo(0, 0);
    input.focus();
  }

  for (const text of SUGGESTIONS) {
    const item = el("li");
    const button = el("button", "suggestion", text);
    button.type = "button";
    button.addEventListener("click", () => ask(text));
    item.appendChild(button);
    suggestionList.appendChild(item);
  }

  log.addEventListener("click", (event) => {
    const cite = event.target.closest(".cite");
    if (!cite) return;
    const message = cite.closest(".msg-bot");
    const details = message.querySelector(".sources");
    const target = document.getElementById(`src-${message.dataset.id}-${cite.dataset.ref}`);
    if (!details || !target) return;
    details.open = true;
    scrollTo(target, "nearest");
    target.classList.add("flash");
    setTimeout(() => target.classList.remove("flash"), 1200);
  });

  form.addEventListener("submit", (event) => {
    event.preventDefault();
    submit();
  });
  input.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      submit();
    }
  });
  input.addEventListener("input", () => {
    autoGrow();
    updateCount();
    refreshSend();
  });
  resetBtn.addEventListener("click", reset);

  refreshSend();
  fetch("/health").catch(() => {});
})();
