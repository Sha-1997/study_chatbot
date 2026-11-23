// Augmented Ilm (free + notes): study chatbot that answers from your own PDFs and notes.
// Free parts: Groq (or any OpenAI-compatible API), SQLite full-text search, optional Tavily search.
import express from "express";
import multer from "multer";
import Database from "better-sqlite3";
import pdf from "pdf-parse/lib/pdf-parse.js";

const BASE_URL = process.env.LLM_BASE_URL || "https://api.groq.com/openai/v1";
const API_KEY = process.env.LLM_API_KEY || "";
const MODEL = process.env.LLM_MODEL || "openai/gpt-oss-120b";
const TAVILY_KEY = process.env.TAVILY_API_KEY || "";
const PORT = process.env.PORT || 3000;

// ---------- Database ----------
const db = new Database("ilm_memory.db");
db.exec(`
CREATE TABLE IF NOT EXISTS memories (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  category TEXT NOT NULL,
  content TEXT NOT NULL,
  created TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS docs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT NOT NULL,
  chunk_count INTEGER NOT NULL,
  created TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks USING fts5(content, doc_id UNINDEXED, doc_name UNINDEXED);
`);

// ---------- Memory ----------
function saveMemory(category, content) {
  db.prepare("INSERT INTO memories (category, content) VALUES (?, ?)").run(category, content);
  return "Saved.";
}

function recallMemory(query = "") {
  const rows = query
    ? db
        .prepare(
          "SELECT category, content FROM memories WHERE content LIKE ? OR category LIKE ? ORDER BY id DESC LIMIT 20"
        )
        .all(`%${query}%`, `%${query}%`)
    : db.prepare("SELECT category, content FROM memories ORDER BY id DESC LIMIT 20").all();
  return rows.map((r) => `[${r.category}] ${r.content}`).join("\n") || "No memories yet.";
}

// ---------- Notes (RAG with full-text search) ----------
function chunkText(text, size = 1000, overlap = 150) {
  const clean = text.replace(/\r/g, "").replace(/[ \t]+/g, " ").replace(/\n{3,}/g, "\n\n").trim();
  const chunks = [];
  let start = 0;
  while (start < clean.length) {
    let end = Math.min(start + size, clean.length);
    if (end < clean.length) {
      // Prefer to break at a paragraph, sentence, or space near the end
      const window = clean.slice(start, end);
      const cut = Math.max(
        window.lastIndexOf("\n\n"),
        window.lastIndexOf(". "),
        window.lastIndexOf("\n"),
        window.lastIndexOf(" ")
      );
      if (cut > size * 0.5) end = start + cut + 1;
    }
    const piece = clean.slice(start, end).trim();
    if (piece) chunks.push(piece);
    if (end >= clean.length) break;
    start = Math.max(end - overlap, start + 1);
  }
  return chunks;
}

function addDocument(name, text) {
  // Replace an older upload with the same name
  const old = db.prepare("SELECT id FROM docs WHERE name = ?").all(name);
  for (const d of old) removeDocument(d.id);

  const chunks = chunkText(text);
  const { lastInsertRowid } = db
    .prepare("INSERT INTO docs (name, chunk_count) VALUES (?, ?)")
    .run(name, chunks.length);
  const insert = db.prepare("INSERT INTO chunks (content, doc_id, doc_name) VALUES (?, ?, ?)");
  db.transaction(() => {
    for (const c of chunks) insert.run(c, Number(lastInsertRowid), name);
  })();
  return chunks.length;
}

function removeDocument(id) {
  db.prepare("DELETE FROM chunks WHERE doc_id = ?").run(id);
  db.prepare("DELETE FROM docs WHERE id = ?").run(id);
}

function searchNotes(query) {
  const words = (query.match(/[\p{L}\p{N}\p{M}]+/gu) || []).filter((w) => w.length >= 2);
  if (!words.length) return "No searchable words in the query.";
  const match = words.map((w) => `"${w}"`).join(" OR ");
  let rows;
  try {
    rows = db
      .prepare("SELECT doc_name, content FROM chunks WHERE chunks MATCH ? ORDER BY rank LIMIT 5")
      .all(match);
  } catch (e) {
    return `Search error: ${e.message}`;
  }
  if (!rows.length) return "No matching passages found in the student's notes.";
  return rows.map((r) => `[Source: ${r.doc_name}]\n${r.content}`).join("\n\n---\n\n");
}

// ---------- Web search (optional) ----------
async function webSearch(query) {
  const res = await fetch("https://api.tavily.com/search", {
    method: "POST",
    headers: { "Content-Type": "application/json", Authorization: `Bearer ${TAVILY_KEY}` },
    body: JSON.stringify({ query, max_results: 4 }),
  });
  if (!res.ok) return `Search failed (${res.status}).`;
  const data = await res.json();
  return (
    (data.results || []).map((r) => `${r.title}\n${r.url}\n${r.content}`).join("\n\n") ||
    "No results."
  );
}

// ---------- Tools ----------
const TOOLS = [
  {
    type: "function",
    function: {
      name: "search_notes",
      description:
        "Search the student's uploaded PDFs and notes. Use this first for any question that might be covered by their study material. Use a few specific keywords.",
      parameters: {
        type: "object",
        properties: { query: { type: "string" } },
        required: ["query"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "save_memory",
      description:
        "Save something worth remembering about the student: goals, level, subjects, weak topics, quiz results, or preferences.",
      parameters: {
        type: "object",
        properties: {
          category: {
            type: "string",
            enum: ["profile", "goal", "weak_topic", "quiz_result", "preference"],
          },
          content: { type: "string" },
        },
        required: ["category", "content"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "recall_memory",
      description: "Look up saved notes about the student, optionally filtered by keyword.",
      parameters: { type: "object", properties: { query: { type: "string" } } },
    },
  },
];

if (TAVILY_KEY) {
  TOOLS.push({
    type: "function",
    function: {
      name: "web_search",
      description: "Search the web for current or factual information.",
      parameters: {
        type: "object",
        properties: { query: { type: "string" } },
        required: ["query"],
      },
    },
  });
}

async function runTool(name, args) {
  if (name === "search_notes") return searchNotes(args.query || "");
  if (name === "save_memory") return saveMemory(args.category, args.content);
  if (name === "recall_memory") return recallMemory(args.query || "");
  if (name === "web_search") return await webSearch(args.query);
  return `Unknown tool: ${name}`;
}

function systemPrompt() {
  const docs = db.prepare("SELECT name FROM docs ORDER BY id").all().map((d) => d.name);
  return `You are Augmented Ilm, a patient and encouraging study tutor.

How you teach:
- Explain simply first, then add depth. Use examples.
- After explaining, check understanding with one short question.
- Offer flashcards or a mini quiz when it helps.
- If the student writes in Malayalam, answer in Malayalam.

Using the student's study material:
- Uploaded documents: ${docs.length ? docs.join(", ") : "none yet"}.
- For questions that their material might cover, call search_notes first, then answer from the passages and name the source document.
- If search_notes finds nothing relevant, say so, then answer from general knowledge and make clear it is not from their notes.
- Never invent quotes or page contents.
${TAVILY_KEY ? "- Use web_search for current events or facts not in their notes, and name your sources.\n" : ""}
Memory:
- When you learn something lasting about the student (goal, level, weak topic, quiz result), call save_memory. Don't announce it every time.

What you already know about the student:
${recallMemory()}
`;
}

// ---------- Chat loop ----------
const sessions = new Map();

async function chatTurn(messages) {
  for (let i = 0; i < 6; i++) {
    const res = await fetch(`${BASE_URL}/chat/completions`, {
      method: "POST",
      headers: { "Content-Type": "application/json", Authorization: `Bearer ${API_KEY}` },
      body: JSON.stringify({
        model: MODEL,
        messages: [{ role: "system", content: systemPrompt() }, ...messages],
        tools: TOOLS,
        tool_choice: "auto",
      }),
      signal: AbortSignal.timeout(60000),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error?.message || `API error ${res.status}`);

    const msg = data.choices[0].message;
    const clean = { role: "assistant", content: msg.content || "" };
    if (msg.tool_calls?.length) clean.tool_calls = msg.tool_calls;
    messages.push(clean);

    if (!msg.tool_calls?.length) return msg.content || "";

    for (const call of msg.tool_calls) {
      let args = {};
      try {
        args = JSON.parse(call.function.arguments || "{}");
      } catch {}
      const result = await runTool(call.function.name, args);
      messages.push({ role: "tool", tool_call_id: call.id, content: String(result) });
    }
  }
  return "Sorry, I got stuck. Please try again.";
}

// ---------- API ----------
const app = express();
app.use(express.json());
app.use(express.static("public"));

const upload = multer({ storage: multer.memoryStorage(), limits: { fileSize: 20 * 1024 * 1024 } });

app.post("/api/chat", async (req, res) => {
  const { sessionId, message } = req.body;
  if (!sessionId || !message) return res.status(400).json({ error: "Missing fields" });
  if (!sessions.has(sessionId)) sessions.set(sessionId, []);
  const messages = sessions.get(sessionId);
  messages.push({ role: "user", content: message });
  try {
    res.json({ reply: await chatTurn(messages) });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: err.message });
  }
});

app.post("/api/upload", upload.single("file"), async (req, res) => {
  if (!req.file) return res.status(400).json({ error: "No file received." });
  const name = Buffer.from(req.file.originalname, "latin1").toString("utf8");
  const lower = name.toLowerCase();
  try {
    let text;
    if (lower.endsWith(".pdf")) text = (await pdf(req.file.buffer)).text;
    else if (lower.endsWith(".txt") || lower.endsWith(".md")) text = req.file.buffer.toString("utf8");
    else return res.status(400).json({ error: "Use a PDF, .txt, or .md file." });

    if (!text || text.trim().length < 20)
      return res.status(422).json({
        error: "No readable text found. Scanned PDFs (images) need OCR first.",
      });

    const chunks = addDocument(name, text);
    res.json({ ok: true, name, chunks });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: err.message });
  }
});

app.get("/api/docs", (_req, res) =>
  res.json({ docs: db.prepare("SELECT id, name, chunk_count FROM docs ORDER BY id DESC").all() })
);

app.delete("/api/docs/:id", (req, res) => {
  removeDocument(Number(req.params.id));
  res.json({ ok: true });
});

app.get("/api/memory", (_req, res) => res.json({ memory: recallMemory() }));

app.delete("/api/memory", (_req, res) => {
  db.exec("DELETE FROM memories");
  res.json({ ok: true });
});

app.listen(PORT, () =>
  console.log(
    `Augmented Ilm running at http://localhost:${PORT} (model: ${MODEL}, web search: ${TAVILY_KEY ? "on" : "off"})`
  )
);