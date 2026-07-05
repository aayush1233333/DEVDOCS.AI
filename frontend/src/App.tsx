import { useState } from "react";
import axios from "axios";

// The URL of our FastAPI backend
const API_URL = "https://devdocs-ai-sizy.onrender.com";

// TypeScript types — these define the shape of our data
interface Message {
  role: "user" | "ai";
  content: string;
  sources?: string[];
}

interface UploadedDoc {
  doc_id: string;
  filename: string;
}

export default function App() {
  // ── State ────────────────────────────────────────────────────────────────
  // State is data that when changed, causes the UI to re-render

  // Which screen are we on? "upload" or "chat"
  const [screen, setScreen] = useState<"upload" | "chat">("upload");

  // The document the user is currently chatting with
  const [activeDoc, setActiveDoc] = useState<UploadedDoc | null>(null);

  // All messages in the current chat
  const [messages, setMessages] = useState<Message[]>([]);

  // The question being typed right now
  const [question, setQuestion] = useState("");

  // Is something loading? (uploading or waiting for answer)
  const [loading, setLoading] = useState(false);

  // Error message if something goes wrong
  const [error, setError] = useState("");

  // ── Upload handler ────────────────────────────────────────────────────────
  const handleUpload = async (file: File) => {
    if (!file.name.endsWith(".pdf")) {
      setError("Only PDF files are supported");
      return;
    }

    setLoading(true);
    setError("");

    // FormData is how we send files over HTTP
    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await axios.post(`${API_URL}/upload`, formData);
      const doc: UploadedDoc = {
        doc_id: response.data.doc_id,
        filename: response.data.filename,
      };

      // Switch to chat screen with this document
      setActiveDoc(doc);
      setMessages([{
        role: "ai",
        content: `✅ **${doc.filename}** uploaded successfully! Ask me anything about it.`,
      }]);
      setScreen("chat");

    } catch (err) {
      setError("Upload failed. Make sure the backend is running.");
    } finally {
      setLoading(false);
    }
  };

  // ── Query handler ─────────────────────────────────────────────────────────
  const handleQuery = async () => {
    if (!question.trim() || !activeDoc) return;

    const userQuestion = question;
    setQuestion(""); // Clear input immediately

    // Add user message to chat
    setMessages(prev => [...prev, { role: "user", content: userQuestion }]);
    setLoading(true);

    try {
      const response = await axios.post(`${API_URL}/query`, {
        doc_id: activeDoc.doc_id,
        question: userQuestion,
      });

      // Add AI response to chat
      setMessages(prev => [...prev, {
        role: "ai",
        content: response.data.answer,
        sources: response.data.sources,
      }]);

    } catch (err) {
      setMessages(prev => [...prev, {
        role: "ai",
        content: "Sorry, something went wrong. Please try again.",
      }]);
    } finally {
      setLoading(false);
    }
  };

  // ── Upload Screen ─────────────────────────────────────────────────────────
  if (screen === "upload") {
    return (
      <div className="min-h-screen bg-gray-950 flex items-center justify-center p-4">
        <div className="w-full max-w-md">

          {/* Header */}
          <div className="text-center mb-8">
            <h1 className="text-4xl font-bold text-white mb-2">DevDocs AI</h1>
            <p className="text-gray-400">Upload a PDF and ask questions about it</p>
          </div>

          {/* Upload box */}
          <div
            className="border-2 border-dashed border-gray-700 rounded-2xl p-12 text-center cursor-pointer hover:border-blue-500 hover:bg-gray-900 transition-all"
            onClick={() => document.getElementById("file-input")?.click()}
            onDragOver={(e) => e.preventDefault()}
            onDrop={(e) => {
              e.preventDefault();
              const file = e.dataTransfer.files[0];
              if (file) handleUpload(file);
            }}
          >
            <div className="text-5xl mb-4">📄</div>
            <p className="text-white text-lg font-medium mb-2">
              Drop your PDF here
            </p>
            <p className="text-gray-500 text-sm">or click to browse</p>

            {/* Hidden file input */}
            <input
              id="file-input"
              type="file"
              accept=".pdf"
              className="hidden"
              onChange={(e) => {
                const file = e.target.files?.[0];
                if (file) handleUpload(file);
              }}
            />
          </div>

          {/* Loading state */}
          {loading && (
            <div className="mt-4 text-center text-blue-400">
              ⏳ Processing your PDF...
            </div>
          )}

          {/* Error state */}
          {error && (
            <div className="mt-4 text-center text-red-400">
              ❌ {error}
            </div>
          )}

        </div>
      </div>
    );
  }

  // ── Chat Screen ───────────────────────────────────────────────────────────
  return (
    <div className="min-h-screen bg-gray-950 flex flex-col">

      {/* Top bar */}
      <div className="bg-gray-900 border-b border-gray-800 px-4 py-3 flex items-center gap-3">
        <button
          onClick={() => { setScreen("upload"); setMessages([]); }}
          className="text-gray-400 hover:text-white transition-colors"
        >
          ← Back
        </button>
        <div className="text-white font-medium">📄 {activeDoc?.filename}</div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.map((msg, i) => (
          <div key={i} className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}>
            <div className={`max-w-2xl rounded-2xl px-4 py-3 ${
              msg.role === "user"
                ? "bg-blue-600 text-white"
                : "bg-gray-800 text-gray-100"
            }`}>
              {/* Message content */}
              <p className="whitespace-pre-wrap">{msg.content}</p>

              {/* Sources */}
              {msg.sources && msg.sources.length > 0 && (
                <div className="mt-3 pt-3 border-t border-gray-700">
                  <p className="text-xs text-gray-400 mb-2">📚 Sources:</p>
                  {msg.sources.map((source, j) => (
                    <p key={j} className="text-xs text-gray-500 mb-1 italic">
                      {j + 1}. {source}
                    </p>
                  ))}
                </div>
              )}
            </div>
          </div>
        ))}

        {/* Loading indicator */}
        {loading && (
          <div className="flex justify-start">
            <div className="bg-gray-800 rounded-2xl px-4 py-3 text-gray-400">
              ⏳ Thinking...
            </div>
          </div>
        )}
      </div>

      {/* Input box */}
      <div className="bg-gray-900 border-t border-gray-800 p-4">
        <div className="flex gap-3 max-w-4xl mx-auto">
          <input
            type="text"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handleQuery()}
            placeholder="Ask a question about your document..."
            className="flex-1 bg-gray-800 text-white rounded-xl px-4 py-3 outline-none focus:ring-2 focus:ring-blue-500 placeholder-gray-500"
          />
          <button
            onClick={handleQuery}
            disabled={loading || !question.trim()}
            className="bg-blue-600 hover:bg-blue-700 disabled:opacity-50 text-white rounded-xl px-6 py-3 font-medium transition-colors"
          >
            Send
          </button>
        </div>
      </div>

    </div>
  );
}
