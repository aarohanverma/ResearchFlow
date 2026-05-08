"use client";

import { useState, useRef, useEffect } from "react";
import { SendIcon, Loader2Icon, SearchIcon } from "lucide-react";
import MarkdownRenderer from "@/components/ui/MarkdownRenderer";
import { useAuthStore } from "@/store/auth";

const API_BASE = process.env.NEXT_PUBLIC_API_URL
  ? `${process.env.NEXT_PUBLIC_API_URL}/api/v1`
  : "/api/v1";

interface Message {
  role: "user" | "assistant";
  content: string;
  citations?: string[];
  scope?: string;
  streaming?: boolean;
  status?: string;
}

export default function ChatPage() {
  const { token } = useAuthStore();
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [namespace, setNamespace] = useState("cs.AI");
  const [loading, setLoading] = useState(false);
  const [statusText, setStatusText] = useState("");
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  async function send() {
    if (!input.trim() || loading) return;
    const query = input.trim();
    setInput("");
    setLoading(true);
    setStatusText("Searching knowledge base…");

    setMessages((m) => [...m, { role: "user", content: query }]);

    // Placeholder assistant message that we'll stream into
    setMessages((m) => [...m, { role: "assistant", content: "", streaming: true }]);

    let acc = "";
    let citations: string[] = [];
    let scope = "";

    try {
      const resp = await fetch(`${API_BASE}/chat`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...(token ? { Authorization: `Bearer ${token}` } : {}),
        },
        body: JSON.stringify({ query, namespace_key: namespace }),
      });

      if (!resp.body) throw new Error("no stream body");
      const reader = resp.body.getReader();
      const decoder = new TextDecoder();

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        const raw = decoder.decode(value, { stream: true });
        for (const line of raw.split("\n")) {
          if (!line.startsWith("data: ")) continue;
          try {
            const event = JSON.parse(line.slice(6));
            if (event.type === "status") {
              setStatusText(event.text ?? "");
            } else if (event.type === "chunk") {
              acc += event.text ?? "";
              setMessages((prev) => [
                ...prev.slice(0, -1),
                { role: "assistant", content: acc, streaming: true },
              ]);
            } else if (event.type === "meta") {
              citations = event.citations ?? [];
              scope = event.scope ?? "";
            } else if (event.type === "done") {
              setMessages((prev) => [
                ...prev.slice(0, -1),
                { role: "assistant", content: acc, citations, scope },
              ]);
            }
          } catch {}
        }
      }
    } catch (err: any) {
      setMessages((prev) => [
        ...prev.slice(0, -1),
        { role: "assistant", content: `Error: ${err.message}` },
      ]);
    } finally {
      setLoading(false);
      setStatusText("");
    }
  }

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="flex items-center gap-3 px-6 py-4 border-b border-gray-800">
        <h1 className="text-xl font-bold text-white">Chat over Knowledge</h1>
        <select
          value={namespace}
          onChange={(e) => setNamespace(e.target.value)}
          className="ml-auto bg-gray-800 border border-gray-700 rounded-lg px-3 py-1.5 text-sm text-gray-300 focus:outline-none"
        >
          {["cs.AI", "cs.ML", "cs.NLP", "cs.CV", "physics.quantum"].map((ns) => (
            <option key={ns} value={ns}>{ns}</option>
          ))}
        </select>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-6 py-6 space-y-6">
        {messages.length === 0 && (
          <div className="text-center text-gray-500 py-16">
            <p className="text-lg font-medium text-gray-400">Ask anything about your research space</p>
            <p className="text-sm mt-2">Questions are answered using only your indexed papers.</p>
            <div className="mt-6 space-y-2">
              {[
                "What are the main approaches to few-shot learning?",
                "Compare transformer and mamba architectures",
                "What open problems exist in LLM alignment?",
              ].map((q) => (
                <button
                  key={q}
                  onClick={() => setInput(q)}
                  className="block mx-auto text-sm text-indigo-400 hover:text-indigo-300 bg-indigo-950/30 hover:bg-indigo-950/50 border border-indigo-900/50 rounded-lg px-4 py-2 transition-colors"
                >
                  &ldquo;{q}&rdquo;
                </button>
              ))}
            </div>
          </div>
        )}

        {messages.map((msg, i) => (
          <div key={i} className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}>
            <div
              className={`max-w-2xl rounded-xl px-5 py-4 text-sm leading-relaxed ${
                msg.role === "user"
                  ? "bg-brand text-white"
                  : "bg-gray-900 border border-gray-800 text-gray-200"
              }`}
            >
              {msg.role === "user" ? (
                <span className="whitespace-pre-wrap">{msg.content}</span>
              ) : (
                <>
                  <MarkdownRenderer content={msg.content} />
                  {msg.streaming && msg.content === "" && (
                    <span className="inline-flex items-center gap-1.5 text-gray-500 text-xs">
                      <SearchIcon size={11} className="animate-pulse" />
                      {statusText || "Thinking…"}
                    </span>
                  )}
                  {msg.streaming && msg.content !== "" && (
                    <span className="inline-block w-1.5 h-4 bg-indigo-400 rounded-sm animate-pulse ml-0.5 align-middle" />
                  )}
                </>
              )}

              {msg.citations && msg.citations.length > 0 && (
                <div className="mt-3 pt-3 border-t border-gray-700">
                  <p className="text-xs text-gray-500 mb-1">
                    Sources {msg.scope && `(scope: ${msg.scope})`}
                  </p>
                  <div className="flex flex-wrap gap-1.5">
                    {msg.citations.slice(0, 5).map((id, j) => (
                      <a
                        key={id}
                        href={`/paper/${id}`}
                        className="text-xs bg-gray-800 text-indigo-400 hover:text-indigo-300 px-2 py-0.5 rounded"
                      >
                        [{j + 1}]
                      </a>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        ))}

        {loading && statusText && messages[messages.length - 1]?.content === "" && (
          <div className="flex justify-start">
            <div className="bg-gray-900 border border-gray-800 rounded-xl px-5 py-4 flex items-center gap-2 text-gray-400 text-sm">
              <Loader2Icon size={14} className="animate-spin" />
              {statusText}
            </div>
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      {/* Input */}
      <div className="px-6 py-4 border-t border-gray-800">
        <div className="flex gap-3">
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && !e.shiftKey && send()}
            placeholder="Ask about your research space…"
            className="flex-1 bg-gray-800 border border-gray-700 rounded-xl px-4 py-3 text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-brand text-sm"
            disabled={loading}
          />
          <button
            onClick={send}
            disabled={loading || !input.trim()}
            className="bg-brand hover:bg-indigo-600 disabled:opacity-40 text-white rounded-xl px-4 py-3 transition-colors"
          >
            <SendIcon size={16} />
          </button>
        </div>
      </div>
    </div>
  );
}
