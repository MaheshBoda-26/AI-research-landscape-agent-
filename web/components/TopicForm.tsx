"use client";

import { useState } from "react";

const EXAMPLES = [
  "retrieval-augmented generation",
  "diffusion policy learning",
  "test-time scaling for LLM reasoning",
  "mixture-of-experts routing",
  "mechanistic interpretability",
];

export function TopicForm({
  onSubmit,
  disabled,
  initialValue = "",
}: {
  onSubmit: (topic: string) => void;
  disabled?: boolean;
  initialValue?: string;
}) {
  const [topic, setTopic] = useState(initialValue);
  const cleaned = topic.trim();
  const valid = cleaned.length >= 2;

  return (
    <form
      onSubmit={(event) => {
        event.preventDefault();
        if (valid && !disabled) onSubmit(cleaned);
      }}
      className="w-full"
    >
      <div className="flex gap-2">
        <input
          value={topic}
          onChange={(event) => setTopic(event.target.value)}
          autoFocus
          aria-label="ML topic"
          placeholder="e.g. retrieval-augmented generation"
          className="h-11 flex-1 rounded-lg border border-[var(--color-edge)] bg-[var(--color-panel)] px-3.5 text-sm outline-none placeholder:text-[var(--color-muted)]/70 focus:border-[var(--color-accent)]/60 focus:ring-2 focus:ring-[var(--color-accent)]/15"
        />
        <button
          type="submit"
          disabled={!valid || disabled}
          className="h-11 rounded-lg bg-[var(--color-accent)] px-5 text-sm font-semibold text-[#0b1020] transition-opacity hover:opacity-90 disabled:opacity-35"
        >
          Map it
        </button>
      </div>

      <div className="mt-3 flex flex-wrap gap-1.5">
        {EXAMPLES.map((example) => (
          <button
            key={example}
            type="button"
            onClick={() => setTopic(example)}
            className="rounded-full border border-[var(--color-edge)] px-2.5 py-1 text-[11px] text-[var(--color-muted)] transition-colors hover:border-[var(--color-accent)]/50 hover:text-[var(--color-ink)]"
          >
            {example}
          </button>
        ))}
      </div>
    </form>
  );
}
