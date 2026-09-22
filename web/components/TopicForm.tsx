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
      <div className="flex flex-col gap-2.5 sm:flex-row">
        <label htmlFor="topic" className="visually-hidden">
          Research topic
        </label>
        <input
          id="topic"
          name="topic"
          value={topic}
          onChange={(event) => setTopic(event.target.value)}
          autoFocus
          autoComplete="off"
          placeholder="e.g. retrieval-augmented generation"
          className="h-11 min-w-0 flex-1 rounded-[var(--radius-md)] border border-[var(--color-edge)] bg-[var(--color-panel)] px-3.5 text-sm text-[var(--color-ink)] outline-none transition-colors placeholder:text-[var(--color-muted)] focus:border-[var(--color-accent)]/60 focus:ring-2 focus:ring-[var(--color-accent)]/15"
        />
        <button
          type="submit"
          disabled={!valid || disabled}
          className="button button--solid press sm:min-w-[7rem]"
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
            className="cursor-pointer rounded-full border border-[var(--color-edge)] px-3 py-1.5 text-xs text-[var(--color-muted)] transition-colors hover:border-[var(--color-accent)]/50 hover:text-[var(--color-ink)] focus-visible:outline-[var(--color-accent)]"
          >
            {example}
          </button>
        ))}
      </div>
    </form>
  );
}
