import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

/**
 * Strip arXiv RSS boilerplate and decode common LaTeX escapes for display.
 * Input: "arXiv:2604.123v1 Announce Type: cross Abstract: The paper..."
 * Output: "The paper..."
 */
export function cleanAbstract(raw: string): string {
  if (!raw) return "";

  // Remove arXiv RSS prefix: "arXiv:XXXX Announce Type: ... Abstract:"
  let text = raw
    .replace(/^arXiv:\S+\s+Announce\s+Type:\s+\S+\s+Abstract:\s*/i, "")
    .replace(/^Abstract:\s*/i, "")
    .trim();

  // Decode common LaTeX escapes
  text = text
    .replace(/\\['`^"~=.uvcHr]\{?([a-zA-Z])\}?/g, "$1")  // accented chars
    .replace(/\\'([a-zA-Z])/g, "$1")   // e.g. moir\'e → moire
    .replace(/\\"([a-zA-Z])/g, "$1")
    .replace(/\\`([a-zA-Z])/g, "$1")
    .replace(/\\\^([a-zA-Z])/g, "$1")
    .replace(/\\~([a-zA-Z])/g, "$1")
    .replace(/\\textit\{([^}]+)\}/g, "$1")
    .replace(/\\textbf\{([^}]+)\}/g, "$1")
    .replace(/\\emph\{([^}]+)\}/g, "$1")
    .replace(/\$([^$]+)\$/g, "$1")       // inline math — strip delimiters
    .replace(/--/g, "–")
    .replace(/---/g, "—");

  return text;
}
