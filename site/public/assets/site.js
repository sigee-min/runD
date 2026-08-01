const CPP_KEYWORDS = new Set(
  [
    "alignas",
    "alignof",
    "auto",
    "break",
    "case",
    "catch",
    "class",
    "concept",
    "const",
    "consteval",
    "constexpr",
    "constinit",
    "continue",
    "co_await",
    "co_return",
    "co_yield",
    "decltype",
    "default",
    "delete",
    "do",
    "else",
    "enum",
    "explicit",
    "export",
    "extern",
    "for",
    "friend",
    "if",
    "inline",
    "namespace",
    "new",
    "noexcept",
    "operator",
    "private",
    "protected",
    "public",
    "requires",
    "return",
    "sizeof",
    "static",
    "static_assert",
    "struct",
    "switch",
    "template",
    "this",
    "throw",
    "try",
    "typedef",
    "typename",
    "union",
    "using",
    "virtual",
    "void",
    "volatile",
    "while",
  ],
);

const CPP_LITERALS = new Set(["false", "nullptr", "true"]);
const CPP_TYPES = new Set(
  [
    "bool",
    "char",
    "char8_t",
    "char16_t",
    "char32_t",
    "double",
    "float",
    "int",
    "int8_t",
    "int16_t",
    "int32_t",
    "int64_t",
    "long",
    "ptrdiff_t",
    "short",
    "signed",
    "size_t",
    "uint8_t",
    "uint16_t",
    "uint32_t",
    "uint64_t",
    "unsigned",
    "wchar_t",
    "array",
    "span",
    "string",
    "string_view",
    "vector",
  ],
);

const CMAKE_KEYWORDS = new Set(
  [
    "CONFIG",
    "EXACT",
    "INTERFACE",
    "LANGUAGES",
    "PRIVATE",
    "PUBLIC",
    "REQUIRED",
  ],
);

const SHELL_KEYWORDS = new Set(
  [
    "case",
    "do",
    "done",
    "elif",
    "else",
    "esac",
    "fi",
    "for",
    "function",
    "if",
    "in",
    "then",
    "until",
    "while",
  ],
);

const SHELL_COMMANDS = new Set(
  [
    "brew",
    "cat",
    "cd",
    "chmod",
    "cmake",
    "ctest",
    "curl",
    "git",
    "grep",
    "mkdir",
    "ninja",
    "otool",
    "printf",
    "set",
    "shasum",
    "sw_vers",
    "sysctl",
    "tar",
    "uname",
    "xcrun",
  ],
);

function pushToken(tokens, type, value) {
  if (!value) return;
  const previous = tokens.at(-1);
  if (previous?.type === type) {
    previous.value += value;
    return;
  }
  tokens.push({ type, value });
}

function isIdentifierStart(character) {
  return /[A-Za-z_]/.test(character);
}

function isIdentifierPart(character) {
  return /[A-Za-z0-9_]/.test(character);
}

function readQuoted(source, start, quote) {
  let index = start + 1;
  while (index < source.length) {
    if (source[index] === "\\") {
      index += 2;
      continue;
    }
    index += 1;
    if (source[index - 1] === quote) break;
  }
  return Math.min(index, source.length);
}

function nextNonSpaceIndex(source, start) {
  let index = start;
  while (index < source.length && /\s/.test(source[index])) index += 1;
  return index;
}

function tokenizeCpp(source) {
  const tokens = [];
  let index = 0;

  while (index < source.length) {
    const character = source[index];
    const next = source[index + 1] ?? "";

    if (character === "#") {
      const lineStart = source.lastIndexOf("\n", index - 1) + 1;
      if (source.slice(lineStart, index).trim() === "") {
        const lineEnd = source.indexOf("\n", index);
        const end = lineEnd === -1 ? source.length : lineEnd;
        pushToken(tokens, "preprocessor", source.slice(index, end));
        index = end;
        continue;
      }
    }

    if (character === "/" && next === "/") {
      const lineEnd = source.indexOf("\n", index);
      const end = lineEnd === -1 ? source.length : lineEnd;
      pushToken(tokens, "comment", source.slice(index, end));
      index = end;
      continue;
    }

    if (character === "/" && next === "*") {
      const close = source.indexOf("*/", index + 2);
      const end = close === -1 ? source.length : close + 2;
      pushToken(tokens, "comment", source.slice(index, end));
      index = end;
      continue;
    }

    if (character === '"' || character === "'") {
      const end = readQuoted(source, index, character);
      pushToken(tokens, "string", source.slice(index, end));
      index = end;
      continue;
    }

    if (/\d/.test(character)) {
      const match = source.slice(index).match(
        /^(?:0[xX][0-9A-Fa-f]+|0[bB][01]+|\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)(?:[uUlLfF]+)?/,
      );
      if (match) {
        pushToken(tokens, "number", match[0]);
        index += match[0].length;
        continue;
      }
    }

    if (isIdentifierStart(character)) {
      let end = index + 1;
      while (end < source.length && isIdentifierPart(source[end])) end += 1;
      const word = source.slice(index, end);
      const nextIndex = nextNonSpaceIndex(source, end);
      const followsScope = source.slice(nextIndex, nextIndex + 2) === "::";
      const followsCall = source[nextIndex] === "(";
      const precededByScope = source.slice(Math.max(0, index - 2), index) === "::";

      let type = "";
      if (CPP_KEYWORDS.has(word)) type = "keyword";
      else if (CPP_LITERALS.has(word)) type = "literal";
      else if (
        CPP_TYPES.has(word) ||
        /^(?:u?int\d+_t)$/.test(word) ||
        /^[A-Z][A-Za-z0-9_]*$/.test(word)
      ) {
        type = "type";
      } else if (followsCall) {
        type = "function";
      } else if (followsScope || precededByScope) {
        type = "namespace";
      }

      pushToken(tokens, type, word);
      index = end;
      continue;
    }

    pushToken(tokens, "", character);
    index += 1;
  }

  return tokens;
}

function tokenizeCmake(source) {
  const tokens = [];
  let index = 0;

  while (index < source.length) {
    const character = source[index];

    if (character === "#") {
      const lineEnd = source.indexOf("\n", index);
      const end = lineEnd === -1 ? source.length : lineEnd;
      pushToken(tokens, "comment", source.slice(index, end));
      index = end;
      continue;
    }

    if (character === '"' || character === "'") {
      const end = readQuoted(source, index, character);
      pushToken(tokens, "string", source.slice(index, end));
      index = end;
      continue;
    }

    if (character === "$" && source[index + 1] === "{") {
      const close = source.indexOf("}", index + 2);
      const end = close === -1 ? source.length : close + 1;
      pushToken(tokens, "variable", source.slice(index, end));
      index = end;
      continue;
    }

    if (/\d/.test(character)) {
      const match = source.slice(index).match(/^\d+(?:\.\d+)*/);
      if (match) {
        pushToken(tokens, "number", match[0]);
        index += match[0].length;
        continue;
      }
    }

    if (isIdentifierStart(character)) {
      let end = index + 1;
      while (end < source.length && isIdentifierPart(source[end])) end += 1;
      const word = source.slice(index, end);
      const nextIndex = nextNonSpaceIndex(source, end);
      const followsScope = source.slice(nextIndex, nextIndex + 2) === "::";
      const precededByScope = source.slice(Math.max(0, index - 2), index) === "::";

      let type = "";
      if (CMAKE_KEYWORDS.has(word.toUpperCase())) type = "keyword";
      else if (source[nextIndex] === "(") type = "function";
      else if (followsScope || precededByScope) type = "namespace";

      pushToken(tokens, type, word);
      index = end;
      continue;
    }

    pushToken(tokens, "", character);
    index += 1;
  }

  return tokens;
}

function isShellCommandPosition(source, start) {
  const lineStart = source.lastIndexOf("\n", start - 1) + 1;
  const before = source.slice(lineStart, start);
  if (before.trim() !== "") return /(?:&&|\|\||[;|])\s*$/.test(before);

  if (lineStart === 0) return true;
  const previousLineEnd = lineStart - 1;
  const previousLineStart = source.lastIndexOf("\n", previousLineEnd - 1) + 1;
  return !source.slice(previousLineStart, previousLineEnd).trimEnd().endsWith("\\");
}

function tokenizeShell(source) {
  const tokens = [];
  let index = 0;

  while (index < source.length) {
    const character = source[index];

    if (character === "#") {
      const lineEnd = source.indexOf("\n", index);
      const end = lineEnd === -1 ? source.length : lineEnd;
      pushToken(tokens, "comment", source.slice(index, end));
      index = end;
      continue;
    }

    if (character === '"' || character === "'") {
      const end = readQuoted(source, index, character);
      pushToken(tokens, "string", source.slice(index, end));
      index = end;
      continue;
    }

    if (character === "$") {
      const braced = source[index + 1] === "{";
      const match = braced
        ? source.slice(index).match(/^\$\{[^}]+\}/)
        : source.slice(index).match(/^\$[A-Za-z_][A-Za-z0-9_]*/);
      if (match) {
        pushToken(tokens, "variable", match[0]);
        index += match[0].length;
        continue;
      }
    }

    if (character === "-" && /[-A-Za-z]/.test(source[index + 1] ?? "")) {
      const match = source.slice(index).match(/^--?[^\s\\]+/);
      if (match) {
        pushToken(tokens, "option", match[0]);
        index += match[0].length;
        continue;
      }
    }

    if (/\d/.test(character)) {
      const match = source.slice(index).match(/^\d+(?:\.\d+)*/);
      if (match) {
        pushToken(tokens, "number", match[0]);
        index += match[0].length;
        continue;
      }
    }

    if (/[A-Za-z_./]/.test(character)) {
      const match = source.slice(index).match(/^[A-Za-z0-9_./:+-]+/);
      if (match) {
        const word = match[0];
        const commandPosition = isShellCommandPosition(source, index);
        const commandName = word.split("/").at(-1) ?? word;

        let type = "";
        if (SHELL_KEYWORDS.has(word)) type = "keyword";
        else if (
          commandPosition &&
          (SHELL_COMMANDS.has(word) ||
            SHELL_COMMANDS.has(commandName) ||
            word.startsWith("./") ||
            word.startsWith("tools/"))
        ) {
          type = "function";
        }

        pushToken(tokens, type, word);
        index += word.length;
        continue;
      }
    }

    pushToken(tokens, "", character);
    index += 1;
  }

  return tokens;
}

function tokenizePlain(source) {
  const tokens = [];
  let index = 0;

  while (index < source.length) {
    const character = source[index];
    if (character === '"' || character === "'") {
      const end = readQuoted(source, index, character);
      pushToken(tokens, "string", source.slice(index, end));
      index = end;
      continue;
    }

    if (/\d/.test(character)) {
      const match = source.slice(index).match(
        /^(?:0[xX][0-9A-Fa-f]+|\d+(?:[.,]\d+)*(?:\.\d+)?)/,
      );
      if (match) {
        pushToken(tokens, "number", match[0]);
        index += match[0].length;
        continue;
      }
    }

    pushToken(tokens, "", character);
    index += 1;
  }

  return tokens;
}

function detectCodeLanguage(source, sourcePath = "", label = "") {
  const normalizedPath = sourcePath.toLowerCase();
  const normalizedLabel = label.toLowerCase();

  if (normalizedPath.endsWith(".cpp") || normalizedPath.endsWith(".hpp")) return "cpp";
  if (normalizedLabel.includes("cmakelists") || normalizedLabel === "cmake") return "cmake";
  if (normalizedLabel.includes("shell")) return "shell";

  if (
    /\b(?:cmake_minimum_required|find_package|add_executable|add_library|target_link_libraries)\s*\(/.test(
      source,
    )
  ) {
    return "cmake";
  }

  if (
    /#include\s*[<"]|\brund::|\bstd::|\b(?:auto|const|constexpr|return|using)\s+|Target::|;\s*(?:\n|$)/.test(
      source,
    )
  ) {
    return "cpp";
  }

  if (
    /^(?:\s*)(?:set -|chmod |cmake |\.\/|tools\/|brew |xcrun |shasum |tar )/m.test(source)
  ) {
    return "shell";
  }

  return "plain";
}

function tokenizeCode(source, language) {
  if (language === "cpp") return tokenizeCpp(source);
  if (language === "cmake") return tokenizeCmake(source);
  if (language === "shell") return tokenizeShell(source);
  return tokenizePlain(source);
}

function renderCodeTokens(code, tokens, language) {
  const fragment = document.createDocumentFragment();
  for (const token of tokens) {
    if (!token.type) {
      fragment.append(document.createTextNode(token.value));
      continue;
    }

    const span = document.createElement("span");
    span.className = `tok-${token.type}`;
    span.textContent = token.value;
    fragment.append(span);
  }

  code.replaceChildren(fragment);
  code.dataset.highlighted = language;
}

function highlightCodeBlocks(root = document) {
  for (const code of root.querySelectorAll("pre.code > code")) {
    const source = code.textContent ?? "";
    const sourcePath = code.dataset.source ?? "";
    const field = code.closest(".doc-code, .hero-example");
    const label =
      field?.querySelector(".code-file, .example-heading")?.textContent?.trim() ?? "";
    const language = detectCodeLanguage(source, sourcePath, label);
    renderCodeTokens(code, tokenizeCode(source, language), language);
  }
}

if (typeof globalThis !== "undefined") {
  Object.defineProperty(globalThis, "__runDSyntax", {
    configurable: true,
    value: Object.freeze({
      detectCodeLanguage,
      tokenizeCode,
    }),
  });
}

if (typeof document !== "undefined") {
  highlightCodeBlocks();

  const menuButton = document.querySelector("[data-menu-button]");
  const menu = document.querySelector("[data-menu]");

  function closeMenu({ restoreFocus = false } = {}) {
    if (!menuButton || !menu) return;
    menuButton.setAttribute("aria-expanded", "false");
    menuButton.setAttribute("aria-label", "Open navigation");
    menu.dataset.open = "false";
    document.body.classList.remove("menu-open");
    if (restoreFocus) menuButton.focus();
  }

  if (menuButton && menu) {
    menuButton.addEventListener("click", () => {
      const shouldOpen = menuButton.getAttribute("aria-expanded") !== "true";
      menuButton.setAttribute("aria-expanded", String(shouldOpen));
      menuButton.setAttribute(
        "aria-label",
        shouldOpen ? "Close navigation" : "Open navigation",
      );
      menu.dataset.open = String(shouldOpen);
      document.body.classList.toggle("menu-open", shouldOpen);

      if (shouldOpen) {
        menu.querySelector("a")?.focus();
      }
    });

    menu.addEventListener("click", (event) => {
      if (event.target instanceof HTMLAnchorElement) closeMenu();
    });

    document.addEventListener("keydown", (event) => {
      if (event.key === "Escape" && menu.dataset.open === "true") {
        closeMenu({ restoreFocus: true });
      }
    });

    window.addEventListener("resize", () => {
      if (window.innerWidth >= 840) closeMenu();
    });
  }

  const copyButtons = [...document.querySelectorAll("[data-copy]")];
  for (const button of copyButtons) {
    button.addEventListener("click", async () => {
      const selector = button.dataset.copy;
      const source = selector ? document.querySelector(selector) : null;
      if (!source) return;

      try {
        await navigator.clipboard.writeText(source.textContent ?? "");
        const previous = button.textContent;
        button.textContent = "Copied";
        button.dataset.copied = "true";
        window.setTimeout(() => {
          button.textContent = previous;
          delete button.dataset.copied;
        }, 1600);
      } catch {
        button.textContent = "Select code to copy";
      }
    });
  }

  const currentPath = window.location.pathname.replace(/\/+$/, "/");
  for (const link of document.querySelectorAll("[data-doc-link]")) {
    const href = new URL(link.href).pathname.replace(/\/+$/, "/");
    if (href === currentPath) {
      link.setAttribute("aria-current", "page");
    }
  }
}
