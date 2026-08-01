import { readFile, readdir, stat } from "node:fs/promises";
import { resolve } from "node:path";
import { runInNewContext } from "node:vm";
import { pages as requiredPages } from "./pages.mjs";

const root = resolve(import.meta.dirname, "..");
const failures = [];
const sourceByPage = new Map();
const repositoryRoot = resolve(root, "..");
const packageVersionAuthority = await readFile(
  resolve(repositoryRoot, "cmake/root/package.cmake"),
  "utf8",
);
const packageVersion = packageVersionAuthority.match(
  /set\(RUND_PACKAGE_VERSION "([0-9]+\.[0-9]+\.[0-9]+)"\)/,
)?.[1];
if (!packageVersion) {
  failures.push("cmake/root/package.cmake: package version authority is invalid");
}
const stylePath = resolve(root, "public/assets/styles.css");
const styles = await readFile(stylePath, "utf8");
const siteScriptPath = resolve(root, "public/assets/site.js");
const siteScript = await readFile(siteScriptPath, "utf8");
for (const menuLabel of ["Open navigation", "Close navigation"]) {
  if (!siteScript.includes(menuLabel)) {
    failures.push(`public/assets/site.js: missing menu state label: ${menuLabel}`);
  }
}
const forbiddenVisualEffects = [
  "gradient",
  "box-shadow",
  "backdrop-filter",
  "mask-image",
  "color-mix",
];

for (const effect of forbiddenVisualEffects) {
  if (styles.includes(effect)) {
    failures.push(`public/assets/styles.css: forbidden visual effect: ${effect}`);
  }
}
if (styles.includes("--font-serif")) {
  failures.push("public/assets/styles.css: retired display serif returned");
}
for (const requiredTypeStack of ["ui-sans-serif", "ui-monospace"]) {
  if (!styles.includes(requiredTypeStack)) {
    failures.push(`public/assets/styles.css: missing system type stack: ${requiredTypeStack}`);
  }
}
if (/font-size:\s*(?:8|9|10)px/.test(styles)) {
  failures.push("public/assets/styles.css: unreadably small fixed type returned");
}

const syntaxContext = {};
runInNewContext(siteScript, syntaxContext, { filename: siteScriptPath });
const syntax = syntaxContext.__runDSyntax;
if (!syntax) {
  failures.push("public/assets/site.js: syntax highlighter contract is unavailable");
} else {
  const syntaxCases = [
    {
      language: "cpp",
      sourcePath: "package/tests/consumer/example/compute.cpp",
      source:
        '#include <rund/compute.hpp>\nauto result = rund::compute::on(Target::cpu(), input).map("step", [](auto x) { return x + 2; }); // checked',
      types: [
        "preprocessor",
        "keyword",
        "namespace",
        "type",
        "function",
        "string",
        "number",
        "comment",
      ],
    },
    {
      language: "cmake",
      label: "CMakeLists.txt",
      source: `find_package(runD ${packageVersion} EXACT CONFIG REQUIRED)\ntarget_link_libraries(app PRIVATE runD::sdk)`,
      types: ["function", "number", "keyword", "namespace"],
    },
    {
      language: "shell",
      label: "shell",
      source: 'cmake -S . -B build\nprintf "$PWD"\n$RUND_PREFIX\n# checked',
      types: ["function", "option", "string", "variable", "comment"],
    },
  ];

  for (const syntaxCase of syntaxCases) {
    const detected = syntax.detectCodeLanguage(
      syntaxCase.source,
      syntaxCase.sourcePath ?? "",
      syntaxCase.label ?? "",
    );
    if (detected !== syntaxCase.language) {
      failures.push(
        `public/assets/site.js: expected ${syntaxCase.language}, detected ${detected}`,
      );
      continue;
    }

    const tokens = syntax.tokenizeCode(syntaxCase.source, detected);
    if (tokens.map((token) => token.value).join("") !== syntaxCase.source) {
      failures.push(`public/assets/site.js: ${detected} highlighting changed source bytes`);
    }
    const tokenTypes = new Set(tokens.map((token) => token.type));
    for (const expectedType of syntaxCase.types) {
      if (!tokenTypes.has(expectedType)) {
        failures.push(
          `public/assets/site.js: ${detected} highlighting missed ${expectedType}`,
        );
      }
    }
  }
}
for (const mobileNavigationRule of [
  "inset: 68px 0 0;",
  "overflow-y: auto;",
  "overscroll-behavior-y: contain;",
]) {
  if (!styles.includes(mobileNavigationRule)) {
    failures.push(
      `public/assets/styles.css: missing scroll-safe mobile navigation rule: ${mobileNavigationRule}`,
    );
  }
}

for (const page of requiredPages) {
  const path = resolve(root, page);
  const html = await readFile(path, "utf8");
  sourceByPage.set(page, html);
  const h1Count = (html.match(/<h1\b/g) ?? []).length;

  if (h1Count !== 1) {
    failures.push(`${page}: expected one h1, found ${h1Count}`);
  }
  if (!html.includes('href="#main"')) {
    failures.push(`${page}: missing skip link`);
  }
  if (!html.includes('id="main"')) {
    failures.push(`${page}: missing main landmark target`);
  }
  if (!html.includes("<title>")) {
    failures.push(`${page}: missing title`);
  }
  const header = html.match(/<header class="site-header">[\s\S]*?<\/header>/)?.[0] ?? "";
  if (!header) {
    failures.push(`${page}: missing global header`);
  } else {
    if (!header.includes(`<span class="brand-version">${packageVersion} Alpha</span>`)) {
      failures.push(`${page}: global header has a divergent brand version`);
    }
    if (!header.includes('class="brand" href="/runD/" aria-label="runD home"')) {
      failures.push(`${page}: global header brand link is inconsistent`);
    }
    if (!header.includes('aria-label="Open navigation"')) {
      failures.push(`${page}: global header menu has a divergent accessible label`);
    }

    const navLinks = [...header.matchAll(/<a\b[^>]*class="[^"]*\bnav-link\b[^"]*"[^>]*>[\s\S]*?<\/a>/g)]
      .map((match) => ({
        current: match[0].includes('aria-current="page"'),
        href: match[0].match(/href="([^"]+)"/)?.[1] ?? "",
        label: match[0]
          .replace(/<[^>]+>/g, "")
          .replace(/\s+/g, " ")
          .trim(),
        tag: match[0],
      }));
    const expectedNavigation = [
      ["/runD/", "Overview"],
      ["/runD/docs/", "Docs"],
      ["/runD/docs/api/", "API"],
      ["https://github.com/sigee-min/runD", "GitHub"],
      ["/runD/docs/start/", "Quick Start"],
    ];

    if (
      JSON.stringify(navLinks.map(({ href, label }) => [href, label])) !==
      JSON.stringify(expectedNavigation)
    ) {
      failures.push(`${page}: global header navigation order or labels diverged`);
    }
    if (!navLinks.at(-1)?.tag.includes("nav-cta")) {
      failures.push(`${page}: global header Quick Start is not the primary action`);
    }

    const expectedCurrent =
      page === "index.html"
        ? 0
        : page === "docs/api/index.html"
          ? 2
          : page === "docs/start/index.html"
            ? 4
            : page === "404.html"
              ? -1
              : 1;
    const currentIndices = navLinks
      .map(({ current }, index) => (current ? index : -1))
      .filter((index) => index !== -1);
    const expectedCurrentIndices = expectedCurrent === -1 ? [] : [expectedCurrent];
    if (JSON.stringify(currentIndices) !== JSON.stringify(expectedCurrentIndices)) {
      failures.push(`${page}: global header active destination is incorrect`);
    }
  }
  if (/wiki/i.test(html)) {
    failures.push(`${page}: retired Wiki authority or link returned`);
  }
  if (html.includes("og:image") || html.includes("twitter:image")) {
    failures.push(`${page}: generated social image metadata is not admitted`);
  }
  if (html.includes("summary_large_image")) {
    failures.push(`${page}: large social card requires a deliberately designed asset`);
  }
  if (page !== "404.html") {
    for (const property of ["og:title", "og:description", "og:url"]) {
      if (!html.includes(`property="${property}"`)) {
        failures.push(`${page}: missing text sharing metadata: ${property}`);
      }
    }
  }
  for (const match of html.matchAll(/href="#([^"]+)"/g)) {
    if (!html.includes(`id="${match[1]}"`)) {
      failures.push(`${page}: missing same-page fragment #${match[1]}`);
    }
  }
  const fragmentCount = (html.match(/<code data-kind="fragment">/g) ?? []).length;
  const labeledFragmentCount = (
    html.match(
      /<div class="code-toolbar">[\s\S]*?contextual fragment[\s\S]*?<\/div>\s*<pre class="code"><code data-kind="fragment">/gi,
    ) ?? []
  ).length;
  if (fragmentCount !== labeledFragmentCount) {
    failures.push(
      `${page}: ${fragmentCount} contextual code fragments, ${labeledFragmentCount} labeled`,
    );
  }
}

const decodedCodeSources = new Set();
const decodeCode = (html) =>
  html
    .replaceAll("&lt;", "<")
    .replaceAll("&gt;", ">")
    .replaceAll("&amp;", "&")
    .replaceAll("&quot;", '"')
    .replaceAll("&#39;", "'");

for (const [page, html] of sourceByPage) {
  for (const match of html.matchAll(/<code data-source="([^"]+)">([\s\S]*?)<\/code>/g)) {
    const sourcePath = match[1];
    if (!sourcePath.startsWith("package/tests/consumer/example/")) {
      failures.push(`${page}: checked example uses an unadmitted source path: ${sourcePath}`);
      continue;
    }
    const checked = (await readFile(resolve(repositoryRoot, sourcePath), "utf8"))
      .replaceAll("\r\n", "\n")
      .trimEnd();
    const rendered = decodeCode(match[2]).replaceAll("\r\n", "\n").trimEnd();
    if (rendered !== checked) {
      failures.push(`${page}: checked example diverged from ${sourcePath}`);
    }
    decodedCodeSources.add(sourcePath);
  }
}

for (const expectedExample of [
  "package/tests/consumer/example/compute.cpp",
  "package/tests/consumer/example/parity.cpp",
  "package/tests/consumer/example/program.cpp",
  "package/tests/consumer/example/replay.cpp",
  "package/tests/consumer/example/runtime.cpp",
  "package/tests/consumer/example/network.cpp",
  "package/tests/consumer/example/telemetry.cpp",
  "package/tests/consumer/example/async.cpp",
  "package/tests/consumer/example/compute-session.cpp",
  "package/tests/consumer/example/task.cpp",
  "package/tests/consumer/example/host.cpp",
  "package/tests/consumer/example/storage.cpp",
  "package/tests/consumer/example/cluster.cpp",
]) {
  if (!decodedCodeSources.has(expectedExample)) {
    failures.push(`site docs: missing checked example ${expectedExample}`);
  }
}

for (const [page, header, example] of [
  ["docs/compute/index.html", "rund/compute/async.hpp", "async.cpp"],
  ["docs/compute/index.html", "rund/compute/session.hpp", "compute-session.cpp"],
  ["docs/runtime/index.html", "rund/task.hpp", "task.cpp"],
  ["docs/runtime/index.html", "rund/host.hpp", "host.cpp"],
  ["docs/runtime/index.html", "rund/storage.hpp", "storage.cpp"],
  ["docs/runtime/index.html", "cluster/cluster.hpp", "cluster.cpp"],
]) {
  const html = sourceByPage.get(page);
  if (!html.includes(`&lt;${header}&gt;`)) {
    failures.push(`${page}: missing callable entry journey for <${header}>`);
  }
  if (!html.includes(`data-source="package/tests/consumer/example/${example}"`)) {
    failures.push(`${page}: missing checked source for <${header}>`);
  }
}

const computePage = sourceByPage.get("docs/compute/index.html");
for (const checkpointOrWindowApi of [
  "tile_repeat&lt;0u&gt;",
  "write_window",
  "LatestDeviceState",
  "SnapshotStorage",
  "snapshot_into",
]) {
  if (!computePage.includes(checkpointOrWindowApi)) {
    failures.push(
      `docs/compute/index.html: missing Pipeline boundary API ${checkpointOrWindowApi}`,
    );
  }
}

const landing = sourceByPage.get("index.html");
const landingSectionCount = (landing.match(/<section\b/g) ?? []).length;
if (landingSectionCount !== 5) {
  failures.push(`index.html: expected five focused sections, found ${landingSectionCount}`);
}

for (const retiredBlock of [
  'class="trust-strip"',
  'class="hero-note"',
  'class="release-sheet"',
  'class="shape-table"',
  'class="use-grid"',
  'class="closing"',
  'class="parity-panel"',
  'class="byte-ledger"',
  'class="code-demo"',
  'class="target-switch"',
  'class="proof-grid"',
  "data-target=",
  "data-route=",
  "data-evidence-target",
  "data-execution-target",
  "data-code-target",
]) {
  if (landing.includes(retiredBlock)) {
    failures.push(`index.html: retired repetitive block returned: ${retiredBlock}`);
  }
}

if (landing.includes("<table")) {
  failures.push("index.html: dense landing table returned");
}

if (!landing.includes("public API excerpt")) {
  failures.push("index.html: incomplete code fragment must be labeled as an excerpt");
}

for (const requiredPerformanceClaim of [
  "Compute-heavy warm map",
  "324.959 µs",
  "123.666 µs",
  "190.542 µs",
  "64 small GPU jobs",
  "7,347.062 → 227.645 µs",
  "9,136.105 → 583.146 µs",
]) {
  if (!landing.includes(requiredPerformanceClaim)) {
    failures.push(`index.html: missing scoped performance evidence: ${requiredPerformanceClaim}`);
  }
}

for (const misleadingPerformanceClaim of [
  "CPU won the light workload",
  "Light resident map",
]) {
  if (landing.includes(misleadingPerformanceClaim)) {
    failures.push(`index.html: setup cost presented as compute work: ${misleadingPerformanceClaim}`);
  }
}

const apiPage = sourceByPage.get("docs/api/index.html");
const headerRegistry = await readFile(
  resolve(repositoryRoot, "package/docs/surface/headers.tsv"),
  "utf8",
);
for (const row of headerRegistry.trim().split("\n").slice(1)) {
  const [kind, path] = row.split("\t");
  if (kind === "direct" && !apiPage.includes(`&lt;${path}&gt;`)) {
    failures.push(`docs/api/index.html: missing direct header ${path}`);
  }
}
for (const nonexistentHeader of ["rund/fixed.hpp", "rund/telemetry.hpp"]) {
  if (apiPage.includes(nonexistentHeader)) {
    failures.push(`docs/api/index.html: nonexistent public header returned: ${nonexistentHeader}`);
  }
}

async function walk(directory) {
  const entries = await readdir(directory);
  const paths = [];
  for (const entry of entries) {
    const path = resolve(directory, entry);
    const info = await stat(path);
    if (info.isDirectory()) {
      paths.push(...(await walk(path)));
    } else {
      paths.push(path);
    }
  }
  return paths;
}

const distRoot = resolve(root, "dist");
const distPaths = await walk(distRoot);

for (const [page, html] of sourceByPage) {
  for (const match of html.matchAll(/href="([^"]+)"/g)) {
    const href = match[1];
    if (!href.startsWith("/runD/")) continue;

    const url = new URL(href, "https://example.invalid");
    let relative = url.pathname.slice("/runD/".length);
    if (relative === "" || relative.endsWith("/")) relative += "index.html";

    const target = resolve(root, relative);
    try {
      const info = await stat(target);
      if (!info.isFile()) failures.push(`${page}: internal link is not a file: ${href}`);
    } catch {
      const publicTarget = resolve(root, "public", relative);
      try {
        const info = await stat(publicTarget);
        if (!info.isFile()) failures.push(`${page}: missing internal link: ${href}`);
      } catch {
        failures.push(`${page}: missing internal link: ${href}`);
      }
    }

    if (url.hash && relative.endsWith(".html")) {
      const targetHtml = sourceByPage.get(relative) ?? await readFile(target, "utf8");
      const id = decodeURIComponent(url.hash.slice(1));
      if (!targetHtml.includes(`id="${id}"`)) {
        failures.push(`${page}: missing fragment ${href}`);
      }
    }
  }
}

for (const path of distPaths) {
  if (!path.endsWith(".html")) continue;
  const html = await readFile(path, "utf8");
  if (html.includes('="/assets/')) {
    failures.push(`${path}: built asset bypasses /runD/ base`);
  }
}

if (failures.length > 0) {
  console.error(failures.join("\n"));
  process.exit(1);
}

console.log(`site contract: ${requiredPages.length} pages checked`);
