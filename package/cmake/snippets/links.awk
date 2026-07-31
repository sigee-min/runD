BEGIN {
  failed = 0
  in_fence = 0
  link_count = 0
  anchored_count = 0

  while ((getline path < registry) > 0) {
    paths[path] = 1
  }
  close(registry)
}

function diagnostic(document, line, message) {
  print document ":" line ": " message > "/dev/stderr"
  failed = 1
}

function relative_path(filename, result) {
  result = filename
  sub("^" root, "", result)
  return result
}

function directory(path, result) {
  result = path
  if (result !~ /\//) {
    return ""
  }
  sub(/\/[^\/]*$/, "", result)
  return result
}

function normalize(base, path, joined, count, fields, i, depth, part,
                   result) {
  if (path ~ /^\//) {
    joined = substr(path, 2)
  } else if (base == "") {
    joined = path
  } else {
    joined = base "/" path
  }

  count = split(joined, fields, "/")
  depth = 0
  for (i = 1; i <= count; ++i) {
    part = fields[i]
    if (part == "" || part == ".") {
      continue
    }
    if (part == "..") {
      if (depth == 0) {
        return ""
      }
      --depth
      continue
    }
    normalized[++depth] = part
  }

  result = ""
  for (i = 1; i <= depth; ++i) {
    result = result (i == 1 ? "" : "/") normalized[i]
    delete normalized[i]
  }
  return result
}

function fence_width(line, marker, copy, width, spaces) {
  copy = line
  spaces = 0
  while (substr(copy, spaces + 1, 1) == " " && spaces < 4) {
    ++spaces
  }
  if (spaces > 3) {
    return 0
  }
  copy = substr(copy, spaces + 1)
  if (substr(copy, 1, 1) != marker) {
    return 0
  }
  width = 0
  while (substr(copy, width + 1, 1) == marker) {
    ++width
  }
  return width
}

function opens_fence(line, width) {
  width = fence_width(line, "`")
  if (width >= 3) {
    fence_marker = "`"
    fence_length = width
    return 1
  }
  width = fence_width(line, "~")
  if (width >= 3) {
    fence_marker = "~"
    fence_length = width
    return 1
  }
  return 0
}

function closes_fence(line, copy, width, suffix, spaces) {
  width = fence_width(line, fence_marker)
  if (width < fence_length) {
    return 0
  }
  copy = line
  spaces = 0
  while (substr(copy, spaces + 1, 1) == " " && spaces < 4) {
    ++spaces
  }
  copy = substr(copy, spaces + 1)
  suffix = substr(copy, width + 1)
  return suffix ~ /^[[:space:]]*$/
}

function without_inline_code(line, output, cursor, total, width, closing,
                             tail, found) {
  output = ""
  total = length(line)
  cursor = 1
  while (cursor <= total) {
    if (substr(line, cursor, 1) != "`") {
      output = output substr(line, cursor, 1)
      ++cursor
      continue
    }

    width = 1
    while (substr(line, cursor + width, 1) == "`") {
      ++width
    }
    tail = cursor + width
    found = 0
    while (tail <= total) {
      if (substr(line, tail, width) == substr(line, cursor, width) &&
          substr(line, tail + width, 1) != "`") {
        closing = tail + width
        found = 1
        break
      }
      ++tail
    }
    if (found) {
      cursor = closing
    } else {
      output = output substr(line, cursor, width)
      cursor += width
    }
  }
  return output
}

function heading_text(line, count, text) {
  count = 0
  while (substr(line, count + 1, 1) == "#" && count < 6) {
    ++count
  }
  if (count == 0 || substr(line, count + 1, 1) !~ /[[:space:]]/) {
    return ""
  }
  text = substr(line, count + 2)
  sub(/[[:space:]]+#+[[:space:]]*$/, "", text)
  return text
}

function slug(text, value) {
  value = tolower(text)
  gsub(/`/, "", value)
  gsub(/<[^>]*>/, "", value)
  gsub(/[^[:alnum:] _-]/, "", value)
  gsub(/[[:space:]]+/, "-", value)
  return value
}

function register_heading(document, text, base, key, duplicate, value) {
  base = slug(text)
  if (base == "") {
    return
  }
  key = document SUBSEP base
  duplicate = heading_count[key]++
  value = duplicate == 0 ? base : base "-" duplicate
  headings[document SUBSEP value] = 1
}

function is_external(target) {
  return target ~ /^[[:alpha:]][[:alnum:].+~-]*:/ || target ~ /^\/\//
}

function register_link(document, line, target, path, fragment, hash,
                       resolved) {
  if (target == "" || is_external(target)) {
    return
  }

  hash = index(target, "#")
  if (hash > 0) {
    path = substr(target, 1, hash - 1)
    fragment = substr(target, hash + 1)
  } else {
    path = target
    fragment = ""
  }

  if (path == "") {
    resolved = document
  } else {
    resolved = normalize(directory(document), path)
    if (resolved == "") {
      diagnostic(document, line, "local link escapes the repository: " target)
      return
    }
  }

  ++link_count
  link_document[link_count] = document
  link_line[link_count] = line
  link_target[link_count] = target
  link_path[link_count] = resolved
  link_fragment[link_count] = fragment
  if (fragment != "") {
    ++anchored_count
  }
}

function scan_links(document, line_number, line, remaining, matched, target) {
  remaining = without_inline_code(line)
  while (match(remaining, /!?\[[^][]*\]\([^()[:space:]]+\)/)) {
    matched = substr(remaining, RSTART, RLENGTH)
    target = matched
    sub(/^!?\[[^][]*\]\(/, "", target)
    sub(/\)$/, "", target)
    register_link(document, line_number, target)
    remaining = substr(remaining, RSTART + RLENGTH)
  }
}

FNR == 1 {
  document = relative_path(FILENAME)
  in_fence = 0
}

{
  if (in_fence) {
    if (closes_fence($0)) {
      in_fence = 0
    }
    next
  }
  if (opens_fence($0)) {
    in_fence = 1
    next
  }

  text = heading_text($0)
  if (text != "") {
    register_heading(document, text)
  }
  scan_links(document, FNR, $0)
}

END {
  for (i = 1; i <= link_count; ++i) {
    path = link_path[i]
    if (!(path in paths)) {
      diagnostic(link_document[i], link_line[i],
                 "local link target is unavailable: " link_target[i])
      continue
    }
    fragment = link_fragment[i]
    if (fragment != "" && !((path SUBSEP fragment) in headings)) {
      diagnostic(link_document[i], link_line[i],
                 "local link anchor is unavailable: " link_target[i])
    }
  }
  if (failed) {
    exit 1
  }
  print "docs links: " link_count " local, " anchored_count " anchored"
}
