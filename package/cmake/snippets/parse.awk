BEGIN {
  compile_count = 0
  fragment_count = 0
  run_count = 0
  snippet_count = 0
  failed = 0
  in_cpp = 0
  in_other = 0
}

function diagnostic(message) {
  print relative ":" FNR ": " message > "/dev/stderr"
  failed = 1
}

function diagnostic_at(line, message) {
  print relative ":" line ": " message > "/dev/stderr"
  failed = 1
}

function begin_file() {
  if (NR > 1 && in_cpp) {
    diagnostic_at(opening_line, "unterminated C++ fence")
    close(generated)
  }
  relative = FILENAME
  sub("^" root, "", relative)
  in_cpp = 0
  in_other = 0
}

FNR == 1 {
  begin_file()
}

in_cpp {
  if ($0 ~ /^```[[:space:]]*$/) {
    close(generated)
    if (code_lines == 0) {
      diagnostic("compile-class C++ fence is empty")
    }
    mode = execute ? "run" : "compile"
    canonical_value = canonical == "" ? "-" : canonical
    print target "\t" mode "\t" relative "\t" opening_line "\t" \
        stable_source "\t" canonical_value >> manifest
    in_cpp = 0
    next
  }
  print $0 > generated
  code_lines++
  next
}

in_other {
  if ($0 ~ /^```[[:space:]]*$/) {
    in_other = 0
  }
  next
}

$0 ~ /^```(cpp|c\+\+)([[:space:]]|$)/ {
  opening_line = FNR
  line = $0
  sub(/[[:space:]]+$/, "", line)
  count = split(line, field, /[[:space:]]+/)
  if (field[1] != "```cpp") {
    diagnostic("C++ fences must use the canonical cpp language tag")
    in_other = 1
    next
  }
  if (count < 2 || (field[2] != "compile" && field[2] != "fragment")) {
    diagnostic("C++ fence must be classified as compile or fragment")
    in_other = 1
    next
  }
  if (field[2] == "fragment") {
    if (count != 2) {
      diagnostic("fragment C++ fence has unsupported metadata")
    }
    fragment_count++
    in_other = 1
    next
  }

  execute = 0
  canonical = ""
  metadata_failed = 0
  for (token_index = 3; token_index <= count; ++token_index) {
    if (field[token_index] == "run" && !execute) {
      execute = 1
    } else if (field[token_index] ~ /^source=/ && canonical == "") {
      canonical = substr(field[token_index], 8)
    } else {
      diagnostic("compile C++ fence has unsupported or duplicate metadata")
      metadata_failed = 1
    }
  }
  if (canonical != "" &&
      (canonical !~ /^package\/tests\/consumer\/example\/[A-Za-z0-9.\/-]+\.cpp$/ ||
       canonical ~ /(^|\/)\.\.?($|\/)/ || canonical ~ /\/\//)) {
    diagnostic("canonical snippet source must be a package consumer example")
    metadata_failed = 1
  }
  if (metadata_failed) {
    in_other = 1
    next
  }

  snippet_count++
  compile_count++
  if (execute) {
    run_count++
  }
  target = sprintf("rund-doc-%04d", snippet_count)
  generated = sprintf("%s/snippet-%04d.cpp", write_output, snippet_count)
  stable_source = sprintf("%s/snippet-%04d.cpp", source_output, snippet_count)
  code_lines = 0
  in_cpp = 1
  next
}

$0 ~ /^```/ {
  in_other = 1
  next
}

END {
  if (in_cpp) {
    diagnostic_at(opening_line, "unterminated C++ fence")
    close(generated)
  }
  if (failed) {
    exit 1
  }
  print "docs snippets: " compile_count " compile, " run_count \
      " run, " fragment_count " fragment"
}
