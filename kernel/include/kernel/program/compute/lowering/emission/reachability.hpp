#pragma once

#include <kernel/program/compute/lowering/model.hpp>

#include <cctype>
#include <string>
#include <string_view>
#include <vector>

namespace rund::kernel::compute_lowering_detail {

struct GeneratedHelperBlock {
  std::string name;
  std::size_t begin = 0u;
  std::size_t end = 0u;
};

[[nodiscard]] inline bool GeneratedIdentifierCharacter(const char value) {
  return std::isalnum(static_cast<unsigned char>(value)) != 0 || value == '_';
}

[[nodiscard]] inline bool
ContainsGeneratedIdentifier(const std::string_view text,
                            const std::string_view identifier) {
  std::size_t offset = text.find(identifier);
  while (offset != std::string_view::npos) {
    const bool left_boundary =
        offset == 0u || !GeneratedIdentifierCharacter(text[offset - 1u]);
    const std::size_t right = offset + identifier.size();
    const bool right_boundary =
        right == text.size() || !GeneratedIdentifierCharacter(text[right]);
    if (left_boundary && right_boundary) {
      return true;
    }
    offset = text.find(identifier, offset + 1u);
  }
  return false;
}

[[nodiscard]] inline std::string
GeneratedFunctionName(const std::string_view header) {
  const std::size_t open = header.rfind('(');
  if (open == std::string_view::npos) {
    return {};
  }
  std::size_t begin = open;
  while (begin != 0u && GeneratedIdentifierCharacter(header[begin - 1u])) {
    --begin;
  }
  if (begin == open) {
    return {};
  }
  return std::string{header.substr(begin, open - begin)};
}

[[nodiscard]] inline std::string
GeneratedStructName(const std::string_view header) {
  const std::size_t marker = header.rfind("struct ");
  if (marker == std::string_view::npos) {
    return {};
  }
  std::size_t begin = marker + std::string_view{"struct "}.size();
  while (begin < header.size() &&
         std::isspace(static_cast<unsigned char>(header[begin])) != 0) {
    ++begin;
  }
  std::size_t end = begin;
  while (end < header.size() && GeneratedIdentifierCharacter(header[end])) {
    ++end;
  }
  return begin == end ? std::string{}
                      : std::string{header.substr(begin, end - begin)};
}

[[nodiscard]] inline std::vector<GeneratedHelperBlock>
GeneratedHelperBlocks(const std::string &source) {
  std::vector<GeneratedHelperBlock> blocks;
  std::size_t previous_top_level_end = 0u;
  for (std::size_t open = 0u; open < source.size(); ++open) {
    if (source[open] != '{') {
      continue;
    }
    std::size_t depth = 1u;
    std::size_t close = open + 1u;
    for (; close < source.size() && depth != 0u; ++close) {
      if (source[close] == '{') {
        ++depth;
      } else if (source[close] == '}') {
        --depth;
      }
    }
    if (depth != 0u) {
      return {};
    }

    const std::string_view header{source.data() + previous_top_level_end,
                                  open - previous_top_level_end};
    std::string name = GeneratedFunctionName(header);
    if (name.empty()) {
      name = GeneratedStructName(header);
    }
    if (!name.empty()) {
      std::size_t begin = source.rfind('\n', open);
      begin = begin == std::string::npos ? 0u : begin + 1u;
      if (const std::size_t name_offset = source.rfind(name, open);
          name_offset != std::string::npos) {
        const std::size_t name_line = source.rfind('\n', name_offset);
        begin = name_line == std::string::npos ? 0u : name_line + 1u;
      }
      std::size_t end = close;
      if (end < source.size() && source[end] == ';') {
        ++end;
      }
      if (end < source.size() && source[end] == '\n') {
        ++end;
      }
      blocks.push_back(GeneratedHelperBlock{
          .name = std::move(name), .begin = begin, .end = end});
    }
    previous_top_level_end = close;
    open = close - 1u;
  }
  return blocks;
}

[[nodiscard]] inline std::string
KeepReachableGeneratedHelpers(const std::string &source,
                              const std::vector<std::string> &roots) {
  const std::vector<GeneratedHelperBlock> blocks =
      GeneratedHelperBlocks(source);
  if (blocks.empty() || roots.empty()) {
    return {};
  }

  std::vector<bool> reachable(blocks.size(), false);
  for (std::size_t index = 0u; index < blocks.size(); ++index) {
    for (const std::string &root : roots) {
      if (blocks[index].name == root) {
        reachable[index] = true;
        break;
      }
    }
  }

  bool changed = true;
  while (changed) {
    changed = false;
    for (std::size_t caller = 0u; caller < blocks.size(); ++caller) {
      if (!reachable[caller]) {
        continue;
      }
      const std::string_view body{source.data() + blocks[caller].begin,
                                  blocks[caller].end - blocks[caller].begin};
      for (std::size_t callee = 0u; callee < blocks.size(); ++callee) {
        if (!reachable[callee] &&
            ContainsGeneratedIdentifier(body, blocks[callee].name)) {
          reachable[callee] = true;
          changed = true;
        }
      }
    }
  }

  std::string out;
  std::size_t cursor = 0u;
  for (std::size_t index = 0u; index < blocks.size(); ++index) {
    if (cursor < blocks[index].begin) {
      out.append(source, cursor, blocks[index].begin - cursor);
    }
    if (reachable[index]) {
      out.append(source, blocks[index].begin,
                 blocks[index].end - blocks[index].begin);
    }
    cursor = blocks[index].end;
  }
  if (cursor < source.size()) {
    out.append(source, cursor, source.size() - cursor);
  }
  return out;
}

inline void AddGeneratedRoot(std::vector<std::string> &roots,
                             std::string root) {
  for (const std::string &existing : roots) {
    if (existing == root) {
      return;
    }
  }
  roots.push_back(std::move(root));
}

[[nodiscard]] inline std::vector<std::string>
CanonicalFixedHelperRoots(const ParsedIR &parsed, const ComputeScalar scalar) {
  std::vector<std::string> roots;
  const char *const width = scalar == ComputeScalar::Lane64 ? "64" : "32";
  const auto add = [&](const char *const prefix) {
    AddGeneratedRoot(roots, std::string{prefix} + width);
  };
  const auto add_lane = [&](const char *const prefix) {
    AddGeneratedRoot(roots, std::string{prefix} + "Lane" + width);
  };
  for (const ParsedNode &node : parsed.nodes) {
    switch (static_cast<IrOp>(node.op)) {
    case IrOp::AddSat:
      add("RundAddSat");
      break;
    case IrOp::AddSatUnsigned:
      add("RundAddSatUnsigned");
      break;
    case IrOp::SubSat:
      add("RundSubSat");
      break;
    case IrOp::NegPositiveFixed:
      add_lane("RundNegPositiveFixed");
      break;
    case IrOp::Sin:
      add("RundSin");
      break;
    case IrOp::Cos:
      add("RundCos");
      break;
    case IrOp::Tan:
      add("RundSin");
      add("RundCos");
      break;
    case IrOp::Exp:
      add("RundExp");
      break;
    case IrOp::Log:
      add("RundLog");
      break;
    case IrOp::Atan2:
      add("RundAtan2");
      break;
    default:
      break;
    }
  }
  return roots;
}

[[nodiscard]] inline bool ParsedIrHasOp(const ParsedIR &parsed,
                                        const IrOp op) noexcept {
  for (const ParsedNode &node : parsed.nodes) {
    if (node.op == static_cast<u8>(op)) {
      return true;
    }
  }
  return false;
}

} // namespace rund::kernel::compute_lowering_detail
