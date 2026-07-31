#pragma once

#include <kernel/program/compute/lowering/names.hpp>
#include <kernel/program/compute/lowering/text.hpp>
#include <string_view>
#include <utility>

namespace rund::kernel {
namespace compute_lowering_detail {

[[nodiscard]] inline std::string BindingSymbol(const ParsedBinding &binding) {
  std::string symbol;
  if (binding.kind == 1u) {
    symbol = "param_";
  } else if (binding.kind == 2u) {
    symbol = "read_";
  } else {
    symbol = "write_";
  }
  symbol += SafeIdentifier(binding.name);
  return symbol;
}

struct BindingLayout {
  std::string name_hex;
  std::string symbol;
  u64 param_offset = 0u;
  u32 buffer = 0u;
};

[[nodiscard]] inline std::string
BindingBaseSymbol(const std::string_view symbol) {
  return "RundBase_" + std::string{symbol};
}

[[nodiscard]] inline std::string
BindingBaseSymbol(const BindingLayout &layout) {
  return BindingBaseSymbol(layout.symbol);
}

[[nodiscard]] inline std::string
BindingStrideSymbol(const std::string_view symbol) {
  return "RundStride_" + std::string{symbol};
}

[[nodiscard]] inline std::string
BindingStrideSymbol(const BindingLayout &layout) {
  return BindingStrideSymbol(layout.symbol);
}

[[nodiscard]] inline std::vector<BindingLayout>
BuildBindingLayout(const ParsedIR &parsed) {
  u32 read_count = 0u;
  for (const ParsedBinding &binding : parsed.bindings) {
    if (binding.kind == 2u) {
      ++read_count;
    }
  }

  std::vector<BindingLayout> layouts;
  layouts.resize(parsed.bindings.size());
  u64 param_offset = 0u;
  u32 next_read_buffer = 1u;
  u32 next_write_buffer = 1u + read_count;
  for (std::size_t index = 0u; index < parsed.bindings.size(); ++index) {
    const ParsedBinding &binding = parsed.bindings[index];
    BindingLayout layout{
        .name_hex = HexText(binding.name),
        .symbol = BindingSymbol(binding),
    };
    if (binding.kind == 1u) {
      layout.param_offset = param_offset;
      param_offset += binding.element_bytes;
    } else if (binding.kind == 2u) {
      layout.buffer = next_read_buffer;
      ++next_read_buffer;
    } else {
      layout.buffer = next_write_buffer;
      ++next_write_buffer;
    }
    layouts[index] = std::move(layout);
  }
  return layouts;
}

} // namespace compute_lowering_detail

} // namespace rund::kernel
