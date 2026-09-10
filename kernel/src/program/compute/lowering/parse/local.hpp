#pragma once

#include <kernel/program/compute/lowering/parse.hpp>

namespace rund::kernel::compute_lowering_detail {

[[nodiscard]] bool BindingIs(const ParsedIR &, u32 index, u8 kind) noexcept;
[[nodiscard]] bool DuplicateBindingName(const ParsedIR &,
                                        const ParsedBinding &);
[[nodiscard]] bool ValidNodeRef(u32 node, u32 current_node) noexcept;
[[nodiscard]] ParsedIR RejectParsed(const char *reason);

struct ParsedBindingRead final {
  ParsedBinding binding{};
  bool ok = false;
  const char *reason = "compute_ir_malformed";
};

[[nodiscard]] ParsedBindingRead ReadParsedBinding(Reader &);
[[nodiscard]] const char *ValidateParsedBinding(const ParsedIR &,
                                                const ParsedBinding &);
[[nodiscard]] const char *AppendParsedBinding(Reader &, ParsedIR &);

struct ParsedNodeRead final {
  ParsedNode node{};
  bool ok = false;
  const char *reason = "compute_ir_malformed";
};

[[nodiscard]] ParsedNodeRead ReadParsedNode(Reader &);
[[nodiscard]] const char *ValidateParsedNodeOperands(const ParsedIR &,
                                                     const ParsedNode &,
                                                     ComputeScalar,
                                                     u32 current_node) noexcept;
[[nodiscard]] const char *AppendParsedNode(Reader &, ParsedIR &, ComputeScalar,
                                           u32 index, u32 &write_count);
[[nodiscard]] ParsedIR ParseComputeIRUnchecked(const ComputeIR &,
                                               const std::vector<u8> &);

} // namespace rund::kernel::compute_lowering_detail
