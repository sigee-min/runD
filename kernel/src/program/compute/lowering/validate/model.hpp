#pragma once

#include <kernel/program/compute/lowering/validate.hpp>

#include <vector>

namespace rund::kernel::compute_lowering_detail {

struct EffectiveNodeDomains final {
  std::vector<ComputeDomain> values;
  bool valid = true;
};

[[nodiscard]] const char *
ValidateFixedNodeFormat(const ParsedIR &parsed, const ParsedNode &node,
                        ComputeScalar scalar) noexcept;
[[nodiscard]] bool
WidthChangingWriteIsCanonicalMask(const ParsedIR &parsed,
                                  ComputeScalar scalar) noexcept;
[[nodiscard]] bool ReadAtIndexBinding(const ParsedIR &parsed,
                                      u32 binding) noexcept;
[[nodiscard]] const char *
ValidateWriteModes(const ParsedIR &parsed, ComputeScalar scalar,
                   const EffectiveNodeDomains &domains) noexcept;
[[nodiscard]] bool BindingDomainsMatchGraph(const ParsedIR &parsed,
                                            ComputeScalar scalar) noexcept;
[[nodiscard]] EffectiveNodeDomains
ResolveEffectiveNodeDomains(const ParsedIR &parsed, ComputeScalar scalar);

} // namespace rund::kernel::compute_lowering_detail
