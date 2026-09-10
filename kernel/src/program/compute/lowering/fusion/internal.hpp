#pragma once

#include <kernel/program/compute/lowering/fusion/build.hpp>

#include <vector>

namespace rund::kernel::compute_lowering_detail {

inline constexpr u32 kNoBinding = ~u32{0u};

struct FusedSource final {
  ParsedIR parsed{};
  u32 intermediate_read_binding = kNoBinding;
};

struct FusedBindingMap final {
  std::vector<ParsedBinding> bindings{};
  std::vector<u32> indices{};
  std::vector<u32> offsets{};
};

struct FusedNodeMap final {
  std::vector<ParsedNode> nodes{};
  bool ok = false;
};

struct BuiltFusedParsed final {
  ParsedIR parsed{};
  bool ok = false;
  const char *reason = "compute_fusion_invalid";
};

[[nodiscard]] u32 ReadNodeCount(const ParsedIR &, u32) noexcept;
[[nodiscard]] FusedBindingMap BuildFusedBindingMap(
    const std::vector<FusedSource> &, u32);
[[nodiscard]] FusedNodeMap BuildFusedNodeMap(
    const std::vector<FusedSource> &, const FusedBindingMap &, u32);
[[nodiscard]] BuiltFusedParsed BuildFusedParsed(
    const std::vector<FusedSource> &, ComputeScalar, ComputeDomain, u32, u32);
[[nodiscard]] ComputeIR BuildFusedIR(ParsedIR &, ComputeScalar, ComputeDomain);

} // namespace rund::kernel::compute_lowering_detail
