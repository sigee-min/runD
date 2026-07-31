#include "contract/program/compute/reject/model.hpp"

#include <kernel/program/compute/lowering/entry.hpp>

#include <array>
#include <cstdio>

namespace program_compute_contract::rejection_support {

namespace {

inline constexpr std::array kApis{
    rund::kernel::ComputeApi::Cpu,
    rund::kernel::ComputeApi::Metal,
    rund::kernel::ComputeApi::Vulkan,
};

} // namespace

bool Accepts(const rund::kernel::ComputeIR &ir) {
  for (const auto api : kApis) {
    const auto artifact = rund::kernel::LowerComputeIR(ir, api);
    if (!artifact.ok || artifact.source_text.empty() ||
        artifact.canonical_ir_bytes != ir.canonical_bytes) {
      std::fprintf(stderr,
                   "lowering accept mismatch api=%u reason=%s ok=%d "
                   "source=%zu canonical=%zu expected=%zu\n",
                   static_cast<unsigned>(api), artifact.reason,
                   artifact.ok ? 1 : 0, artifact.source_text.size(),
                   artifact.canonical_ir_bytes.size(),
                   ir.canonical_bytes.size());
      return false;
    }
  }
  return true;
}

bool Rejects(const rund::kernel::ComputeIR &ir, const std::string_view reason) {
  for (const auto api : kApis) {
    const auto artifact = rund::kernel::LowerComputeIR(ir, api);
    if (artifact.ok || std::string_view{artifact.reason} != reason ||
        !artifact.source_text.empty()) {
      std::fprintf(stderr,
                   "lowering reject mismatch api=%u expected=%.*s actual=%s "
                   "ok=%d source=%zu\n",
                   static_cast<unsigned>(api), static_cast<int>(reason.size()),
                   reason.data(), artifact.reason, artifact.ok ? 1 : 0,
                   artifact.source_text.size());
      return false;
    }
  }
  return true;
}

} // namespace program_compute_contract::rejection_support
