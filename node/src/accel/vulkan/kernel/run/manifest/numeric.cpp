#include "internal.hpp"

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

#include "../../../kernel.hpp"
#include "../../../numeric/source.hpp"
#include "../../../numeric/state.hpp"
#include "../../pipeline/source.hpp"

namespace rund::node::accel::detail {
namespace {

[[nodiscard]] bool VulkanNumericSourceBytes(const rund::kernel::NodeKind kind,
                                            const bool wide,
                                            std::uint64_t &bytes) noexcept {
  switch (kind) {
  case rund::kernel::NodeKind::Transform:
    return wide ? TransformSource64Bytes(bytes) : TransformSourceBytes(bytes);
  case rund::kernel::NodeKind::Matrix:
    return wide ? MatrixSource64Bytes(bytes) : MatrixSourceBytes(bytes);
  case rund::kernel::NodeKind::Factor:
    return wide ? FactorSource64Bytes(bytes) : FactorSourceBytes(bytes);
  case rund::kernel::NodeKind::Solve:
    return wide ? SolveSource64Bytes(bytes) : SolveSourceBytes(bytes);
  case rund::kernel::NodeKind::Spectrum:
    return wide ? SpectrumSource64Bytes(bytes) : SpectrumSourceBytes(bytes);
  default:
    return false;
  }
}

} // namespace

bool BuildVulkanNumericManifest(const KernelExecutionStep &step,
                                PreparedBackendManifest &manifest) noexcept {
  const std::uint64_t bindings =
      step.kind() == rund::kernel::NodeKind::Matrix
          ? 4u
          : (step.kind() == rund::kernel::NodeKind::Factor ||
                     step.kind() == rund::kernel::NodeKind::Spectrum
                 ? 5u
                 : 6u);
  manifest = PreparedBackendManifest{.source_build_count = 1u,
                                     .source_library_dependency_count = 1u,
                                     .pipeline_stage_count = 1u,
                                     .descriptor_set_count = 1u,
                                     .descriptor_binding_count = bindings,
                                     .descriptor_lease_count = 1u,
                                     .descriptor_dependency_count = 1u};
  bool wide = false;
  switch (step.kind()) {
  case rund::kernel::NodeKind::Transform:
    wide = step.operation.get<operation::Transform>().plan.element_bytes ==
           sizeof(rund::kernel::u64);
    break;
  case rund::kernel::NodeKind::Matrix:
    wide = step.operation.get<operation::Matrix>().plan.element_bytes ==
           sizeof(rund::kernel::u64);
    break;
  case rund::kernel::NodeKind::Factor:
    wide = step.operation.get<operation::Factor>().plan.element_bytes ==
           sizeof(rund::kernel::u64);
    break;
  case rund::kernel::NodeKind::Solve:
    wide = step.operation.get<operation::Solve>().plan.element_bytes ==
           sizeof(rund::kernel::u64);
    break;
  case rund::kernel::NodeKind::Spectrum:
    wide = step.operation.get<operation::Spectrum>().plan.element_bytes ==
           sizeof(rund::kernel::u64);
    break;
  default:
    return false;
  }
  std::uint64_t source_bytes = 0u;
  return VulkanNumericSourceBytes(step.kind(), wide, source_bytes) &&
         AddPreparedBackendCacheDependency(
             manifest,
             PreparedBackendCacheDependency{
                 .source_recipe = 0x76756c6b2e6e7500ull +
                                  static_cast<std::uint64_t>(step.kind()),
                 .source_upper_bytes = source_bytes,
                 .pipeline_stage_count = 1u,
             });
}

} // namespace rund::node::accel::detail

#endif
