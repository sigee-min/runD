#pragma once

#include "../view.hpp"

#include <cstdint>
#include <memory>
#include <vector>

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)

struct MetalAdapter;

// The projection owns these temporary replacements until the candidate view
// has been completely rebound.  They are deliberately not retained state.
struct MetalViewReplacement final {
  MetalResidentBufferResult resident{};
};

[[nodiscard]] std::shared_ptr<void>
AcquireMetalViewPipeline(MetalAdapter &adapter, const char *key,
                         const char *function);

[[nodiscard]] bool MetalViewRequiresLowering(const BoundStep &source) noexcept;
[[nodiscard]] bool ValidateMetalViewSource(const BoundStep &source) noexcept;
[[nodiscard]] bool
MetalViewReferenceNeedsLowering(const rund::kernel::ResidentBufferRef &ref,
                                bool &normalize_singleton) noexcept;

[[nodiscard]] MetalResidentBufferResult
ResolveMetalViewExternal(const rund::AccelDevice &pick,
                         const rund::kernel::ResidentBufferRef &ref,
                         const std::shared_ptr<void> &handle);

[[nodiscard]] MetalResidentBufferResult
ResolveMetalViewDense(const rund::AccelDevice &pick, std::uint64_t binding,
                      const rund::kernel::ResidentBufferRef &requested,
                      KernelPreparationMode mode, const KernelViewLayout *views,
                      const RunBinds *view_binds, bool &planned);

[[nodiscard]] rund::AccelCheck
BindMetalViewArguments(const RunBinds &original,
                       const std::vector<MetalViewReplacement> &replacements,
                       const std::vector<std::uint32_t> &replacement_by_binding,
                       MetalViewLowering &view);

[[nodiscard]] rund::AccelCheck
ProjectMetalView(const rund::AccelDevice &pick, const BoundStep &source,
                 KernelPreparationMode mode, const KernelViewLayout *views,
                 const RunBinds *view_binds,
                 std::shared_ptr<MetalViewLowering> &out);

#endif

} // namespace rund::node::accel::detail
