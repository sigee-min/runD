#pragma once

#include "../dispatch.hpp"

namespace rund::compute::detail::virtual_run_dispatch {

[[nodiscard]] VirtualRunWriteCertainty
epoch_certainty(bool poison_pipeline) noexcept;

[[nodiscard]] VirtualRunDispatchResult
dispatch_poolless(VirtualPipelineState &, std::span<VirtualBacking *const>,
                  VirtualBacking &, const VirtualRunProjection &,
                  const VirtualDeviceVsmCandidate &, std::uint64_t, Stats &,
                  VirtualRunResources &, bool &) noexcept;

[[nodiscard]] VirtualRunDispatchResult
dispatch_pooled(VirtualPipelineState &, std::span<VirtualBacking *const>,
                VirtualBacking &, VirtualBacking &, VirtualRunProjection &,
                const VirtualDeviceVsmCandidate &, std::uint64_t, Stats &,
                VirtualRunWork &, VirtualRunTransaction &,
                VirtualRunResources &, bool &) noexcept;

[[nodiscard]] VirtualRunDispatchResult
dispatch_graph(VirtualPipelineState &, std::span<VirtualBacking *const>,
               VirtualBacking &, const VirtualRunProjection &, Stats &,
               VirtualRunWork &) noexcept;

[[nodiscard]] VirtualRunDispatchResult
dispatch_accelerator(VirtualPipelineState &, VirtualBacking &, VirtualBacking &,
                     const VirtualRunProjection &, const VirtualRunAdmission &,
                     Stats &, bool &handled) noexcept;

[[nodiscard]] VirtualRunDispatchResult
dispatch_epochs(VirtualPipelineState &, VirtualBacking &, VirtualBacking &,
                const VirtualRunProjection &, Stats &, VirtualRunWork &,
                VirtualRunTransaction *) noexcept;

} // namespace rund::compute::detail::virtual_run_dispatch
