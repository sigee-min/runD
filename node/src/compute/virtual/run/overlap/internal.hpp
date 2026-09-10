#pragma once

#include "model.hpp"

#include "../epoch.hpp"

#include <array>
#include <cstdint>

namespace rund::compute::detail::virtual_run_overlap {

[[nodiscard]] Status prepare_cpu_epoch(
    VirtualPipelineState &, VirtualBacking &, const VirtualRunProjection &,
    std::uint64_t epoch, std::array<bool, 2u> &, Stats &, PreparedEpoch &,
    TimelineInterval &, TimelineInterval &, bool coherent_lookahead,
    bool &poison, VirtualInputReuseSeed *) noexcept;

[[nodiscard]] Status prepare_accel_epoch(
    VirtualPipelineState &, VirtualBacking &, const VirtualRunProjection &,
    std::uint64_t epoch, std::array<bool, 2u> &, Stats &, PreparedEpoch &,
    TimelineInterval &, TimelineInterval &, bool coherent_lookahead,
    bool &poison) noexcept;

[[nodiscard]] bool wait_prefetch_cpu(residency::Pool &,
                                     std::array<bool, 2u> &, bool publish,
                                     bool source_known = true) noexcept;
[[nodiscard]] bool wait_prefetch_accel(residency::Pool &,
                                       std::array<bool, 2u> &, bool publish,
                                       bool source_known = true) noexcept;

[[nodiscard]] Status submit_epoch(PreparedEpoch &) noexcept;
[[nodiscard]] bool complete_cpu_epoch(PreparedEpoch &, residency::Authority &,
                                       std::uint64_t cycle, bool success,
                                       bool invalidate_cycle = false,
                                       bool invalidate_direct = false) noexcept;
[[nodiscard]] bool complete_accel_epoch(
    PreparedEpoch &, residency::Authority &, std::uint64_t cycle, bool success,
    bool use_cycle, bool invalidate_cycle = false,
    bool invalidate_direct = false) noexcept;

[[nodiscard]] Status finish_cpu_epoch(
    PreparedEpoch &, const VirtualRunProjection &, Stats &, std::uint64_t cycle,
    bool &poison) noexcept;
[[nodiscard]] Status finish_accel_epoch(
    PreparedEpoch &, const VirtualRunProjection &, Stats &, std::uint64_t cycle,
    bool use_cycle, bool &poison) noexcept;

[[nodiscard]] Status flush_cpu_epoch(
    PreparedEpoch &, VirtualBacking &, const VirtualRunProjection &, Stats &,
    ::rund::node::hash_detail::Fnv &, VirtualReduction *, PageOutTimeline &,
    bool &poison) noexcept;
[[nodiscard]] Status flush_accel_epoch(
    PreparedEpoch &, VirtualBacking &, const VirtualRunProjection &, Stats &,
    ::rund::node::hash_detail::Fnv &, VirtualReduction *, PageOutTimeline &,
    bool &poison) noexcept;

} // namespace rund::compute::detail::virtual_run_overlap
