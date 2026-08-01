#pragma once

#include "../recurrence.hpp"

namespace rund::node::accel::detail {

[[nodiscard]] bool SamePlan(const rund::kernel::ComputePlan &left,
                            const rund::kernel::ComputePlan &right) noexcept;
[[nodiscard]] bool SameView(const rund::kernel::ResidentBindingRange &left,
                            std::uint64_t left_index,
                            const rund::kernel::ResidentBindingRange &right,
                            std::uint64_t right_index) noexcept;
[[nodiscard]] bool Aliases(const rund::kernel::ResidentBindingRange &left,
                           std::uint64_t left_index,
                           const rund::kernel::ResidentBindingRange &right,
                           std::uint64_t right_index) noexcept;
[[nodiscard]] bool SameWindows(const BoundStep &left,
                               const BoundStep &right) noexcept;
[[nodiscard]] bool
SameMapBinding(const rund::kernel::BindingSet &left,
               const rund::kernel::BindingSet &right) noexcept;
[[nodiscard]] bool
SameArtifact(const rund::kernel::LoweringArtifact &left,
             const rund::kernel::LoweringArtifact &right) noexcept;
[[nodiscard]] bool ExactRecurrenceMarker(
    std::span<const BackendBatchEntry> entries,
    bool &writes_each_iteration) noexcept;
[[nodiscard]] bool ExactNestedMapRecurrenceMarker(
    std::span<const BackendBatchEntry> entries) noexcept;
[[nodiscard]] bool ExactHistoryOutputs(
    std::span<const BackendBatchEntry> entries, std::uint64_t output_count,
    MapRecurrenceHistory &history);

} // namespace rund::node::accel::detail
