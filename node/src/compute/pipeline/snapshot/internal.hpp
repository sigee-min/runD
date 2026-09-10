#pragma once

#include "../state.hpp"

#include <cstddef>
#include <cstdint>

namespace rund::compute::detail::snapshot_detail {

[[nodiscard]] std::uint64_t
hash(const StateSnapshotState &) noexcept;

[[nodiscard]] bool
valid_layout(const StateSnapshotState &) noexcept;

[[nodiscard]] Status
shape(const PipelinePublicationState &, std::size_t &bytes) noexcept;

[[nodiscard]] Status prepare_metadata(const PipelinePublicationState &,
                                      std::uint64_t generation,
                                      std::size_t byte_capacity,
                                      std::size_t field_capacity,
                                      StateSnapshotState &);

[[nodiscard]] Status capture_payload(PipelineState &, PipelinePublicationState &,
                                     StateSnapshotState &,
                                     std::uint64_t &transfer_count) noexcept;

[[nodiscard]] Status restore_locked(PipelineState &, PipelinePublicationState &,
                                     const StateSnapshotState &) noexcept;

} // namespace rund::compute::detail::snapshot_detail
