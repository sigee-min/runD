#pragma once

#include "model.hpp"

namespace rund::compute::detail::graph_reduce {

struct InputPromotionResult final {
  Status status{Status::fail(Reason::PipelineInvalid)};
  Interval interval{};
  bool transfer_complete{};
  bool released{};
};

[[nodiscard]] Status issue_input_promotion(
    residency::Authority &, const residency::TiledGraphInvocation &, Ticket &,
    std::size_t stage, std::uint64_t destination_token) noexcept;

[[nodiscard]] InputPromotionResult
transfer_input_promotion(residency::Authority &, PipelineState &,
                         const VirtualRunProjection &, residency::EpochLease,
                         Stats &, Ticket &) noexcept;

[[nodiscard]] bool cancel_input_promotion(residency::Authority &,
                                          Ticket &) noexcept;

} // namespace rund::compute::detail::graph_reduce
