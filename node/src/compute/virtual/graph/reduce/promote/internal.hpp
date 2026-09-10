#pragma once

#include "../../../run/cache.hpp"
#include "../projection.hpp"
#include "../promote.hpp"

#include <array>

namespace rund::compute::detail::graph_reduce::promote_detail {

struct Projection final {
  std::array<residency::execution::GraphPromotePage,
             residency::execution::GraphPromoteCapacity>
      promoted{};
  std::array<residency::PrefetchedPage,
             residency::execution::GraphPromoteCapacity>
      pages{};
  std::array<residency::execution::GraphPromoteCompletion,
             residency::execution::GraphPromoteCapacity>
      completions{};
  std::size_t page_count{};
};

struct UploadProjection final {
  std::array<UploadRequest, residency::execution::GraphPromoteCapacity>
      requests{};
  std::size_t request_count{};
  std::uint64_t bytes{};
};

[[nodiscard]] bool project(const VirtualRunProjection &, residency::EpochLease,
                           const Ticket &, Projection &) noexcept;

[[nodiscard]] bool project_upload(PipelineState &, const VirtualRunProjection &,
                                  const Projection &,
                                  UploadProjection &) noexcept;

[[nodiscard]] Status copy(PipelineState &, const VirtualRunProjection &,
                          residency::EpochLease, const Projection &, Stats &,
                          Interval &) noexcept;

[[nodiscard]] InputPromotionResult terminal(residency::Authority &, Ticket &,
                                            Projection &, Status,
                                            Interval) noexcept;

} // namespace rund::compute::detail::graph_reduce::promote_detail
