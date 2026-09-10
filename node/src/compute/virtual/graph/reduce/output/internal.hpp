#pragma once

#include "../output.hpp"

#include "../../../../pipeline/transfer.hpp"

#include <array>
#include <cstddef>
#include <cstdint>

namespace rund::compute::detail::graph_reduce::output_detail {

struct Projection final {
  std::array<PipelineFrameDownload, PipelineLeafCapacity> downloads{};
  std::array<residency::execution::GraphDrainCompletion,
             residency::execution::GraphDrainCapacity>
      completions{};
  std::array<std::uint64_t, residency::execution::GraphDrainCapacity>
      page_bytes{};
  std::uint64_t expected_bytes{};
  std::size_t page_count{};
};

struct Download final {
  Status status{Status::fail(Reason::PipelineInvalid)};
  Interval interval{};
  bool complete{};
};

[[nodiscard]] bool project(const VirtualRunProjection &, const Ticket &,
                           Projection &) noexcept;

[[nodiscard]] Download execute(PipelineState &, Projection &) noexcept;

[[nodiscard]] OutputDrainResult terminal(residency::Authority &, Ticket &,
                                         Projection &, Download) noexcept;

} // namespace rund::compute::detail::graph_reduce::output_detail
