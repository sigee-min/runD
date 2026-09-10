#pragma once

#include "model.hpp"
#include "stage.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>

namespace rund::compute::detail {
struct VirtualReduction;
}

namespace rund::compute::detail::graph_reduce {

struct OutputDrainResult final {
  Status status{Status::fail(Reason::PipelineInvalid)};
  Interval interval{};
  bool transfer_complete{};
  bool released{};
};

[[nodiscard]] Status issue_output_drain(VirtualPipelineState &,
                                        residency::Pool &,
                                        const VirtualRunProjection &,
                                        const residency::TiledGraphPlan &,
                                        std::size_t, Ticket &) noexcept;

[[nodiscard]] OutputDrainResult
transfer_output_drain(residency::Authority &, const VirtualRunProjection &,
                      Ticket &) noexcept;

[[nodiscard]] Status
finish_output(residency::Authority &, const VirtualRunProjection &,
              VirtualReduction &, Stats &, Wavefront &, StageController &,
              Ticket &, Timeline *hidden_by, bool &child_poison) noexcept;

} // namespace rund::compute::detail::graph_reduce
