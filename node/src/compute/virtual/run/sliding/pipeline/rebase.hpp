#pragma once

#include "../../../../status.hpp"

#include <cstdint>

namespace rund::compute::detail::sliding_product_detail {

struct SlidingProductRun;

enum class RebaseDisposition : std::uint8_t {
  Ready,
  RestoredFailure,
  Poisoned,
};

struct RebaseResult final {
  Status status{Status::fail(Reason::PipelinePoisoned)};
  RebaseDisposition disposition{RebaseDisposition::Poisoned};
};

[[nodiscard]] RebaseResult rebase_pipeline_sliding(SlidingProductRun &,
                                                   Status) noexcept;

} // namespace rund::compute::detail::sliding_product_detail
