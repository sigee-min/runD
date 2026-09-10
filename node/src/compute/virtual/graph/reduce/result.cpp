#include "result.hpp"

namespace rund::compute::detail::graph_reduce {

VirtualGraphResult failed(const Status status, const std::uint64_t page,
                          const bool poison) noexcept {
  return VirtualGraphResult{
      .status = status, .failed_page = page, .poison_pipeline = poison};
}

} // namespace rund::compute::detail::graph_reduce
