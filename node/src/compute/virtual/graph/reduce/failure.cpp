#include "internal.hpp"

namespace rund::compute::detail::graph_reduce {

VirtualGraphResult finish_failure(GraphExecutionContext &context,
                                  const Status status, const std::uint64_t page,
                                  const bool poison, const Phase phase,
                                  const Check check, const std::uint32_t stage,
                                  const std::uint64_t batch) noexcept {
  context.state.failure_log.note(stage, batch, phase, check, status);
  return context.abort.finish(status, page, poison);
}

} // namespace rund::compute::detail::graph_reduce
