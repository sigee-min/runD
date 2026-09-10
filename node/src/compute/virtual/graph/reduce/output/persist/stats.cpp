#include "../persist.hpp"

#include <rund/counter.hpp>

namespace rund::compute::detail::graph_reduce::output_persist_detail {

void record(Stats &stats, const std::uint64_t io_ns, const std::uint64_t bytes,
            const std::size_t completed_pages) noexcept {
  ::rund::detail::counter::Accumulate(stats.pipeline.residency.backing_io_ns,
                                      io_ns);
  ::rund::detail::counter::Accumulate(
      stats.pipeline.residency.backing_write_bytes, bytes);
  ::rund::detail::counter::Accumulate(stats.pipeline.residency.page_out_bytes,
                                      bytes);
  ::rund::detail::counter::Accumulate(stats.pipeline.residency.page_out_count,
                                      completed_pages);
}

} // namespace rund::compute::detail::graph_reduce::output_persist_detail
