#include "memory/local.hpp"

#include <new>

namespace rund::compute::detail::resource_detail {

Status plan_memory(graph::Info &info, const std::span<const MemoryNode> nodes,
                   const std::uint64_t page_bytes) noexcept {
  try {
    memory_detail::Work work;
    if (const Status status =
            memory_detail::validate(info, nodes, page_bytes, work);
        !status) {
      return status;
    }
    if (const Status status = memory_detail::collect(info, nodes, work);
        !status) {
      return status;
    }
    if (const Status status =
            memory_detail::classify(info, nodes, page_bytes, work);
        !status) {
      return status;
    }
    memory_detail::identify_sources(info, nodes, work);
    return memory_detail::finalize(info, page_bytes, work);
  } catch (const std::bad_alloc &) {
    return Status::fail(Reason::GraphCapacity);
  }
}

} // namespace rund::compute::detail::resource_detail
