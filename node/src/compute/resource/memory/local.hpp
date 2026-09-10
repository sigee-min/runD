#pragma once

#include "model.hpp"

#include <cstdint>
#include <span>

namespace rund::compute::detail::resource_detail::memory_detail {

[[nodiscard]] Status validate(graph::Info &info,
                              std::span<const MemoryNode> nodes,
                              std::uint64_t page_bytes, Work &work);

[[nodiscard]] Status collect(const graph::Info &info,
                             std::span<const MemoryNode> nodes, Work &work);

[[nodiscard]] Status classify(graph::Info &info,
                              std::span<const MemoryNode> nodes,
                              std::uint64_t page_bytes, Work &work);

[[nodiscard]] bool whole_value(const graph::Resource &value,
                               const graph::Access &access) noexcept;

[[nodiscard]] bool count_value(const graph::Resource &value) noexcept;

void identify_sources(graph::Info &info, std::span<const MemoryNode> nodes,
                      const Work &work) noexcept;

[[nodiscard]] Status finalize(graph::Info &info, std::uint64_t page_bytes,
                              Work &work);

} // namespace rund::compute::detail::resource_detail::memory_detail
