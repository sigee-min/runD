#pragma once

#include "../../../device/residency/execution/model.hpp"

#include <cstdint>

namespace rund::compute::detail {

struct VirtualPipelineState;
struct VirtualRunProjection;

struct VirtualFetchFill final {
  residency::execution::FetchFill policy{residency::execution::FetchFill::None};
  std::uint64_t element_bytes{};
  std::uint64_t value{};
};

// Projects the already-validated Virtual geometry into the exact byte recipe
// sealed by the Direct Plan. The backing service executes this recipe; it does
// not reinterpret Window operation, type, or boundary semantics.
[[nodiscard]] bool project_virtual_fetch_fill(const VirtualPipelineState &,
                                              const VirtualRunProjection &,
                                              VirtualFetchFill &) noexcept;

} // namespace rund::compute::detail
