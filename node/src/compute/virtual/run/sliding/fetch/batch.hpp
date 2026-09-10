#pragma once

#include "../internal.hpp"
#include "reuse.hpp"

namespace rund::compute::detail::sliding_product_detail {

enum class FetchBatchIssue : std::uint8_t {
  Ready,
  Pending,
  PartialFailure,
};

struct FetchBatchEntry final {
  residency::execution::SlidingFetch fetch{};
  residency::execution::FetchSource source{};
  FetchBackingSlice slice{};
  std::byte *frame{};
};

struct FetchBatch final {
  std::array<FetchBatchEntry,
             residency::execution::WindowFootprintSourceCapacity>
      entries{};
  std::array<VirtualRead, residency::execution::WindowFootprintSourceCapacity>
      ranges{};
  std::size_t count{};
  std::size_t range_count{};
};

struct FetchBatchIo final {
  FetchIo io{};
  bool backing{};
  bool may_write{};
};

[[nodiscard]] FetchBatchIssue issue_fetch_batch(SlidingProductRun &,
                                                SlidingProductWork &,
                                                FetchBatch &) noexcept;
[[nodiscard]] FetchBatchIo read_fetch_batch(SlidingProductRun &,
                                            FetchBatch &) noexcept;
[[nodiscard]] Status retire_fetch_batch(SlidingProductRun &, FetchBatch &,
                                        Status, bool) noexcept;

} // namespace rund::compute::detail::sliding_product_detail
