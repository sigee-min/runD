#pragma once

#include "../../persistent_product/backing.hpp"
#include "../../persistent_product/local.hpp"

#include "src/compute/virtual/state.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>

namespace rund_node_test_device_vsm_product::reduce_sum_test {

inline constexpr std::uint64_t PageElements = 16u;

struct PreparedReduce final {
  std::shared_ptr<rund::compute::detail::VirtualPipelineState> state{};
  std::shared_ptr<rund_node_test_persistent_product::PersistentProductBacking>
      input{};
  std::shared_ptr<rund_node_test_persistent_product::PersistentProductBacking>
      output{};
  std::uint64_t expected{};
  std::uint64_t element_count{};
  std::uint32_t element_bytes{};
};

[[nodiscard]] bool PrepareReduceProduct(rund::compute::Backend, std::uint64_t,
                                        std::uint32_t, bool, PreparedReduce &,
                                        bool &);
[[nodiscard]] bool SeedSafe(PreparedReduce &) noexcept;
[[nodiscard]] bool ExactReduceOutput(const PreparedReduce &) noexcept;
[[nodiscard]] bool RunReduceProductCases(
    rund::compute::Backend,
    rund_node_test_persistent_product::NativeQueueCounter) noexcept;

} // namespace rund_node_test_device_vsm_product::reduce_sum_test
