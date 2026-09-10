#pragma once

#include "../../persistent_product/backing.hpp"
#include "../../persistent_product/local.hpp"
#include "../route.hpp"

#include "src/compute/virtual/state.hpp"

#include <kernel/program/compute/reduce/model.hpp>

#include <cstdint>
#include <memory>

namespace rund_node_test_device_vsm_product::graph_multi_test {

inline constexpr std::uint64_t PageElements = 16u;

struct PreparedGraphMulti final {
  std::shared_ptr<rund::compute::detail::VirtualPipelineState> state{};
  std::shared_ptr<rund_node_test_persistent_product::PersistentProductBacking>
      first{};
  std::shared_ptr<rund_node_test_persistent_product::PersistentProductBacking>
      second{};
  std::shared_ptr<rund_node_test_persistent_product::PersistentProductBacking>
      output{};
  std::uint64_t expected{};
  std::uint64_t input_bytes{};
  rund::kernel::ReduceOp operation{rund::kernel::ReduceOp::Sum};
};

[[nodiscard]] bool PrepareGraphMulti(rund::compute::Backend, std::uint64_t,
                                     rund::kernel::ReduceOp,
                                     PreparedGraphMulti &, bool &);
[[nodiscard]] bool ExactGraphMultiOutput(const PreparedGraphMulti &) noexcept;
[[nodiscard]] bool RunGraphMultiProductCases(
    rund::compute::Backend,
    rund_node_test_persistent_product::NativeQueueCounter) noexcept;
[[nodiscard]] bool RunGraphMaximumInputProduct(
    rund::compute::Backend,
    rund_node_test_persistent_product::NativeQueueCounter) noexcept;

} // namespace rund_node_test_device_vsm_product::graph_multi_test
