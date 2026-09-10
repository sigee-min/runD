#pragma once

#include "../../persistent_product/backing.hpp"
#include "../../persistent_product/local.hpp"
#include "../route.hpp"

#include "src/compute/virtual/state.hpp"

#include <kernel/program/compute/reduce/model.hpp>

#include <cstdint>
#include <memory>

namespace rund_node_test_device_vsm_product::graph_test {

inline constexpr std::uint64_t PageElements = 16u;

struct PreparedGraph final {
  std::shared_ptr<rund::compute::detail::VirtualPipelineState> state{};
  std::shared_ptr<rund_node_test_persistent_product::PersistentProductBacking>
      input{};
  std::shared_ptr<rund_node_test_persistent_product::PersistentProductBacking>
      output{};
  std::uint64_t expected{};
  std::uint64_t input_bytes{};
  std::uint64_t authored_nodes{};
  std::uint64_t lowered_nodes{};
  std::uint32_t stage_count{};
  rund::kernel::ReduceOp operation{rund::kernel::ReduceOp::Sum};
};

[[nodiscard]] bool PrepareGraphProduct(rund::compute::Backend, std::uint64_t,
                                       PreparedGraph &, bool &);
[[nodiscard]] bool PrepareFusedGraphProduct(rund::compute::Backend,
                                            std::uint64_t, PreparedGraph &,
                                            bool &);
[[nodiscard]] bool PrepareTypedGraphProduct(rund::compute::Backend,
                                            std::uint64_t, PreparedGraph &,
                                            bool &);
[[nodiscard]] bool PrepareFusedTypedGraphProduct(rund::compute::Backend,
                                                 std::uint64_t, PreparedGraph &,
                                                 bool &);
[[nodiscard]] bool PrepareBranchGraphProduct(rund::compute::Backend,
                                             std::uint64_t, PreparedGraph &,
                                             bool &);
[[nodiscard]] bool PrepareGraphOverflow(rund::compute::Backend, std::uint64_t,
                                        PreparedGraph &, bool &);
[[nodiscard]] bool PrepareGraphCountNonzero(rund::compute::Backend,
                                            std::uint64_t, PreparedGraph &,
                                            bool &);
[[nodiscard]] bool PrepareGraphMin(rund::compute::Backend, std::uint64_t,
                                   PreparedGraph &, bool &);
[[nodiscard]] bool PrepareGraphMax(rund::compute::Backend, std::uint64_t,
                                   PreparedGraph &, bool &);
[[nodiscard]] bool ExactGraphOutput(const PreparedGraph &) noexcept;
[[nodiscard]] bool CheckFusedGraphSemantic() noexcept;
[[nodiscard]] bool RunGraphProductCases(
    rund::compute::Backend,
    rund_node_test_persistent_product::NativeQueueCounter) noexcept;

} // namespace rund_node_test_device_vsm_product::graph_test
