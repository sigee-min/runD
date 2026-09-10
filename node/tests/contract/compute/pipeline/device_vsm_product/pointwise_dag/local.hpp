#pragma once

#include "../../persistent_product/backing.hpp"
#include "../../persistent_product/local.hpp"

#include "src/compute/virtual/state.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace rund_node_test_device_vsm_product::pointwise_dag_test {

inline constexpr std::uint64_t PageElements = 16u;

enum class PointwiseDagType : std::uint8_t {
  U32,
  U64,
};

struct PreparedPointwiseDag final {
  std::shared_ptr<rund::compute::detail::VirtualPipelineState> state{};
  std::shared_ptr<rund_node_test_persistent_product::PersistentProductBacking>
      input{};
  std::shared_ptr<rund_node_test_persistent_product::PersistentProductBacking>
      output{};
  std::vector<std::byte> expected{};
  std::uint64_t element_count{};
  std::uint64_t element_bytes{};
  PointwiseDagType type{PointwiseDagType::U64};
};

[[nodiscard]] bool PreparePointwiseDagProduct(rund::compute::Backend,
                                              std::uint64_t, PointwiseDagType,
                                              PreparedPointwiseDag &, bool &);
[[nodiscard]] bool
ExactPointwiseDagOutput(const PreparedPointwiseDag &) noexcept;
[[nodiscard]] bool RunPointwiseDagProductCases(
    rund::compute::Backend,
    rund_node_test_persistent_product::NativeQueueCounter) noexcept;

} // namespace rund_node_test_device_vsm_product::pointwise_dag_test
