#pragma once

#include "backing.hpp"
#include "route.hpp"

#include "src/compute/virtual/state.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace rund_node_test_persistent_product {

inline constexpr std::size_t ProductPageElements = 16u;
inline constexpr std::size_t ProductFrameCapacity = 2u;

struct PublicationSnapshot final {
  std::uint64_t generation{};
  std::uint64_t payload_epoch{};
};

struct PreparedProduct final {
  std::shared_ptr<rund::compute::detail::VirtualPipelineState> state{};
  std::shared_ptr<PersistentProductBacking> input{};
  std::shared_ptr<PersistentProductBacking> output{};
  std::vector<std::uint32_t> expected{};
};

} // namespace rund_node_test_persistent_product
