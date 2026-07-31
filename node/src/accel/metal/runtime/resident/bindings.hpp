#pragma once

#include "../scoped/buffers.hpp"

#include "../../resident/model.hpp"
#include "../../state.hpp"
#include <array>
#include <cstddef>
#include <vector>

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
struct MetalResidentBindings {
  const rund::kernel::BindingSet *bindings = nullptr;
  std::array<MetalResidentBufferResult, kInlineMetalBufferCount> inputs{};
  std::vector<MetalResidentBufferResult> overflow_inputs{};
  std::array<MetalResidentBufferResult, kInlineMetalBufferCount> outputs{};
  std::vector<MetalResidentBufferResult> overflow_outputs{};

  [[nodiscard]] const MetalResidentBufferResult &
  input(const rund::kernel::u64 index) const {
    if (index < kInlineMetalBufferCount) {
      return inputs[static_cast<std::size_t>(index)];
    }
    return overflow_inputs[static_cast<std::size_t>(index -
                                                    kInlineMetalBufferCount)];
  }

  [[nodiscard]] const MetalResidentBufferResult &
  output(const rund::kernel::u64 index) const {
    if (index < kInlineMetalBufferCount) {
      return outputs[static_cast<std::size_t>(index)];
    }
    return overflow_outputs[static_cast<std::size_t>(index -
                                                     kInlineMetalBufferCount)];
  }
};
#endif

} // namespace rund::node::accel::detail
