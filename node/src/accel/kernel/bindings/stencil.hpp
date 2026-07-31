#pragma once

#include <kernel/program/compute/binding/model.hpp>

#include <memory>

namespace rund::node::accel::detail {

struct StencilBinds {
  const rund::kernel::ResidentBufferRef *input = nullptr;
  const std::shared_ptr<void> *input_handle = nullptr;
  const rund::kernel::ResidentBufferRef *output = nullptr;
  const std::shared_ptr<void> *output_handle = nullptr;
};

} // namespace rund::node::accel::detail
