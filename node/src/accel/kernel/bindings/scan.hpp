#pragma once

#include <kernel/program/compute/binding/model.hpp>

#include <memory>

namespace rund::node::accel::detail {

struct ScanBinds {
  const rund::kernel::ResidentBufferRef *input = nullptr;
  const std::shared_ptr<void> *input_handle = nullptr;
  const rund::kernel::ResidentBufferRef *output = nullptr;
  const std::shared_ptr<void> *output_handle = nullptr;
  const rund::kernel::ResidentBufferRef *logical_count = nullptr;
  const std::shared_ptr<void> *logical_count_handle = nullptr;
};

} // namespace rund::node::accel::detail
