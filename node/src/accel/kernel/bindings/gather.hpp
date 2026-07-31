#pragma once

#include <kernel/program/compute/binding/model.hpp>

#include <memory>

namespace rund::node::accel::detail {

struct GatherBinds {
  const rund::kernel::ResidentBufferRef *values = nullptr;
  const std::shared_ptr<void> *values_handle = nullptr;
  const rund::kernel::ResidentBufferRef *indices = nullptr;
  const std::shared_ptr<void> *indices_handle = nullptr;
  const rund::kernel::ResidentBufferRef *logical_count = nullptr;
  const std::shared_ptr<void> *logical_count_handle = nullptr;
  const rund::kernel::ResidentBufferRef *output = nullptr;
  const std::shared_ptr<void> *output_handle = nullptr;
};

} // namespace rund::node::accel::detail
