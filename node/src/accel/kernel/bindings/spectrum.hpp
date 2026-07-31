#pragma once

#include <kernel/program/compute/binding/model.hpp>

#include <memory>

namespace rund::node::accel::detail {

struct SpectrumBinds {
  const rund::kernel::ResidentBufferRef *input = nullptr;
  const std::shared_ptr<void> *input_handle = nullptr;
  const rund::kernel::ResidentBufferRef *values = nullptr;
  const std::shared_ptr<void> *values_handle = nullptr;
  const rund::kernel::ResidentBufferRef *vectors = nullptr;
  const std::shared_ptr<void> *vectors_handle = nullptr;
  const rund::kernel::ResidentBufferRef *status = nullptr;
  const std::shared_ptr<void> *status_handle = nullptr;
};

} // namespace rund::node::accel::detail
