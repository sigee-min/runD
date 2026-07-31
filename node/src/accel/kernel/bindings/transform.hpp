#pragma once

#include <kernel/program/compute/binding/model.hpp>

#include <memory>

namespace rund::node::accel::detail {

struct TransformBinds {
  const rund::kernel::ResidentBufferRef *input_real = nullptr;
  const std::shared_ptr<void> *input_real_handle = nullptr;
  const rund::kernel::ResidentBufferRef *input_imag = nullptr;
  const std::shared_ptr<void> *input_imag_handle = nullptr;
  const rund::kernel::ResidentBufferRef *output_real = nullptr;
  const std::shared_ptr<void> *output_real_handle = nullptr;
  const rund::kernel::ResidentBufferRef *output_imag = nullptr;
  const std::shared_ptr<void> *output_imag_handle = nullptr;
};

} // namespace rund::node::accel::detail
