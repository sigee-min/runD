#pragma once

#include <kernel/program/compute/binding/model.hpp>

#include <memory>

namespace rund::node::accel::detail {

struct MatrixBinds {
  const rund::kernel::ResidentBufferRef *left = nullptr;
  const std::shared_ptr<void> *left_handle = nullptr;
  const rund::kernel::ResidentBufferRef *right = nullptr;
  const std::shared_ptr<void> *right_handle = nullptr;
  const rund::kernel::ResidentBufferRef *output = nullptr;
  const std::shared_ptr<void> *output_handle = nullptr;
};

} // namespace rund::node::accel::detail
