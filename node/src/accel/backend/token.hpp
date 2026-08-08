#pragma once

#include "ops/table.hpp"

#include <accel/device.hpp>

#include <memory>
#include <utility>

namespace rund::node::accel::detail {

struct PickToken final {
  PickToken(rund::AccelDevice value, const BackendOps &operations)
      : raw(std::move(value)), ops(&operations) {}

  const rund::AccelDevice raw;
  const BackendOps *const ops;
};

[[nodiscard]] inline std::shared_ptr<void>
PublicPickOwner(const std::shared_ptr<PickToken> &token) noexcept {
  return std::static_pointer_cast<void>(token);
}

[[nodiscard]] rund::AccelDevice SealPick(rund::AccelDevice raw,
                                         const BackendOps &ops);

[[nodiscard]] std::shared_ptr<PickToken>
AdmitPick(const rund::AccelDevice &pick) noexcept;

} // namespace rund::node::accel::detail
