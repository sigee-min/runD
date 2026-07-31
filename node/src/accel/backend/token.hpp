#pragma once

#include "ops/table.hpp"

#include <accel/check.hpp>
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

struct PickAdmission final {
  rund::AccelCheck check{false, "compute_adapter_invalid"};
  std::shared_ptr<PickToken> token{};

  [[nodiscard]] const rund::AccelDevice *raw() const noexcept {
    return token == nullptr ? nullptr : &token->raw;
  }

  [[nodiscard]] const BackendOps *ops() const noexcept {
    return token == nullptr ? nullptr : token->ops;
  }
};

[[nodiscard]] inline std::shared_ptr<void>
PublicPickOwner(const std::shared_ptr<PickToken> &token) noexcept {
  return std::static_pointer_cast<void>(token);
}

[[nodiscard]] rund::AccelDevice SealPick(rund::AccelDevice raw,
                                         const BackendOps &ops);

[[nodiscard]] PickAdmission AdmitPick(const rund::AccelDevice &pick) noexcept;

} // namespace rund::node::accel::detail
