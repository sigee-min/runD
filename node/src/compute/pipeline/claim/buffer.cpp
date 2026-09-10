#include "../claim.hpp"

#include <limits>
#include <mutex>

namespace rund::compute::detail {

Status acquire_claims(DeviceState &device,
                      const std::span<const BufferClaim> claims,
                      const bool reject_poison) noexcept {
  if (device.claims == nullptr) {
    return Status::fail(Reason::DeviceInvalid);
  }
  std::lock_guard lock{device.claims->gate};
  std::size_t acquired = 0u;
  const auto rollback = [&]() noexcept {
    for (std::size_t index = 0u; index < acquired; ++index) {
      const BufferClaim claim = claims[index];
      if (claim.write) {
        claim.buffer->writer = false;
      } else {
        --claim.buffer->readers;
      }
    }
  };
  for (const BufferClaim claim : claims) {
    Reason failure = Reason::Ok;
    if (claim.buffer == nullptr || claim.buffer->device.get() != &device) {
      failure = Reason::BindingDeviceMismatch;
    } else if (reject_poison && claim.buffer->poisoned) {
      failure = Reason::BufferPoisoned;
    } else if (claim.write ? claim.buffer->writer || claim.buffer->readers != 0u
                           : claim.buffer->writer) {
      failure = Reason::BufferBusy;
    } else if (!claim.write &&
               claim.buffer->readers ==
                   std::numeric_limits<
                       decltype(claim.buffer->readers)>::max()) {
      failure = Reason::BufferBusy;
    }
    if (failure != Reason::Ok) {
      rollback();
      return Status::fail(failure);
    }
    if (claim.write) {
      claim.buffer->writer = true;
    } else {
      ++claim.buffer->readers;
    }
    ++acquired;
  }
  return Status::success();
}

void release_claims(DeviceState &device,
                    const std::span<const BufferClaim> claims) noexcept {
  if (device.claims == nullptr) {
    return;
  }
  std::lock_guard lock{device.claims->gate};
  for (const BufferClaim claim : claims) {
    if (claim.buffer == nullptr) {
      continue;
    }
    if (claim.write) {
      claim.buffer->writer = false;
    } else if (claim.buffer->readers != 0u) {
      --claim.buffer->readers;
    }
  }
}

void publish_claims(DeviceState &device,
                    const std::span<const BufferClaim> claims,
                    const bool succeeded, const bool poison_writes) noexcept {
  if (device.claims == nullptr) {
    return;
  }
  std::lock_guard lock{device.claims->gate};
  for (const BufferClaim claim : claims) {
    if (claim.buffer == nullptr) {
      continue;
    }
    if (claim.write) {
      if (succeeded) {
        ++claim.buffer->generation;
      } else if (poison_writes) {
        // Failure publication is ordered before writer release.
        claim.buffer->poisoned = true;
      }
      claim.buffer->writer = false;
    } else if (claim.buffer->readers != 0u) {
      --claim.buffer->readers;
    }
  }
}

bool buffer_poisoned(const BufferState &buffer) noexcept {
  if (buffer.device == nullptr || buffer.device->claims == nullptr) {
    return true;
  }
  std::lock_guard lock{buffer.device->claims->gate};
  return buffer.poisoned;
}

} // namespace rund::compute::detail
