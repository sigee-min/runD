#pragma once

#include <cstdint>
#include <vector>

#include "../model.hpp"

namespace rund::node {

enum class ReactorDrainBatchDisposition : std::uint8_t {
  Rejected,
  Failed,
  Complete,
};

class ReactorDrainBatch final {
public:
  [[nodiscard]] constexpr ReactorDrainBatchDisposition
  disposition() const noexcept {
    return disposition_;
  }

  [[nodiscard]] constexpr ReactorDrainBatch as_failed() const noexcept {
    return disposition_ == ReactorDrainBatchDisposition::Rejected
               ? *this
               : ReactorDrainBatch{ReactorDrainBatchDisposition::Failed, ready_,
                                   removed_waits_};
  }

  [[nodiscard]] constexpr const std::vector<ReactorReady> &
  ready() const noexcept {
    return *ready_;
  }

  [[nodiscard]] constexpr const std::vector<ReactorWait> &
  removed_waits() const noexcept {
    return *removed_waits_;
  }

private:
  friend ReactorDrainBatch
  ReactorBuildDrainBatch(ReactorRuntime &reactor,
                         const std::vector<ReactorReady> &ordered) noexcept;

  [[nodiscard]] static constexpr ReactorDrainBatch rejected() noexcept {
    return ReactorDrainBatch{ReactorDrainBatchDisposition::Rejected};
  }

  [[nodiscard]] static constexpr ReactorDrainBatch
  failed(const std::vector<ReactorReady> &ready,
         const std::vector<ReactorWait> &removed_waits) noexcept {
    return ReactorDrainBatch{ReactorDrainBatchDisposition::Failed, &ready,
                             &removed_waits};
  }

  [[nodiscard]] static constexpr ReactorDrainBatch
  complete(const std::vector<ReactorReady> &ready,
           const std::vector<ReactorWait> &removed_waits) noexcept {
    return ReactorDrainBatch{ReactorDrainBatchDisposition::Complete, &ready,
                             &removed_waits};
  }

  constexpr explicit ReactorDrainBatch(
      const ReactorDrainBatchDisposition disposition) noexcept
      : disposition_(disposition) {}

  constexpr ReactorDrainBatch(
      const ReactorDrainBatchDisposition disposition,
      const std::vector<ReactorReady> *const ready,
      const std::vector<ReactorWait> *const removed_waits) noexcept
      : disposition_(disposition), ready_(ready),
        removed_waits_(removed_waits) {}

  ReactorDrainBatchDisposition disposition_ =
      ReactorDrainBatchDisposition::Rejected;
  const std::vector<ReactorReady> *ready_ = nullptr;
  const std::vector<ReactorWait> *removed_waits_ = nullptr;
};

[[nodiscard]] ReactorDrainBatch
ReactorBuildDrainBatch(ReactorRuntime &reactor,
                       const std::vector<ReactorReady> &ordered) noexcept;

} // namespace rund::node
