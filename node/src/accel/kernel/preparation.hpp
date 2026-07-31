#pragma once

#include <cstdint>

namespace rund::node::accel::detail {

enum class KernelPreparationMode : std::uint8_t {
  Standalone,
  PipelinePrivate,
};

[[nodiscard]] constexpr bool
IsPipelinePrivatePreparation(const KernelPreparationMode mode) noexcept {
  return mode == KernelPreparationMode::PipelinePrivate;
}

inline thread_local KernelPreparationMode ActiveKernelPreparationMode =
    KernelPreparationMode::Standalone;

[[nodiscard]] inline KernelPreparationMode
CurrentKernelPreparationMode() noexcept {
  return ActiveKernelPreparationMode;
}

class KernelPreparationScope final {
public:
  explicit KernelPreparationScope(const KernelPreparationMode mode) noexcept
      : previous_{ActiveKernelPreparationMode} {
    ActiveKernelPreparationMode = mode;
  }

  KernelPreparationScope(const KernelPreparationScope &) = delete;
  KernelPreparationScope &operator=(const KernelPreparationScope &) = delete;

  ~KernelPreparationScope() { ActiveKernelPreparationMode = previous_; }

private:
  KernelPreparationMode previous_;
};

} // namespace rund::node::accel::detail
