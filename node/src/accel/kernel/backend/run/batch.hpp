#pragma once

// Included by backend/run.hpp inside rund::node::accel::detail.

struct BackendBatchEntry final {
  const BackendRun *run = nullptr;
  const std::shared_ptr<void> *prepared = nullptr;
  rund::RuntimeStats *stats = nullptr;
  BackendRecurrence recurrence{};
  // Index into the common compiler's proved TileTransducer table. Backends
  // consume this classification; they do not rediscover it from native state.
  std::uint32_t transducer{std::numeric_limits<std::uint32_t>::max()};
  // Status/resource description is owned once per compact template. Native
  // capture may reference that template many times; occurrence_index is the
  // lexicographic command-order failure key.
  std::uint32_t template_index{};
  std::uint32_t occurrence_index{};

  [[nodiscard]] constexpr bool transduced_action() const noexcept {
    return transducer != std::numeric_limits<std::uint32_t>::max();
  }
};
