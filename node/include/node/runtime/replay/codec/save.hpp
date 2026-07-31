#pragma once

#include <node/runtime/replay/model.hpp>
#include <rund/replay/code.hpp>

#include <cstddef>
#include <cstdint>
#include <span>

namespace rund::node::replay_detail::payload {
class Store;
}

namespace rund::node::replay_detail::artifact {

struct Sink final {
  void *state = nullptr;
  bool (*write)(void *, std::span<const std::byte>) noexcept = nullptr;
  std::uint64_t max_bytes = ~std::uint64_t{0u};

  [[nodiscard]] bool valid() const noexcept {
    return state != nullptr && write != nullptr;
  }
};

struct Result final {
  ::rund::replay::Code code = ::rund::replay::Code::ArtifactNotSaved;
  std::uint64_t bytes = 0u;
  std::uint64_t writes = 0u;

  [[nodiscard]] bool ok() const noexcept {
    return code == ::rund::replay::Code::Ok;
  }
};

[[nodiscard]] Result save(const RuntimeReplayRecord &record,
                          const payload::Store *payloads, Sink sink) noexcept;

} // namespace rund::node::replay_detail::artifact
