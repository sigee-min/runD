#pragma once

#include "../host/payload/store.hpp"
#include "../input/plan.hpp"
#include "index.hpp"

#include <rund/host/event.hpp>

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

namespace rund::replay::detail::scope {

enum class Mode : std::uint8_t {
  Live,
  Record,
  Replay,
  Scenario,
};

// Prepared replay evidence has one immutable owner. A strict or Scenario scope
// installs only a shared reference; validation and payload indexing are never
// repeated on the warm execution path.
class Expected final {
public:
  Expected(std::vector<::rund::host::Event> events,
           ::rund::node::replay_detail::payload::Store payloads)
      : events_(std::move(events)), payloads_(std::move(payloads)),
        index_(payloads_) {}

  Expected(const Expected &) = delete;
  Expected &operator=(const Expected &) = delete;
  Expected(Expected &&) = delete;
  Expected &operator=(Expected &&) = delete;

  [[nodiscard]] const std::vector<::rund::host::Event> &events() const noexcept {
    return events_;
  }

  [[nodiscard]] const ::rund::node::replay_detail::payload::Store &
  payloads() const noexcept {
    return payloads_;
  }

  [[nodiscard]] const Index &index() const noexcept { return index_; }

private:
  std::vector<::rund::host::Event> events_{};
  ::rund::node::replay_detail::payload::Store payloads_{};
  Index index_;
};

using ExpectedOwner = std::shared_ptr<const Expected>;

struct Plan final {
  Mode mode = Mode::Live;
  ExpectedOwner expected{};
  std::shared_ptr<const ::rund::node::replay_detail::InputPlan> choices{};

  [[nodiscard]] bool valid() const noexcept {
    switch (mode) {
    case Mode::Live:
    case Mode::Record:
      return expected == nullptr && choices == nullptr;
    case Mode::Replay:
      return expected != nullptr && choices == nullptr;
    case Mode::Scenario:
      return expected != nullptr && choices != nullptr;
    }
    return false;
  }
};

} // namespace rund::replay::detail::scope
