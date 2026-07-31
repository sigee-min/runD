#pragma once

#include <rund/replay/record.hpp>

#include <node/runtime/replay/diff.hpp>
#include <node/runtime/replay/record.hpp>

#include "../scope/session.hpp"

#include <atomic>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

namespace rund::replay {

struct Record::Data {
  explicit Data(Session &session, Session::Result &&result,
                std::uint64_t start_hash);
  explicit Data(node::RuntimeReplayRecord &&value);

  node::RuntimeReplayRecord record{};
  std::vector<Capture> captures{};
  mutable detail::scope::ExpectedOwner prepared{};
  mutable std::atomic<const detail::scope::Expected *> published{nullptr};
  mutable std::mutex prepare_mutex{};
  Code facade_code = Code::Ok;
};

struct Diff::Data {
  explicit Data(node::RuntimeReplayDiff value) : diff(std::move(value)) {}

  node::RuntimeReplayDiff diff{};
};

struct Window::Data {
  explicit Data(node::RuntimeReplayMismatchWindow value);

  node::RuntimeReplayMismatchWindow window{};
  std::vector<InputPoint> expected_inputs{};
  std::vector<InputPoint> actual_inputs{};
  std::vector<Trace> expected_trace{};
  std::vector<Trace> actual_trace{};
};

} // namespace rund::replay
