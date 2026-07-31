#pragma once

#include <node/runtime/replay.hpp>
#include <rund/session.hpp>
#include <rund/replay.hpp>
#include <rund/task/await.hpp>

#include <chrono>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

struct RuntimeReplayFixture {
  std::uint64_t value = 0u;
  rund::SessionConfig spec{};
  rund::node::RuntimeReplayRecord record{};
  std::vector<std::byte> encoded{};
};

[[nodiscard]] inline std::vector<std::byte>
SaveReplayRecord(const rund::node::RuntimeReplayRecord &record) {
  std::vector<std::byte> artifact{};
  const rund::node::replay_detail::artifact::Result saved =
      rund::node::replay_detail::artifact::save(
          record, nullptr,
          rund::node::replay_detail::artifact::Sink{
              .state = &artifact,
              .write = [](void *const state,
                          const std::span<const std::byte> bytes) noexcept {
                try {
                  auto &output = *static_cast<std::vector<std::byte> *>(state);
                  output.insert(output.end(), bytes.begin(), bytes.end());
                  return true;
                } catch (...) {
                  return false;
                }
              }});
  if (!saved.ok()) {
    throw std::runtime_error{
        std::string{rund::replay::error(saved.code)}};
  }
  return artifact;
}

[[nodiscard]] inline std::span<const std::byte>
ReplayArtifact(const std::vector<std::byte> &bytes) noexcept {
  return bytes;
}

template <typename T>
[[nodiscard]] std::vector<std::byte> SaveReplayArtifact(const T &value) {
  std::vector<std::byte> artifact{};
  const rund::replay::Save saved =
      value.save([&artifact](const std::span<const std::byte> bytes) noexcept {
        try {
          artifact.insert(artifact.end(), bytes.begin(), bytes.end());
          return true;
        } catch (...) {
          return false;
        }
      });
  if (!saved) {
    throw std::runtime_error{std::string{saved.error()}};
  }
  return artifact;
}

[[nodiscard]] inline rund::node::RuntimeReplayRecord
CloneReplayRecord(const rund::node::RuntimeReplayRecord &record) {
  rund::node::RuntimeReplayDecodeResult decoded =
      rund::node::DecodeRuntimeReplayRecord(SaveReplayRecord(record));
  return decoded.ok() ? std::move(decoded.record)
                      : rund::node::RuntimeReplayRecord{};
}

inline rund::task::Task<void> ReplaySleep(
    const std::chrono::nanoseconds delay = std::chrono::nanoseconds{1}) {
  (void)co_await rund::task::sleep(delay);
}

template <class Fn>
rund::task::Task<void> ReplaySleepThen(const std::chrono::nanoseconds delay,
                                       Fn after) {
  (void)co_await rund::task::sleep(delay);
  after();
}

int MakeRuntimeReplayFixture(RuntimeReplayFixture &fixture);
int CheckReplayRecordContract(const RuntimeReplayFixture &fixture);
int CheckReplayDecodeContract(const RuntimeReplayFixture &fixture);
int CheckReplayDiffContract(const RuntimeReplayFixture &fixture);
int CheckReplayFailureContract();
int CheckReplayRunContract(const RuntimeReplayFixture &fixture);
int RunRuntimeTaskReplayEvidenceContract();
int RunRuntimeTaskReplayPayloadCodecContract();
int RunRuntimeTaskReplayPayloadContract();
int RunRuntimeTaskReplayPayloadStoreContract();
