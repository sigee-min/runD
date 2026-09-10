#include "local.hpp"

#include "test/assert.hpp"

#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace replay_telemetry_contract {

namespace {

void CheckUserThrow() {
  Collector observer{};
  rund::Session session{};
  TEST_ASSERT(session.open(
      Config(rund::telemetry::bind(observer, rund::telemetry::Level::Basic),
             SessionId() + 1u)));

  std::size_t before = observer.count;
  const rund::replay::Live failed =
      rund::replay::live(session, [](rund::replay::Context &) {
        throw std::runtime_error{"telemetry replay callback"};
      });
  TEST_ASSERT(!failed);
  TEST_ASSERT(failed.code() == rund::replay::Code::RuntimeScopeCallbackFailed);
  const rund::telemetry::Event &failure = observer.one(before);
  TEST_ASSERT(failure.session == SessionId() + 1u);
  TEST_ASSERT(failure.replay.code == failed.code());
  TEST_ASSERT(failure.replay.mode == rund::telemetry::Mode::Live);
  TEST_ASSERT(failure.error() == failed.error());
  ExpectBasic(failure, rund::telemetry::Level::Basic);

  before = observer.count;
  const rund::replay::Live healthy =
      rund::replay::live(session, [](rund::replay::Context &) noexcept {});
  TEST_ASSERT(healthy);
  TEST_ASSERT(observer.one(before).replay.code == rund::replay::Code::Ok);
  TEST_ASSERT(session.close());
}

struct Reentry final {
  rund::Session *session = nullptr;
  std::uint32_t calls = 0u;
  std::uint32_t rejected = 0u;

  void operator()(const rund::telemetry::Event &) {
    ++calls;
    const rund::Session::Result nested = session->scope([] {});
    if (!nested && nested.code() == rund::ReasonCode::RuntimeReentryForbidden) {
      ++rejected;
    }
  }
};

void CheckReentry() {
  Reentry observer{};
  rund::Session session{};
  observer.session = &session;
  TEST_ASSERT(session.open(
      Config(rund::telemetry::bind(observer, rund::telemetry::Level::Basic),
             SessionId() + 2u)));
  TEST_ASSERT(
      rund::replay::live(session, [](rund::replay::Context &) noexcept {}));
  TEST_ASSERT(
      rund::replay::live(session, [](rund::replay::Context &) noexcept {}));
  TEST_ASSERT(observer.calls == 2u);
  TEST_ASSERT(observer.rejected == observer.calls);
  TEST_ASSERT(session.close());
}

struct Throwing final {
  std::uint32_t calls = 0u;

  void operator()(const rund::telemetry::Event &) {
    ++calls;
    throw std::runtime_error{"telemetry sink"};
  }
};

[[nodiscard]] bool SawSkipped(const rund::Trace &trace) noexcept {
  for (const rund::TraceRecord &record : trace.records) {
    if (record.event == rund::TraceEvent::TelemetrySkipped) {
      return record.code.runtime_code() ==
             rund::ReasonCode::TelemetrySinkFailed;
    }
  }
  return false;
}

void CheckSinkThrow() {
  Throwing observer{};
  rund::Session session{};
  TEST_ASSERT(session.open(
      Config(rund::telemetry::bind(observer, rund::telemetry::Level::Basic),
             SessionId() + 3u)));
  TEST_ASSERT(
      rund::replay::live(session, [](rund::replay::Context &) noexcept {}));
  TEST_ASSERT(
      rund::replay::live(session, [](rund::replay::Context &) noexcept {}));
  TEST_ASSERT(observer.calls == 2u);
  TEST_ASSERT(SawSkipped(session.trace()));
  TEST_ASSERT(session.close());
}

} // namespace

int CheckFailures() {
  CheckUserThrow();
  CheckReentry();
  CheckSinkThrow();
  return 0;
}

} // namespace replay_telemetry_contract
