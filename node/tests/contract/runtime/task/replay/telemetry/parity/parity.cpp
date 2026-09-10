#include "local.hpp"

#include "test/assert.hpp"

#include <cstddef>

namespace replay_telemetry_contract {

namespace {

[[nodiscard]] bool SameReplay(const rund::telemetry::Replay &left,
                              const rund::telemetry::Replay &right) noexcept {
  return left.code == right.code && left.mode == right.mode &&
         left.plan == right.plan && left.input_rows == right.input_rows &&
         left.input_bytes == right.input_bytes &&
         left.produced_rows == right.produced_rows &&
         left.choices == right.choices &&
         left.evidence_rows == right.evidence_rows &&
         left.evidence_bytes == right.evidence_bytes &&
         left.retained_bytes == right.retained_bytes &&
         left.copied_bytes == right.copied_bytes &&
         left.physical_bytes == right.physical_bytes &&
         left.allocated_bytes == right.allocated_bytes &&
         left.reserved_bytes == right.reserved_bytes &&
         left.storage_growths == right.storage_growths &&
         left.result_hash == right.result_hash;
}

[[nodiscard]] bool
SameNontimeFindings(const rund::telemetry::Event &left,
                    const rund::telemetry::Event &right) noexcept {
  const rund::telemetry::Findings left_findings = left.findings();
  const rund::telemetry::Findings right_findings = right.findings();
  std::size_t left_index = 0u;
  std::size_t right_index = 0u;
  while (true) {
    while (left_index != left_findings.size() &&
           left_findings[left_index].cost ==
               rund::telemetry::Cost::CriticalPath) {
      ++left_index;
    }
    while (right_index != right_findings.size() &&
           right_findings[right_index].cost ==
               rund::telemetry::Cost::CriticalPath) {
      ++right_index;
    }
    if (left_index == left_findings.size() ||
        right_index == right_findings.size()) {
      return left_index == left_findings.size() &&
             right_index == right_findings.size();
    }
    if (left_findings[left_index++] != right_findings[right_index++]) {
      return false;
    }
  }
}

void ExpectParity(const LevelRun &basic, const LevelRun &detail) {
  TEST_ASSERT(basic.identities == detail.identities);
  TEST_ASSERT(basic.producer_calls == detail.producer_calls);
  for (std::size_t index = 0u; index < basic.events.size(); ++index) {
    const rund::telemetry::Event &left = basic.events[index];
    const rund::telemetry::Event &right = detail.events[index];
    TEST_ASSERT(left.level == rund::telemetry::Level::Basic);
    TEST_ASSERT(right.level == rund::telemetry::Level::Detail);
    TEST_ASSERT(left.source == right.source);
    TEST_ASSERT(left.session == right.session);
    TEST_ASSERT(left.scope == right.scope);
    TEST_ASSERT(SameReplay(left.replay, right.replay));
    TEST_ASSERT(left.queue.depth == right.queue.depth);
    TEST_ASSERT(left.queue.capacity == right.queue.capacity);
    TEST_ASSERT(SameNontimeFindings(left, right));
    const rund::telemetry::Findings basic_findings = left.findings();
    const rund::telemetry::Findings detail_findings = right.findings();
    TEST_ASSERT(basic_findings.size() <= rund::telemetry::Findings::Capacity);
    TEST_ASSERT(detail_findings.size() <= rund::telemetry::Findings::Capacity);
    TEST_ASSERT(basic_findings[basic_findings.size() - 1u].accuracy ==
                rund::telemetry::Accuracy::Unavailable);
    TEST_ASSERT(detail_findings[detail_findings.size() - 1u].cost ==
                rund::telemetry::Cost::CriticalPath);
    TEST_ASSERT(left.error() == right.error());
    TEST_ASSERT(left.detail.prepare_ns == 0u);
    TEST_ASSERT(left.detail.work_ns == 0u);
    TEST_ASSERT(left.detail.finish_ns == 0u);
  }
}

} // namespace

int CheckParity() {
  const LevelRun basic = RunLevel(rund::telemetry::Level::Basic);
  const LevelRun detail = RunLevel(rund::telemetry::Level::Detail);
  ExpectParity(basic, detail);
  return 0;
}

} // namespace replay_telemetry_contract
