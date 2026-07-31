#include "test/assert.hpp"

#include <cluster/retry/identity.hpp>
#include <rund/evidence.hpp>

#include <string_view>
#include <type_traits>

static_assert(std::is_same_v<decltype(rund::cluster::RetryDecision{}.code),
                             rund::cluster::RetryCode>);

namespace {

rund::cluster::RunKey MakeCompleteKey() {
  return rund::cluster::RunKey{
      .work = rund::cluster::WorkId{1u},
      .sharded = true,
      .shard = rund::cluster::ShardRef{rund::cluster::JobId{2u},
                                       rund::cluster::ShardId{3u}},
      .input = rund::cluster::InputVersion{4u},
      .time = rund::cluster::LogicalTime{5u},
      .checkpoint = rund::cluster::CheckpointId{6u},
      .program = rund::cluster::ProgramId{7u},
      .fold = rund::cluster::FoldId{8u},
      .numeric = rund::evidence::identify(rund::evidence::i32()),
      .capacity = rund::cluster::CapacityId{9u},
      .output = rund::cluster::OutputId{10u},
  };
}

} // namespace

int RunClusterRetryContract() {
  constexpr rund::cluster::RetryDecision pending{};
  static_assert(!pending.preserves_identity());
  static_assert(pending.code == rund::cluster::RetryCode::NotEvaluated);
  static_assert(pending.reason() == "not_evaluated");

  const rund::cluster::RunKey key = MakeCompleteKey();
  const rund::cluster::RetryDecision preserved = rund::cluster::evaluate_retry(
      rund::cluster::RetryRequest{key, key, rund::cluster::RetryEpoch{1u}});
  TEST_ASSERT(preserved.preserves_identity());
  TEST_ASSERT(preserved.code == rund::cluster::RetryCode::IdentityPreserved);
  TEST_ASSERT(preserved.reason() == "run_identity_preserved");
  TEST_ASSERT(preserved.attempt.run == key);
  TEST_ASSERT(preserved.attempt.retry == rund::cluster::RetryEpoch{1u});

  rund::cluster::RunKey changed = key;
  changed.input = rund::cluster::InputVersion{11u};
  const rund::cluster::RetryDecision rejected = rund::cluster::evaluate_retry(
      rund::cluster::RetryRequest{key, changed, rund::cluster::RetryEpoch{2u}});
  TEST_ASSERT(!rejected.preserves_identity());
  TEST_ASSERT(rejected.code == rund::cluster::RetryCode::IdentityChanged);
  TEST_ASSERT(rejected.reason() == "run_identity_changed");

  rund::cluster::RunKey numeric_changed = key;
  numeric_changed.numeric = rund::evidence::identify(rund::evidence::i64());
  const rund::cluster::RetryDecision strict_rejected =
      rund::cluster::evaluate_retry(rund::cluster::RetryRequest{
          key, numeric_changed, rund::cluster::RetryEpoch{5u}});
  TEST_ASSERT(!strict_rejected.preserves_identity());
  TEST_ASSERT(strict_rejected.reason() == "run_identity_changed");

  rund::cluster::RunKey checkpoint_changed = key;
  checkpoint_changed.checkpoint = rund::cluster::CheckpointId{13u};
  const rund::cluster::RetryDecision checkpoint_rejected =
      rund::cluster::evaluate_retry(rund::cluster::RetryRequest{
          key, checkpoint_changed, rund::cluster::RetryEpoch{6u}});
  TEST_ASSERT(!checkpoint_rejected.preserves_identity());
  TEST_ASSERT(checkpoint_rejected.reason() == "run_identity_changed");

  rund::cluster::RunKey incomplete = key;
  incomplete.capacity = rund::cluster::CapacityId{};
  const rund::cluster::RetryDecision incomplete_decision =
      rund::cluster::evaluate_retry(rund::cluster::RetryRequest{
          key, incomplete, rund::cluster::RetryEpoch{3u}});
  TEST_ASSERT(!incomplete_decision.preserves_identity());
  TEST_ASSERT(incomplete_decision.code ==
              rund::cluster::RetryCode::KeyIncomplete);
  TEST_ASSERT(incomplete_decision.reason() == "run_key_incomplete");

  rund::cluster::RunKey unsharded_a = key;
  unsharded_a.sharded = false;
  unsharded_a.shard = rund::cluster::ShardRef{rund::cluster::JobId{101u},
                                              rund::cluster::ShardId{202u}};
  rund::cluster::RunKey unsharded_b = key;
  unsharded_b.sharded = false;
  unsharded_b.shard = rund::cluster::ShardRef{rund::cluster::JobId{303u},
                                              rund::cluster::ShardId{404u}};
  const rund::cluster::RetryDecision unsharded_preserved =
      rund::cluster::evaluate_retry(rund::cluster::RetryRequest{
          unsharded_a, unsharded_b, rund::cluster::RetryEpoch{4u}});
  TEST_ASSERT(unsharded_preserved.preserves_identity());
  TEST_ASSERT(unsharded_preserved.reason() == "run_identity_preserved");
  return 0;
}
