#include "test/assert.hpp"

#include <cluster/run/identity.hpp>
#include <rund/evidence.hpp>

static_assert(rund::evidence::valid(rund::evidence::i32()));
static_assert(rund::evidence::valid(rund::evidence::i64()));
static_assert(rund::evidence::valid(rund::evidence::fixed<16, 16>()));
static_assert(rund::evidence::valid(rund::evidence::fixed<32, 32>()));
static_assert(rund::evidence::valid(rund::evidence::strict_f32()));
static_assert(rund::evidence::valid(rund::evidence::strict_f64()));
static_assert(rund::evidence::valid(rund::evidence::diagnostic_f32()));
static_assert(rund::evidence::valid(rund::evidence::diagnostic_f64()));
static_assert(rund::evidence::valid(rund::evidence::presentation_f32()));
static_assert(rund::evidence::valid(rund::evidence::presentation_f64()));
static_assert(!rund::evidence::valid(rund::evidence::Contract{
    .domain = rund::evidence::Domain::F64,
    .arithmetic = rund::evidence::Arithmetic::FloatingPoint,
    .authority = rund::evidence::Authority::Authoritative,
    .determinism = rund::evidence::Determinism::Required,
}));
static_assert(!rund::evidence::valid(rund::evidence::Contract{
    .domain = rund::evidence::Domain::F64,
    .arithmetic = rund::evidence::Arithmetic::StrictFloatingPoint,
    .authority = rund::evidence::Authority::Authoritative,
    .determinism = rund::evidence::Determinism::BestEffort,
}));
static_assert(!rund::evidence::valid(rund::evidence::Contract{
    .domain = rund::evidence::Domain::F64,
    .arithmetic = rund::evidence::Arithmetic::FixedPoint,
    .authority = rund::evidence::Authority::Authoritative,
    .determinism = rund::evidence::Determinism::Required,
}));
static_assert(!rund::evidence::valid(rund::evidence::Contract{
    .domain = rund::evidence::Domain::Fixed,
    .arithmetic = rund::evidence::Arithmetic::HashDigest,
    .authority = rund::evidence::Authority::Diagnostic,
    .determinism = rund::evidence::Determinism::Required,
}));
static_assert(!rund::evidence::valid(rund::evidence::Contract{
    .domain = rund::evidence::Domain::Fixed,
    .arithmetic = rund::evidence::Arithmetic::StrictFloatingPoint,
    .authority = rund::evidence::Authority::Authoritative,
    .determinism = rund::evidence::Determinism::Required,
}));
static_assert(rund::evidence::identify(rund::evidence::i32()) ==
              rund::evidence::identify(rund::evidence::i32()));
static_assert(rund::evidence::identify(rund::evidence::i32()) !=
              rund::evidence::identify(rund::evidence::i64()));

namespace {

rund::cluster::RunKey MakeCompleteKey() {
  return rund::cluster::RunKey{
      .work = rund::cluster::WorkId{10u},
      .sharded = true,
      .shard = rund::cluster::ShardRef{rund::cluster::JobId{20u},
                                       rund::cluster::ShardId{30u}},
      .input = rund::cluster::InputVersion{40u},
      .time = rund::cluster::LogicalTime{50u},
      .checkpoint = rund::cluster::CheckpointId{60u},
      .program = rund::cluster::ProgramId{70u},
      .fold = rund::cluster::FoldId{80u},
      .numeric = rund::evidence::identify(rund::evidence::i32()),
      .capacity = rund::cluster::CapacityId{90u},
      .output = rund::cluster::OutputId{100u},
  };
}

} // namespace

int RunClusterRunContract() {
  rund::cluster::RunKey key = MakeCompleteKey();
  TEST_ASSERT(key.complete());

  rund::cluster::RunAttempt first{key, rund::cluster::RetryEpoch{0u}};
  rund::cluster::RunAttempt retry{key, rund::cluster::RetryEpoch{1u}};
  TEST_ASSERT(first.run == retry.run);
  TEST_ASSERT(first != retry);

  key.output = rund::cluster::OutputId{};
  TEST_ASSERT(!key.complete());

  key = MakeCompleteKey();
  key.capacity = rund::cluster::CapacityId{};
  TEST_ASSERT(!key.complete());

  key = MakeCompleteKey();
  key.fold = rund::cluster::FoldId{};
  TEST_ASSERT(!key.complete());

  key = MakeCompleteKey();
  key.numeric = rund::evidence::Id{};
  TEST_ASSERT(!key.complete());

  rund::cluster::RunKey unsharded_a = MakeCompleteKey();
  unsharded_a.sharded = false;
  unsharded_a.shard = rund::cluster::ShardRef{rund::cluster::JobId{111u},
                                              rund::cluster::ShardId{222u}};
  rund::cluster::RunKey unsharded_b = MakeCompleteKey();
  unsharded_b.sharded = false;
  unsharded_b.shard = rund::cluster::ShardRef{rund::cluster::JobId{333u},
                                              rund::cluster::ShardId{444u}};
  TEST_ASSERT(unsharded_a.complete());
  TEST_ASSERT(unsharded_b.complete());
  TEST_ASSERT(unsharded_a == unsharded_b);

  rund::cluster::RunKey shard_presence_changed = unsharded_a;
  shard_presence_changed.sharded = true;
  TEST_ASSERT(shard_presence_changed != unsharded_a);

  rund::cluster::RunKey shard_job_changed = MakeCompleteKey();
  shard_job_changed.shard.job = rund::cluster::JobId{21u};
  TEST_ASSERT(shard_job_changed != MakeCompleteKey());

  rund::cluster::RunKey shard_id_changed = MakeCompleteKey();
  shard_id_changed.shard.shard = rund::cluster::ShardId{31u};
  TEST_ASSERT(shard_id_changed != MakeCompleteKey());

  rund::cluster::RunKey numeric_changed = MakeCompleteKey();
  numeric_changed.numeric = rund::evidence::identify(rund::evidence::i64());
  TEST_ASSERT(MakeCompleteKey() != numeric_changed);
  return 0;
}
