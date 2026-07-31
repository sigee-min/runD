#include "local/model.hpp"

#include "src/runtime/replay/input/plan.hpp"
#include "test/assert.hpp"

#include <array>
#include <memory>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace {

template <typename T>
concept HasByteCount = requires(T value) { value.byte_count; };

static_assert(std::is_aggregate_v<rund::replay::Input>);
static_assert(!HasByteCount<rund::replay::Input>);
static_assert(!std::is_aggregate_v<rund::replay::Choice>);
static_assert(!std::is_aggregate_v<rund::replay::Binding>);
static_assert(!std::is_copy_constructible_v<rund::replay::Binding>);
static_assert(!std::is_move_constructible_v<rund::replay::Binding>);
static_assert(!std::is_copy_constructible_v<rund::replay::Channel>);
static_assert(!std::is_move_constructible_v<rund::replay::Channel>);
static_assert(noexcept(std::declval<const rund::replay::Binding &>().resume(
    std::declval<const rund::replay::Checkpoint &>())));
static_assert(!std::is_copy_constructible_v<rund::replay::Context>);
static_assert(!std::is_move_constructible_v<rund::replay::Context>);
static_assert(!std::is_copy_constructible_v<rund::replay::Value>);
static_assert(!std::is_move_constructible_v<rund::replay::Value>);
static_assert(!std::is_copy_constructible_v<rund::replay::Live>);
static_assert(std::is_nothrow_move_constructible_v<rund::replay::Live>);
static_assert(std::is_nothrow_move_assignable_v<rund::replay::Live>);
static_assert(std::is_nothrow_copy_constructible_v<rund::replay::Record>);
static_assert(std::is_nothrow_copy_constructible_v<rund::replay::Check>);
static_assert(std::is_trivially_copyable_v<rund::replay::Choice>);
static_assert(!std::is_copy_constructible_v<rund::replay::Resume>);
static_assert(!std::is_move_constructible_v<rund::replay::Resume>);
static_assert(std::is_same_v<decltype(rund::replay::Record::load(
                                 std::span<const std::byte>{})),
                             rund::replay::Load<rund::replay::Record>>);
static_assert(std::is_same_v<decltype(rund::replay::Checkpoint::load(
                                 std::span<const std::byte>{})),
                             rund::replay::Load<rund::replay::Checkpoint>>);

} // namespace

namespace runtime_task_replay_run {

int Surface(Model &model) {
  TEST_ASSERT(model.binding);
  TEST_ASSERT(model.session.open(model.config));
  TEST_ASSERT(model.commands);

  rund::replay::Live live = rund::replay::live(model.session, model.step);
  TEST_ASSERT(live);
  TEST_ASSERT(live.code() == rund::replay::Code::Ok);
  TEST_ASSERT(live.exit_code() == 0);
  TEST_ASSERT(model.producer_calls == 1u);
  TEST_ASSERT(model.callback_calls == 1u);
  TEST_ASSERT(model.observed_size == model.payload.size());
  rund::replay::Live moved_live = std::move(live);
  TEST_ASSERT(moved_live);
  TEST_ASSERT(live.code() == rund::replay::Code::SessionResultMissing);
  TEST_ASSERT(live.scope() == 0u);
  TEST_ASSERT(live.observations().empty());
  TEST_ASSERT(live.events().empty());

  const rund::replay::Live failed_live =
      rund::replay::live(model.session, [](rund::replay::Context &) {
        throw std::runtime_error{"fail"};
      });
  TEST_ASSERT(!failed_live);
  TEST_ASSERT(failed_live.code() ==
              rund::replay::Code::RuntimeScopeCallbackFailed);
  TEST_ASSERT(failed_live.exit_code() == 1);

  model.producer_calls = 0u;
  model.callback_calls = 0u;
  model.baseline = rund::replay::record(model.session, model.step);
  TEST_ASSERT(*model.baseline);
  TEST_ASSERT(model.producer_calls == 1u);
  TEST_ASSERT(model.callback_calls == 1u);
  TEST_ASSERT(model.baseline->input_count() == 1u);
  TEST_ASSERT(model.baseline->storage_report().logical_bytes ==
              model.payload.size());
  TEST_ASSERT(model.baseline->storage_report().encoded_bytes <=
              model.payload.size());
  TEST_ASSERT(model.baseline->storage_report().growths == 0u);

  const rund::replay::Binding input_only{};
  TEST_ASSERT(input_only);
  TEST_ASSERT(!input_only.checkpointable());
  const rund::replay::Checkpoint unavailable =
      input_only.checkpoint(*model.baseline, model.first_state);
  TEST_ASSERT(!unavailable);
  TEST_ASSERT(unavailable.code() == rund::replay::Code::StateSchemaInvalid);

  std::array frozen_bytes{std::byte{0x51}, std::byte{0x52}};
  const std::array frozen_choice{
      model.commands.choice(kSequence, frozen_bytes)};
  const rund::replay::detail::ReplayScenarioPlan frozen_plan =
      rund::replay::detail::prepare_scenario(model.session, *model.baseline,
                                             frozen_choice);
  TEST_ASSERT(frozen_plan.ok());
  const auto owner =
      std::static_pointer_cast<const rund::node::replay_detail::InputPlan>(
          frozen_plan.choices);
  const rund::node::replay_detail::InputPatch *const patch =
      owner->find(kInput.id, kInput.schema, kSequence);
  TEST_ASSERT(patch != nullptr);
  const rund::node::replay_detail::payload::Bytes frozen = owner->bytes(*patch);
  TEST_ASSERT(frozen.size() == frozen_bytes.size());
  TEST_ASSERT(owner->retained_bytes() == frozen_bytes.size());
  TEST_ASSERT(frozen.data() != frozen_bytes.data());
  frozen_bytes[0] = std::byte{0xff};
  TEST_ASSERT(frozen.span()[0] == std::byte{0x51});
  return 0;
}

} // namespace runtime_task_replay_run
