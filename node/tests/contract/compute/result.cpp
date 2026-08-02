#include <rund/compute.hpp>

#include "src/compute/job/control/model.hpp"

#include <cstdint>
#include <memory>
#include <string_view>
#include <type_traits>

template <class... Args>
concept MakesComputeFailure = requires(Args &&...args) {
  rund::compute::Status::fail(static_cast<Args &&>(args)...);
};

static_assert(!MakesComputeFailure<rund::compute::Code, std::string_view>);
static_assert(MakesComputeFailure<rund::compute::Reason>);
static_assert(sizeof(rund::compute::Status) == sizeof(rund::compute::Reason));
static_assert(sizeof(rund::compute::Status) == 2u);
static_assert(std::is_trivially_copyable_v<rund::compute::Location>);
static_assert(std::is_trivially_copyable_v<rund::compute::Failure>);

namespace {

struct ThrowingMove final {
  explicit ThrowingMove(const int initial) noexcept : value(initial) {}
  ThrowingMove(const ThrowingMove &) = delete;
  ThrowingMove &operator=(const ThrowingMove &) = delete;
  ThrowingMove(ThrowingMove &&) { throw 1; }
  ThrowingMove &operator=(ThrowingMove &&) { throw 1; }

  int value = 0;
};

constexpr char kNativeReason[] = "native_pipeline_compile_failed";
constexpr char kNativeReasonCopy[] = "native_pipeline_compile_failed";
constexpr char kOtherNativeReason[] = "native_pipeline_binding_failed";

constexpr rund::compute::Location kCompleteLocation{
    .step = 5u,
    .iteration = 7u,
    .node = 11u,
    .template_index = 13u,
    .occurrence_index = 17u,
    .outer_iteration = 19u,
    .inner_iteration = 23u,
    .nested_phase = rund::compute::PipelineNestedPhase::Action,
    .native_reason_key = kNativeReason,
};

[[nodiscard]] constexpr bool LocationValueContract() noexcept {
  using rund::compute::Location;
  using rund::compute::PipelineNestedPhase;
  constexpr Location same_value{
      .step = 5u,
      .iteration = 7u,
      .node = 11u,
      .template_index = 13u,
      .occurrence_index = 17u,
      .outer_iteration = 19u,
      .inner_iteration = 23u,
      .nested_phase = PipelineNestedPhase::Action,
      .native_reason_key = kNativeReasonCopy,
  };
  if (Location{}.known() || !Location{.step = 0u}.known() ||
      !Location{.iteration = 0u}.known() || !Location{.node = 0u}.known() ||
      !Location{.template_index = 0u}.known() ||
      !Location{.occurrence_index = 0u}.known() ||
      !Location{.outer_iteration = 0u}.known() ||
      !Location{.inner_iteration = 0u}.known() ||
      !Location{.nested_phase = PipelineNestedPhase::Seed}.known() ||
      !Location{.native_reason_key = kNativeReason}.known() ||
      !kCompleteLocation.known() || kCompleteLocation != same_value) {
    return false;
  }
  Location changed = kCompleteLocation;
  changed.step = 6u;
  if (changed == kCompleteLocation) {
    return false;
  }
  changed = kCompleteLocation;
  changed.iteration = 8u;
  if (changed == kCompleteLocation) {
    return false;
  }
  changed = kCompleteLocation;
  changed.node = 12u;
  if (changed == kCompleteLocation) {
    return false;
  }
  changed = kCompleteLocation;
  changed.template_index = 14u;
  if (changed == kCompleteLocation) {
    return false;
  }
  changed = kCompleteLocation;
  changed.occurrence_index = 18u;
  if (changed == kCompleteLocation) {
    return false;
  }
  changed = kCompleteLocation;
  changed.outer_iteration = 20u;
  if (changed == kCompleteLocation) {
    return false;
  }
  changed = kCompleteLocation;
  changed.inner_iteration = 24u;
  if (changed == kCompleteLocation) {
    return false;
  }
  changed = kCompleteLocation;
  changed.nested_phase = PipelineNestedPhase::Fold;
  if (changed == kCompleteLocation) {
    return false;
  }
  changed = kCompleteLocation;
  changed.native_reason_key = kOtherNativeReason;
  return changed != kCompleteLocation;
}

static_assert(LocationValueContract());

} // namespace

int RunComputeResultContract() {
  using rund::compute::Code;
  using rund::compute::Location;
  using rund::compute::Reason;
  using rund::compute::Result;
  using rund::compute::Status;

  constexpr Status success = Status::success();
  constexpr Status rejected_ok = Status::fail(Reason::Ok);
  constexpr Status forged = Status::fail(static_cast<Reason>(0xffffu));
  constexpr Status binding = Status::fail(Reason::ShapeMismatch);
  static_assert(success.ok() && success.code() == Code::Ok &&
                success.exit_code() == 0);
  static_assert(rejected_ok.reason() == Reason::ReasonInvalid);
  static_assert(!forged.ok() && forged.code() == Code::Invalid &&
                forged.exit_code() == 1);
  static_assert(forged.reason() == Reason::ReasonInvalid);
  static_assert(!binding.ok() && binding.reason() == Reason::ShapeMismatch &&
                binding.code() == Code::Binding);

  const auto successful = Result<std::int32_t>::success(1);
  const auto forged_result =
      Result<std::int32_t>::fail(static_cast<Reason>(0xffffu));
  if (!successful || successful.code() != Code::Ok ||
      successful.exit_code() != 0 || forged_result ||
      forged_result.exit_code() != 1 || forged_result.code() != Code::Invalid ||
      forged_result.reason() != Reason::ReasonInvalid) {
    return 1;
  }

  auto transformed = Result<std::int32_t>::success(7).transform(
      [](const std::int32_t value) { return value * 3; });
  if (!transformed || *transformed != 21) {
    return 2;
  }

  bool called = false;
  auto failed = Result<std::int32_t>::fail(Reason::ShapeMismatch)
                    .transform([&](const std::int32_t value) {
                      called = true;
                      return value;
                    });
  if (failed || called || failed.code() != Code::Binding ||
      failed.reason() != Reason::ShapeMismatch) {
    return 3;
  }

  auto located =
      Result<std::int32_t>::fail(Reason::LoweringInvalid, kCompleteLocation)
          .transform([](const std::int32_t value) {
            return static_cast<std::uint64_t>(value);
          });
  if (located || located.reason() != Reason::LoweringInvalid ||
      !located.location().known() || located.location() != kCompleteLocation) {
    return 12;
  }

  auto chained =
      Result<std::int32_t>::success(5).and_then([](const std::int32_t value) {
        return Result<std::uint64_t>::success(
            static_cast<std::uint64_t>(value + 1));
      });
  if (!chained || *chained != 6u) {
    return 4;
  }

  auto propagated = Result<std::int32_t>::fail(Reason::TransferInvalid)
                        .and_then([](const std::int32_t value) {
                          return Result<std::uint64_t>::success(value);
                        });
  if (propagated || propagated.code() != Code::Transfer ||
      propagated.reason() != Reason::TransferInvalid) {
    return 5;
  }
  auto located_chain =
      Result<std::int32_t>::fail(Reason::LoweringInvalid, kCompleteLocation)
          .and_then([](const std::int32_t value) {
            return Result<std::uint64_t>::success(value);
          });
  if (located_chain || located_chain.location() != kCompleteLocation) {
    return 13;
  }

  auto prepared_state = std::make_shared<rund::compute::detail::JobState>();
  prepared_state->prepared.failed_node = 23u;
  auto prepared_failure = rund::compute::detail::finish_prepare(
      prepared_state, Status::fail(Reason::LoweringInvalid));
  if (prepared_failure || prepared_failure.location().step != Location::none ||
      prepared_failure.location().iteration != Location::none ||
      prepared_failure.location().node != 23u ||
      prepared_failure.location().template_index != Location::none ||
      prepared_failure.location().occurrence_index != Location::none ||
      prepared_failure.location().outer_iteration != Location::none ||
      prepared_failure.location().inner_iteration != Location::none ||
      prepared_failure.location().nested_phase !=
          rund::compute::PipelineNestedPhase::None ||
      prepared_failure.location().native_reason_key != nullptr ||
      !prepared_failure.location().known()) {
    return 14;
  }

  if (Result<std::int32_t>::success(9).value_or(4) != 9 ||
      Result<std::int32_t>::fail(Reason::RunInvalid).value_or(4) != 4) {
    return 6;
  }
  const Result<std::int32_t> constant = Result<std::int32_t>::success(8);
  auto constant_result = constant.transform(
      [](const auto value) { return static_cast<std::uint64_t>(value + 2); });
  Result<std::int32_t> lvalue = Result<std::int32_t>::success(11);
  auto lvalue_result = lvalue.and_then([](const auto value) {
    return Result<std::int32_t>::success(value - 1);
  });
  if (!constant_result || *constant_result != 10u || !lvalue_result ||
      *lvalue_result != 10 || constant.value_or(3) != 8) {
    return 7;
  }
  const Status device_busy = Status::fail(Reason::DeviceBusy);
  if (device_busy || device_busy.code() != Code::Execution ||
      device_busy.reason() != Reason::DeviceBusy) {
    return 8;
  }

  Result<ThrowingMove> invalidated =
      Result<ThrowingMove>::fail(Reason::RunInvalid);
  Result<ThrowingMove> moving = Result<ThrowingMove>::success(7);
  try {
    invalidated = std::move(moving);
    return 9;
  } catch (const int) {
  }
  if (invalidated || invalidated.operator->() != nullptr ||
      invalidated.code() != Code::Invalid ||
      invalidated.reason() != Reason::ValueInvalid ||
      invalidated.error() != "compute_value_invalid" ||
      invalidated.exit_code() != 1) {
    return 10;
  }
  bool invalidated_called = false;
  auto invalidated_transform =
      invalidated.transform([&](ThrowingMove &) -> std::int32_t {
        invalidated_called = true;
        return 1;
      });
  if (invalidated_called || invalidated_transform ||
      invalidated_transform.reason() != Reason::ValueInvalid) {
    return 11;
  }
  return 0;
}
