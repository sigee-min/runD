#include "src/accel/kernel/callback.hpp"
#include "src/accel/kernel/prepared/model.hpp"
#include "src/accel/kernel/submission.hpp"

#include <atomic>
#include <memory>
#include <thread>

#include "claim.hpp"

namespace node_accel_contract {

using rund::node::accel::detail::KernelResult;

namespace {

void IgnoreCompletion(void *, KernelResult) noexcept {}

struct SubmissionOwner final {};

} // namespace

[[nodiscard]] bool SubmissionTransitions() {
  using namespace rund::node::accel::detail;
  SubmissionOwner owner{};
  submission::State<SubmissionOwner> state{};
  std::atomic_bool start{};
  std::atomic_uint accepted{};
  const auto claim = [&] {
    while (!start.load(std::memory_order_acquire)) {
      std::this_thread::yield();
    }
    if (submission::Begin(state, owner, IgnoreCompletion, nullptr)) {
      accepted.fetch_add(1u, std::memory_order_relaxed);
    }
  };
  std::thread first{claim};
  std::thread second{claim};
  start.store(true, std::memory_order_release);
  first.join();
  second.join();
  if (accepted.load(std::memory_order_relaxed) != 1u) {
    return false;
  }
  const submission::Claim<SubmissionOwner> taken = submission::Take(state);
  if (!taken || taken.owner != &owner || taken.completion != IgnoreCompletion ||
      taken.user != nullptr || submission::Take(state)) {
    return false;
  }
  if (submission::Begin(state, owner, nullptr, nullptr) ||
      !submission::Begin(state, owner, IgnoreCompletion, &owner)) {
    return false;
  }
  submission::Cancel(state);
  if (submission::Take(state) ||
      !submission::Begin(state, owner, IgnoreCompletion, nullptr)) {
    return false;
  }
  submission::Cancel(state);
  return true;
}

[[nodiscard]] bool PreparedPipelineClaimHasOneAuthority() {
  using namespace rund::node::accel::detail;
  prepared::PipelineSubmission submission{};
  if (submission.active() || submission.pipeline() != nullptr) {
    return false;
  }
  const auto owner = std::make_shared<prepared::PipelineState>();
  submission.owner = owner;
  if (!submission.active() || submission.pipeline() != owner.get()) {
    return false;
  }
  submission.owner.reset();
  return !submission.active() && submission.pipeline() == nullptr;
}

} // namespace node_accel_contract
