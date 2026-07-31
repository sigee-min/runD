[[nodiscard]] ::rund::detail::task::ChannelDecision
FinishOperation(::rund::detail::task::ChannelDecision result) const noexcept {
  const ::rund::detail::task::ChannelDecision complete =
      ::rund::detail::task::ChannelAccess::FinishCommittedOperation();
  if (!complete.status) {
    return complete;
  }
  result.complete_committed = false;
  result.complete_counted = false;
  return result;
}

[[nodiscard]] ::rund::detail::task::ChannelDecision FinishOk() const noexcept {
  return ::rund::detail::task::ChannelAccess::FinishCommit(false);
}

[[nodiscard]] ::rund::detail::task::ChannelDecision
FinishCountedOk() const noexcept {
  return ::rund::detail::task::ChannelAccess::FinishCommit(true);
}

[[nodiscard]] ::rund::detail::task::ChannelDecision
Finalize(::rund::detail::task::ChannelDecision result) const noexcept {
  if (!result.complete_committed) {
    return result;
  }
  if (result.complete_counted && result.status) {
    return FinishCountedOk();
  }
  return FinishOperation(result);
}

[[nodiscard]] ::rund::detail::task::ChannelDecision
FinalizeCommitted(::rund::detail::task::ChannelDecision result,
                  const bool counted = false) const noexcept {
  result.complete_committed = true;
  result.complete_counted = counted;
  return Finalize(result);
}

[[nodiscard]] Status
PublicStatus(::rund::detail::task::ChannelDecision result) const noexcept {
  return Finalize(result).status;
}

[[nodiscard]] RecvDecision<T>
FinalizeRecv(RecvDecision<T> result) const noexcept {
  result.decision = Finalize(result.decision);
  if (!result.decision.status) {
    result.value.reset();
  }
  return result;
}

[[nodiscard]] ReceiveResult<T>
PublicRecv(RecvDecision<T> result) const noexcept {
  result = FinalizeRecv(std::move(result));
  if (!result.decision.status || !result.value) {
    const ReasonCode code = result.decision.status
                                ? ReasonCode::TaskContextMissing
                                : result.decision.status.code();
    return ReceiveResult<T>::fail(code);
  }
  return ReceiveResult<T>::success(std::move(*result.value));
}
