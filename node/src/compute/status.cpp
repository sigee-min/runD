#include "status.hpp"

#include "../hash/fnv.hpp"

#include <array>
#include <cstdint>

namespace rund::compute {
namespace {

constexpr auto Reasons = std::to_array<Reason>({
#define RUND_COMPUTE_REASON(name, value, text) Reason::name,
#include <rund/compute/reason.def>
#undef RUND_COMPUTE_REASON
});

[[nodiscard]] consteval bool complete_schema() {
  std::array<std::uint16_t, 9> counts{};
  std::array<std::uint16_t, 9> maxima{};
  for (const Reason reason : Reasons) {
    if (reason == Reason::Ok) {
      continue;
    }
    if (!detail::valid(reason)) {
      return false;
    }
    const auto group = static_cast<std::size_t>(detail::category(reason));
    const auto value = detail::ordinal(reason);
    if (group == 0u || group >= counts.size() || value == 0u) {
      return false;
    }
    ++counts[group];
    if (value > maxima[group]) {
      maxima[group] = value;
    }
  }
  for (std::size_t category = 1; category < counts.size(); ++category) {
    if (counts[category] != maxima[category]) {
      return false;
    }
  }
  return true;
}

static_assert(complete_schema());

[[nodiscard]] constexpr std::uint64_t
hash(const std::string_view text) noexcept {
  ::rund::node::hash_detail::Fnv value{
      ::rund::node::hash_detail::kFnvStandardOffset};
  for (const unsigned char byte : text) {
    value.Byte(byte);
  }
  return value.Finish();
}

} // namespace

namespace detail {

std::string_view reason_message(const Reason reason) noexcept {
  switch (reason) {
#define RUND_COMPUTE_REASON(name, value, text)                                 \
  case Reason::name:                                                           \
    return text;
#include <rund/compute/reason.def>
#undef RUND_COMPUTE_REASON
  }
  return "compute_reason_invalid";
}

static Reason parse_reason(const std::string_view text) noexcept {
  if (text.empty()) {
    return Reason::ReasonInvalid;
  }
  Reason reason = Reason::ReasonInvalid;
  switch (hash(text)) {
#define RUND_COMPUTE_REASON(name, value, text)                                 \
  case hash(text):                                                             \
    reason = Reason::name;                                                     \
    break;
#include <rund/compute/reason.def>
#undef RUND_COMPUTE_REASON
  default:
    break;
  }
  return reason != Reason::Ok && reason_message(reason) == text
             ? reason
             : Reason::ReasonInvalid;
}

Reason project_reason(const std::string_view text,
                      const Reason boundary) noexcept {
  const Reason exact = parse_reason(text);
  const Reason normalized = valid(boundary) ? boundary : Reason::ReasonInvalid;
  return exact == Reason::ReasonInvalid ? normalized : exact;
}

} // namespace detail

std::string_view Status::error() const noexcept {
  return detail::reason_message(reason_);
}

} // namespace rund::compute
