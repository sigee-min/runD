#pragma once

#include <node/runtime/replay/model.hpp>

namespace rund::node {

struct RuntimeReplayDecodeResult {
  ::rund::replay::Code code = ::rund::replay::Code::RecordNotLoaded;
  RuntimeReplayRecord record{};

  [[nodiscard]] constexpr bool ok() const noexcept {
    return code == ::rund::replay::Code::Ok;
  }
  [[nodiscard]] constexpr explicit operator bool() const noexcept {
    return ok();
  }
  [[nodiscard]] std::string_view error() const noexcept {
    return ::rund::replay::error(code);
  }
};

} // namespace rund::node
