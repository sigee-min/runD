#pragma once

#include <node/runtime/replay/codec/result.hpp>
#include <rund/replay/limits.hpp>

#include <cstddef>
#include <span>

namespace rund::node {

[[nodiscard]] RuntimeReplayDecodeResult
DecodeRuntimeReplayRecord(std::span<const std::byte> encoded,
                          ::rund::replay::Limits limits = {});

} // namespace rund::node
