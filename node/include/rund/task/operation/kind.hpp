#pragma once

#include <cstdint>

namespace rund::detail::task {

enum class OperationKind : std::uint16_t {
  None = 0u,
  RootSubmit = 1u,
  Spawn = 2u,
  Complete = 3u,
  Fail = 4u,
  Yield = 5u,
  SleepZero = 6u,
  TimerPark = 7u,
  TimerWake = 8u,
  JoinPark = 9u,
  JoinWake = 10u,
  ScopeEnter = 11u,
  ScopePark = 12u,
  ScopeWake = 13u,
  ChannelMake = 14u,
  ChannelSend = 15u,
  ChannelRecv = 16u,
  ChannelClose = 17u,
  ChannelWake = 18u,
  IoPark = 19u,
  IoWake = 20u,
  DeadlockWake = 21u,
  ChannelMatch = 22u,
  ChannelMatchBatch = 23u,
  TaskSpawnBatch = 24u,
  TaskTerminalBatch = 25u,
  YieldBatch = 26u,
  JoinBatch = 27u,
  PrimitiveTrap = 28u,
  TaskRootJoinEpochBatch = 29u,
  ExternalPark = 32u,
  ExternalWake = 33u,
};

} // namespace rund::detail::task
