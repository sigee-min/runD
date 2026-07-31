#pragma once

#include <rund/session.hpp>

#include <cstdint>

struct SocketPairCleanup {
  int left = -1;
  int right = -1;
  ~SocketPairCleanup();
  SocketPairCleanup() = default;
  SocketPairCleanup(const SocketPairCleanup &) = delete;
  SocketPairCleanup &operator=(const SocketPairCleanup &) = delete;
};

bool MakeSocketPair(SocketPairCleanup &cleanup);
rund::SessionConfig
ReadyManyRunSpec(std::uint32_t task_capacity = 4u,
                 std::uint32_t timer_capacity = 8u,
                 std::uint32_t reactor_wait_capacity = 128u) noexcept;

int RunRuntimeTaskNetReadyManyReadContract();
int RunRuntimeTaskNetReadyManyWriteContract();
int RunRuntimeTaskNetReadyManyAcceptContract();
int RunRuntimeTaskNetReadyManyConnectContract();

int RunNetReadyManyReadMissingRuntimeCase();
int RunNetReadyManyReadImmediateCase();
int RunNetReadyManyReadBatchCase();
int RunNetReadyManyValidationScaleCase();
int RunNetReadyManyWriteReadWriteCase();
int RunNetReadyManyWriteImmediateOnlyCase();
int RunNetReadyManyConnectMultiInterestCase();
int RunNetReadyManyConnectGenerationCase();
