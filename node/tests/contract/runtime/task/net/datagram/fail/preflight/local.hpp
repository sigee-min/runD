#pragma once

#include "../../local.hpp"

[[nodiscard]] rund::SessionConfig DatagramPreflightRunSpec() noexcept;
[[nodiscard]] bool NetDatagramRejectsWouldBlockPreflight();
[[nodiscard]] bool NetDatagramRejectsNullPreflight();
