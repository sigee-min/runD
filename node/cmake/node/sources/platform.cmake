if(RUND_NODE_USE_UNAVAILABLE_PLATFORM)
  list(APPEND NODE_SOURCES
    src/runtime/platform/unavailable/io.cpp
    src/runtime/platform/unavailable/net.cpp
    src/runtime/platform/unavailable/reactor.cpp
  )
else()
  list(APPEND NODE_SOURCES
    src/runtime/platform/posix/address.cpp
    src/runtime/platform/posix/io.cpp
    src/runtime/platform/posix/probe.cpp
    src/runtime/platform/posix/net.cpp
    src/runtime/platform/posix/net/datagram.cpp
    src/runtime/platform/posix/net/options.cpp
    src/runtime/platform/posix/net/vectored.cpp
  )
endif()

if(RUND_NODE_USE_UNAVAILABLE_PLATFORM)
  # The explicit unavailable owner is the selected reactor backend.
elseif(APPLE OR CMAKE_SYSTEM_NAME STREQUAL "FreeBSD")
  list(APPEND NODE_SOURCES
    src/runtime/platform/mac/reactor/open.cpp
    src/runtime/platform/mac/reactor/events.cpp
    src/runtime/platform/mac/reactor/probe.cpp
    src/runtime/platform/mac/reactor/registration.cpp
    src/runtime/platform/mac/reactor/registration/batch.cpp
    src/runtime/platform/mac/reactor/registration/interest.cpp
    src/runtime/platform/mac/reactor/registration/native.cpp
    src/runtime/platform/mac/reactor/teardown.cpp
    src/runtime/platform/mac/reactor/wait.cpp
  )
elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  list(APPEND NODE_SOURCES
    src/runtime/platform/linux/reactor/open.cpp
    src/runtime/platform/linux/reactor/events.cpp
    src/runtime/platform/linux/reactor/probe.cpp
    src/runtime/platform/linux/reactor/registration.cpp
    src/runtime/platform/linux/reactor/teardown.cpp
    src/runtime/platform/linux/reactor/wait.cpp
  )
elseif(UNIX)
  list(APPEND NODE_SOURCES
    src/runtime/platform/portable/reactor/open.cpp
    src/runtime/platform/portable/reactor/events.cpp
    src/runtime/platform/portable/reactor/probe.cpp
    src/runtime/platform/portable/reactor/registration.cpp
    src/runtime/platform/portable/reactor/teardown.cpp
    src/runtime/platform/portable/reactor/wait.cpp
  )
endif()
