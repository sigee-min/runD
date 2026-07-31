#include "../local.hpp"

#include <cstring>
#include <rund/net/address.hpp>

#include <arpa/inet.h>
#include <netinet/in.h>

rund::net::Address NetStatsLoopbackAnyPort() {
  return rund::net::Address::loopback(rund::net::Family::IPv4);
}
