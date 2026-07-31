struct SchedulerHandleMap final {
  struct Entry final {
    std::uint64_t physical = 0u;
    std::uint64_t canonical = 0u;
    std::uint32_t stamp = 0u;
  };

  [[nodiscard]] bool configure(const std::uint32_t capacity) noexcept {
    if (capacity == 0u) {
      std::vector<Entry>{}.swap(entries);
      mask = 0u;
      limit = 0u;
      epoch = 0u;
      live = 0u;
      next = 1u;
      return true;
    }
    if (static_cast<std::size_t>(capacity) >
        std::numeric_limits<std::size_t>::max() / 2u) {
      return false;
    }
    std::size_t size = 1u;
    const std::size_t target = static_cast<std::size_t>(capacity) * 2u;
    while (size < target) {
      if (size > std::numeric_limits<std::size_t>::max() / 2u) {
        return false;
      }
      size *= 2u;
    }
    try {
      std::vector<Entry> prepared(size);
      entries.swap(prepared);
    } catch (...) {
      return false;
    }
    limit = capacity;
    mask = size - 1u;
    epoch = 0u;
    live = 0u;
    next = 1u;
    return true;
  }

  void begin() noexcept {
    constexpr std::uint32_t max_epoch =
        std::numeric_limits<std::uint32_t>::max() >> 1u;
    if (epoch == max_epoch) {
      std::fill(entries.begin(), entries.end(), Entry{});
      epoch = 1u;
    } else {
      ++epoch;
    }
    live = 0u;
    next = 1u;
  }

  [[nodiscard]] bool find_or_admit(const std::uint64_t physical,
                                   std::uint64_t &canonical) noexcept {
    if (physical == 0u) {
      canonical = 0u;
      return true;
    }
    if (epoch == 0u || entries.empty()) {
      return false;
    }
    const std::uint32_t live_stamp = epoch * 2u + 1u;
    const std::uint32_t retired_stamp = epoch * 2u;
    std::size_t slot = hash(physical) & mask;
    std::size_t retired = entries.size();
    for (std::size_t probe = 0u; probe < entries.size(); ++probe) {
      Entry &entry = entries[slot];
      if (entry.stamp == live_stamp && entry.physical == physical) {
        canonical = entry.canonical;
        return true;
      }
      if (entry.stamp == retired_stamp) {
        if (retired == entries.size()) {
          retired = slot;
        }
      } else if ((entry.stamp >> 1u) != epoch) {
        return admit(retired == entries.size() ? slot : retired, physical,
                     canonical, live_stamp);
      }
      slot = (slot + 1u) & mask;
    }
    return retired != entries.size() &&
           admit(retired, physical, canonical, live_stamp);
  }

  [[nodiscard]] bool retire(const std::uint64_t physical) noexcept {
    if (physical == 0u) {
      return true;
    }
    if (epoch == 0u || entries.empty()) {
      return false;
    }
    const std::uint32_t live_stamp = epoch * 2u + 1u;
    std::size_t slot = hash(physical) & mask;
    for (std::size_t probe = 0u; probe < entries.size(); ++probe) {
      Entry &entry = entries[slot];
      if ((entry.stamp >> 1u) != epoch) {
        return false;
      }
      if (entry.stamp == live_stamp && entry.physical == physical) {
        entry.stamp = epoch * 2u;
        --live;
        return true;
      }
      slot = (slot + 1u) & mask;
    }
    return false;
  }

  std::vector<Entry> entries{};
  std::size_t mask = 0u;
  std::uint32_t limit = 0u;
  std::uint32_t epoch = 0u;
  std::uint32_t live = 0u;
  std::uint64_t next = 1u;

private:
  [[nodiscard]] static std::size_t hash(std::uint64_t value) noexcept {
    value ^= value >> 30u;
    value *= 0xbf58476d1ce4e5b9ull;
    value ^= value >> 27u;
    value *= 0x94d049bb133111ebull;
    value ^= value >> 31u;
    return static_cast<std::size_t>(value);
  }

  [[nodiscard]] bool admit(const std::size_t slot,
                           const std::uint64_t physical,
                           std::uint64_t &canonical,
                           const std::uint32_t live_stamp) noexcept {
    if (live >= limit || next == 0u) {
      return false;
    }
    Entry &entry = entries[slot];
    entry.physical = physical;
    entry.canonical = next++;
    entry.stamp = live_stamp;
    ++live;
    canonical = entry.canonical;
    return true;
  }
};

struct SchedulerPlanState final {
  struct Bases final {
    std::uint64_t task = 1u;
    std::uint64_t scope = 1u;
    std::uint64_t wait = 1u;
    std::uint64_t timer = 1u;
    std::uint64_t observation = 1u;
    std::uint64_t event = 1u;
    std::uint64_t channel = 1u;
    std::uint64_t trace_epoch = 1u;
    std::uint64_t ticket = 1u;
  } bases{};

  ::rund::replay::detail::scope::Plan value{};
  SchedulerHandleMap handles{};
  ReasonCode failure = ReasonCode::Ok;
  bool installed = false;

  [[nodiscard]] bool configure_handles(const std::uint32_t capacity) noexcept {
    return handles.configure(capacity);
  }

  void begin() noexcept {
    failure = ReasonCode::Ok;
    handles.begin();
  }

  void fail() noexcept {
    if (failure == ReasonCode::Ok) {
      failure = ReasonCode::ReplayScopeIdentityInvalid;
    }
  }

  [[nodiscard]] std::uint64_t project(const std::uint64_t physical,
                                      const std::uint64_t first) noexcept {
    if (!installed) {
      return physical;
    }
    if (physical == 0u) {
      return 0u;
    }
    if (first == 0u || physical < first) {
      fail();
      return 0u;
    }
    return physical - first + 1u;
  }

  [[nodiscard]] std::uint64_t task(const std::uint64_t value) noexcept {
    return project(value, bases.task);
  }

  [[nodiscard]] std::uint64_t scope(const std::uint64_t value) noexcept {
    return project(value, bases.scope);
  }

  [[nodiscard]] std::uint64_t wait(const std::uint64_t value) noexcept {
    return project(value, bases.wait);
  }

  [[nodiscard]] std::uint64_t timer(const std::uint64_t value) noexcept {
    return project(value, bases.timer);
  }

  [[nodiscard]] std::uint64_t
  observation(const std::uint64_t value) noexcept {
    return project(value, bases.observation);
  }

  [[nodiscard]] std::uint64_t event(const std::uint64_t value) noexcept {
    return project(value, bases.event);
  }

  [[nodiscard]] std::uint64_t
  channel(const std::uint64_t value) noexcept {
    return project(value, bases.channel);
  }

  [[nodiscard]] std::uint64_t
  trace_epoch(const std::uint64_t value) noexcept {
    return project(value, bases.trace_epoch);
  }

  [[nodiscard]] std::uint64_t ticket(const std::uint64_t value) noexcept {
    return project(value, bases.ticket);
  }

  [[nodiscard]] std::uint64_t handle(const std::uint64_t value) noexcept {
    if (!installed) {
      return value;
    }
    std::uint64_t canonical = 0u;
    if (!handles.find_or_admit(value, canonical)) {
      fail();
      return 0u;
    }
    return canonical;
  }

  [[nodiscard]] int descriptor(const int value) noexcept {
    if (!installed) {
      return value;
    }
    if (value == -1) {
      return -1;
    }
    if (value < 0) {
      fail();
      return -1;
    }
    const std::uint64_t canonical =
        handle(static_cast<std::uint64_t>(value) + 1u);
    if (canonical == 0u ||
        canonical > static_cast<std::uint64_t>(
                        std::numeric_limits<int>::max())) {
      fail();
      return -1;
    }
    return static_cast<int>(canonical - 1u);
  }

  void retire(const std::uint64_t physical) noexcept {
    if (!installed) {
      return;
    }
    if (!handles.retire(physical)) {
      fail();
    }
  }

  [[nodiscard]] ::rund::replay::detail::scope::Mode mode() const noexcept {
    return value.mode;
  }

  [[nodiscard]] const ::rund::replay::detail::scope::Expected *expected() const
      noexcept {
    return value.expected.get();
  }

  [[nodiscard]] const replay_detail::InputPlan *choices() const noexcept {
    return value.choices.get();
  }
};
