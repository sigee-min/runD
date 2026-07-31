namespace consumer_fixture {
[[nodiscard]] bool ValidatePrivateSdkLink();
}  // namespace consumer_fixture

int main() { return consumer_fixture::ValidatePrivateSdkLink() ? 0 : 1; }
