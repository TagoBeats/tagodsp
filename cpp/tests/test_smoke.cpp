#include <catch2/catch_test_macros.hpp>
#include <tagodsp/tagodsp.hpp>

TEST_CASE("library header compiles and version is set") {
    REQUIRE(tagodsp::versionMajor == 0);
}
