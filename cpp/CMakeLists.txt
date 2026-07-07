cmake_minimum_required(VERSION 3.24)
project(tagodsp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(tagodsp INTERFACE)
target_include_directories(tagodsp INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

include(FetchContent)
FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.7.1
)
FetchContent_MakeAvailable(Catch2)

enable_testing()
add_executable(tagodsp_tests tests/test_smoke.cpp)
target_link_libraries(tagodsp_tests PRIVATE tagodsp Catch2::Catch2WithMain)

include(Catch)
catch_discover_tests(tagodsp_tests)
