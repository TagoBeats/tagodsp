// Analytic checks for the FFT and the STFT framing. The golden diff against
// Python lives in test_masking.cpp; these tests exist so a broken transform
// fails on something a human can reason about instead of on a wall of numbers.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <complex>
#include <vector>

#include <tagodsp/fft.hpp>
#include <tagodsp/stft.hpp>

using Catch::Matchers::WithinAbs;

TEST_CASE("fft rejects non-power-of-two sizes") {
    REQUIRE_THROWS_AS(tagodsp::Fft(0), std::invalid_argument);
    REQUIRE_THROWS_AS(tagodsp::Fft(6), std::invalid_argument);
    REQUIRE_NOTHROW(tagodsp::Fft(8));
}

TEST_CASE("fft of a constant is a single DC bin") {
    const std::size_t n = 64;
    std::vector<std::complex<double>> a(n, std::complex<double>(1.0, 0.0));
    tagodsp::Fft(n).forward(a);
    REQUIRE_THAT(a[0].real(), WithinAbs(static_cast<double>(n), 1e-12));
    REQUIRE_THAT(a[0].imag(), WithinAbs(0.0, 1e-12));
    for (std::size_t k = 1; k < n; ++k) {
        REQUIRE_THAT(std::abs(a[k]), WithinAbs(0.0, 1e-12));
    }
}

TEST_CASE("fft puts a bin-centered sine in exactly one bin") {
    // A cosine at bin k0 has magnitude n/2 at k0 and at n - k0, nothing else.
    const std::size_t n = 128;
    const std::size_t k0 = 9;
    std::vector<std::complex<double>> a(n);
    for (std::size_t i = 0; i < n; ++i) {
        const double phase = 2.0 * tagodsp::kPi * static_cast<double>(k0) *
                             static_cast<double>(i) / static_cast<double>(n);
        a[i] = std::complex<double>(std::cos(phase), 0.0);
    }
    tagodsp::Fft(n).forward(a);
    for (std::size_t k = 0; k < n; ++k) {
        const double expected = (k == k0 || k == n - k0) ? static_cast<double>(n) / 2.0 : 0.0;
        REQUIRE_THAT(std::abs(a[k]), WithinAbs(expected, 1e-10));
    }
}

TEST_CASE("fft against a direct DFT on an arbitrary signal") {
    const std::size_t n = 32;
    std::vector<std::complex<double>> a(n);
    std::vector<std::complex<double>> reference(n);
    for (std::size_t i = 0; i < n; ++i) {
        a[i] = std::complex<double>(std::sin(0.7 * static_cast<double>(i)) +
                                        0.3 * static_cast<double>(i % 5),
                                    0.0);
    }
    for (std::size_t k = 0; k < n; ++k) {
        std::complex<double> acc{0.0, 0.0};
        for (std::size_t i = 0; i < n; ++i) {
            const double angle = -2.0 * tagodsp::kPi * static_cast<double>(k) *
                                 static_cast<double>(i) / static_cast<double>(n);
            acc += a[i] * std::complex<double>(std::cos(angle), std::sin(angle));
        }
        reference[k] = acc;
    }
    tagodsp::Fft(n).forward(a);
    for (std::size_t k = 0; k < n; ++k) {
        REQUIRE_THAT(std::abs(a[k] - reference[k]), WithinAbs(0.0, 1e-10));
    }
}

TEST_CASE("rfft keeps the non-redundant half") {
    const std::size_t n = 16;
    tagodsp::Rfft rfft(n);
    REQUIRE(rfft.nBins() == n / 2 + 1);
    std::vector<double> x(n, 0.0);
    x[0] = 1.0; // unit impulse: every bin is 1
    std::vector<std::complex<double>> out(rfft.nBins());
    rfft.forward(x.data(), out.data());
    for (const auto& bin : out) {
        REQUIRE_THAT(bin.real(), WithinAbs(1.0, 1e-12));
        REQUIRE_THAT(bin.imag(), WithinAbs(0.0, 1e-12));
    }
}

TEST_CASE("stft framing matches the centered-pad convention") {
    tagodsp::Stft stft(1024, 256);
    REQUIRE(stft.nBins() == 513);
    // padded length 32000 + 1024, frames = 1 + (33024 - 1024) / 256
    REQUIRE(stft.frameCount(32000) == 126);
    REQUIRE(stft.frameCount(0) == 1); // the pad alone still fills one frame

    const auto freqs = stft.freqs(16000.0);
    REQUIRE_THAT(freqs.front(), WithinAbs(0.0, 1e-12));
    REQUIRE_THAT(freqs.back(), WithinAbs(8000.0, 1e-9));

    const auto times = stft.times(126, 16000.0);
    REQUIRE_THAT(times.back(), WithinAbs(2.0, 1e-12));
}

TEST_CASE("stft of a bin-centered tone concentrates in that bin") {
    const std::size_t nFft = 1024;
    const double sr = 16000.0;
    const std::size_t bin = 40; // 40 * 16000 / 1024 = 625 Hz, exactly on a bin
    tagodsp::Stft stft(nFft, nFft / 4);
    std::vector<double> x(4 * nFft);
    for (std::size_t i = 0; i < x.size(); ++i) {
        x[i] = std::sin(2.0 * tagodsp::kPi * static_cast<double>(bin) * sr /
                        static_cast<double>(nFft) * static_cast<double>(i) / sr);
    }
    const tagodsp::Matrix power = stft.powerSpectrum(x);
    // Pick a frame fully inside the signal so the zero pad plays no part.
    const std::size_t frame = 8;
    std::size_t peak = 0;
    for (std::size_t k = 1; k < power.cols; ++k) {
        if (power.at(frame, k) > power.at(frame, peak)) {
            peak = k;
        }
    }
    REQUIRE(peak == bin);
    // Hann spreads a bin-centered tone over exactly three bins; everything two
    // bins out is far below the peak.
    REQUIRE(power.at(frame, bin - 3) < power.at(frame, peak) * 1e-6);
    REQUIRE(power.at(frame, bin + 3) < power.at(frame, peak) * 1e-6);
}
