// Value diff of the C++ masking core against the Python workbench.
//
// The two are separate implementations, not bindings, so nothing but a diff
// over real numbers keeps them equal. `examples/masking_golden.py` writes
// cpp/tests/golden/masking.txt; this test rebuilds the same fixture from the
// spec in that script's docstring and compares every stage: raw STFT rows, the
// band-power grids, the contention cells, the zones, and the conflict summary.
//
// The comparison is a tolerance, not bit equality, because the two sides reach
// sin() through different libraries. The test reports the largest deviation it
// actually saw, so "green" carries a number instead of a claim. Zone and
// conflict counts are asserted separately: a tolerance can hide a value drift,
// but it cannot hide a threshold decision that flipped.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include <tagodsp/masking.hpp>

using Catch::Matchers::WithinAbs;

namespace {

constexpr double kSr = 16000.0;
constexpr std::size_t kNSamples = 32000;
constexpr std::size_t kNFft = 1024;
constexpr std::size_t kHop = 256;
constexpr std::size_t kNBands = 24;
constexpr double kFLo = 40.0;
constexpr double kFHi = 8000.0;
constexpr double kWindowSeconds = 0.1;
constexpr std::uint64_t kLcgSeed = 12345;

/// Uniform noise in [-1, 1) from the 64-bit LCG the golden script specifies.
std::vector<double> lcgNoise(std::size_t n) {
    std::vector<double> out(n);
    std::uint64_t state = kLcgSeed;
    for (std::size_t i = 0; i < n; ++i) {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        out[i] = static_cast<double>(state >> 11) * 0x1p-53 * 2.0 - 1.0;
    }
    return out;
}

bool gateOn(double t, const std::vector<std::pair<double, double>>& spans) {
    for (const auto& [lo, hi] : spans) {
        if (t >= lo && t < hi) {
            return true;
        }
    }
    return false;
}

std::vector<tagodsp::Track> fixture() {
    std::vector<double> keys(kNSamples), pad(kNSamples);
    const std::vector<std::pair<double, double>> keysGate{{0.0, 0.8}, {1.2, 2.0}};
    const std::vector<std::pair<double, double>> padGate{{0.3, 1.5}};
    for (std::size_t i = 0; i < kNSamples; ++i) {
        const double t = static_cast<double>(i) / kSr;
        keys[i] = gateOn(t, keysGate) ? 0.5 * std::sin(2.0 * tagodsp::kPi * 255.0 * t) +
                                            0.5 * std::sin(2.0 * tagodsp::kPi * 975.0 * t)
                                      : 0.0;
        pad[i] = gateOn(t, padGate) ? 0.4 * std::sin(2.0 * tagodsp::kPi * 265.0 * t) +
                                          0.5 * std::sin(2.0 * tagodsp::kPi * 985.0 * t)
                                    : 0.0;
    }
    std::vector<double> noise = lcgNoise(kNSamples);
    for (double& v : noise) {
        v *= 0.05;
    }
    return {tagodsp::Track{"keys", keys}, tagodsp::Track{"pad", pad},
            tagodsp::Track{"noise", noise}};
}

struct ZoneRow {
    std::string a, b;
    int band;
    double freqLo, freqHi;
    int windowLo, windowHi;
    double score;
};

struct ConflictRow {
    std::string a, b;
    int band;
    double freqLo, freqHi;
    int windowLo, windowHi;
    int activeWindows, occurrences;
    double score;
};

struct Golden {
    std::map<std::string, double> scalars;
    std::vector<double> bandEdges;
    std::map<int, std::vector<double>> stftRows;
    std::map<std::string, std::vector<double>> bandPower;
    std::map<std::string, std::vector<double>> cells;
    std::vector<ZoneRow> zones;
    std::vector<ConflictRow> conflicts;
};

std::vector<double> readRest(std::istringstream& in) {
    std::vector<double> values;
    double v = 0.0;
    while (in >> v) {
        values.push_back(v);
    }
    return values;
}

Golden loadGolden(const std::string& path) {
    std::ifstream file(path);
    REQUIRE(file.is_open());
    Golden g;
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }
        std::istringstream in(line);
        std::string key;
        in >> key;
        if (key == "tracks") {
            continue;
        }
        if (key == "band_edges") {
            g.bandEdges = readRest(in);
        } else if (key == "stft_power_row") {
            std::string name;
            int frame = 0;
            in >> name >> frame;
            g.stftRows[frame] = readRest(in);
        } else if (key == "band_power") {
            std::string name;
            in >> name;
            g.bandPower[name] = readRest(in);
        } else if (key == "cells") {
            std::string a, b;
            in >> a >> b;
            g.cells[a + "|" + b] = readRest(in);
        } else if (key == "zone") {
            ZoneRow z;
            in >> z.a >> z.b >> z.band >> z.freqLo >> z.freqHi >> z.windowLo >> z.windowHi >>
                z.score;
            g.zones.push_back(z);
        } else if (key == "conflict") {
            ConflictRow c;
            in >> c.a >> c.b >> c.band >> c.freqLo >> c.freqHi >> c.windowLo >> c.windowHi >>
                c.activeWindows >> c.occurrences >> c.score;
            g.conflicts.push_back(c);
        } else {
            double value = 0.0;
            in >> value;
            g.scalars[key] = value;
        }
    }
    return g;
}

/// Largest deviation between two runs of numbers, as a fraction of the largest
/// value in the reference run.
///
/// Deliberately not a per-element relative error. Power spectra contain bins
/// sitting in a spectral null, six or more orders of magnitude under the peak,
/// where the value is the residue of a near-total cancellation and a relative
/// comparison measures rounding noise rather than agreement. Relating every
/// deviation to the loudest value in the same run asks the question that
/// matters: can any bin be wrong by an amount that is audible next to the
/// signal it belongs to.
double maxDeviation(const std::vector<double>& got, const std::vector<double>& expected) {
    REQUIRE(got.size() == expected.size());
    double scale = 0.0;
    for (const double v : expected) {
        scale = std::max(scale, std::abs(v));
    }
    if (scale <= 0.0) {
        scale = 1.0;
    }
    double worst = 0.0;
    for (std::size_t i = 0; i < got.size(); ++i) {
        worst = std::max(worst, std::abs(got[i] - expected[i]) / scale);
    }
    return worst;
}

} // namespace

TEST_CASE("masking core matches the Python golden") {
    const Golden golden = loadGolden(std::string(TAGODSP_GOLDEN_DIR) + "/masking.txt");

    // The golden must describe the fixture this test builds, or the diff below
    // would compare two different experiments and still pass.
    REQUIRE(golden.scalars.at("sr") == kSr);
    REQUIRE(golden.scalars.at("n_samples") == static_cast<double>(kNSamples));
    REQUIRE(golden.scalars.at("n_fft") == static_cast<double>(kNFft));
    REQUIRE(golden.scalars.at("hop") == static_cast<double>(kHop));
    REQUIRE(golden.scalars.at("n_bands") == static_cast<double>(kNBands));

    tagodsp::MaskingDetector::Params p;
    p.sr = kSr;
    p.nBands = kNBands;
    p.fLo = kFLo;
    p.fHi = kFHi;
    p.windowSeconds = kWindowSeconds;
    p.gap = static_cast<int>(golden.scalars.at("gap"));
    p.minLen = static_cast<int>(golden.scalars.at("min_len"));
    p.contentionHitDb = golden.scalars.at("contention_hit_db");
    p.contentionScoreDb = golden.scalars.at("contention_score_db");
    p.nFft = kNFft;
    p.hop = kHop;
    const tagodsp::MaskingDetector detector(p);

    const std::vector<tagodsp::Track> tracks = fixture();

    SECTION("band edges") {
        REQUIRE(maxDeviation(detector.bandEdges(), golden.bandEdges) < 1e-12);
    }

    SECTION("raw stft rows") {
        const tagodsp::Stft stft(kNFft, kHop);
        const tagodsp::Matrix power = stft.powerSpectrum(tracks[0].x);
        double worst = 0.0;
        for (const auto& [frame, expected] : golden.stftRows) {
            std::vector<double> row(power.cols);
            for (std::size_t k = 0; k < power.cols; ++k) {
                row[k] = power.at(static_cast<std::size_t>(frame), k);
            }
            worst = std::max(worst, maxDeviation(row, expected));
        }
        std::cout << "stft rows, max deviation vs frame peak: " << worst << "\n";
        REQUIRE(worst < 1e-12);
    }

    const tagodsp::MaskingResult result = detector.analyze(tracks);

    SECTION("window count") {
        REQUIRE(result.nWindows == static_cast<int>(golden.scalars.at("n_windows")));
    }

    SECTION("band power grids") {
        double worst = 0.0;
        for (const auto& tg : result.bandPower) {
            const auto& expected = golden.bandPower.at(tg.name);
            worst = std::max(worst, maxDeviation(tg.grid.data, expected));
        }
        std::cout << "band power, max deviation vs grid peak: " << worst << "\n";
        REQUIRE(worst < 1e-12);
    }

    SECTION("contention cells") {
        double worst = 0.0;
        for (const auto& pg : result.cells) {
            const auto& expected = golden.cells.at(pg.trackA + "|" + pg.trackB);
            // These are dB, so compare absolutely: a relative test near the
            // -120 dB floor would be meaningless.
            REQUIRE(pg.grid.data.size() == expected.size());
            for (std::size_t i = 0; i < expected.size(); ++i) {
                worst = std::max(worst, std::abs(pg.grid.data[i] - expected[i]));
            }
        }
        std::cout << "contention cells, max absolute deviation: " << worst << " dB\n";
        REQUIRE(worst < 1e-9);
    }

    SECTION("zones") {
        REQUIRE(result.zones.size() == golden.zones.size());
        for (std::size_t i = 0; i < result.zones.size(); ++i) {
            const auto& got = result.zones[i];
            const auto& want = golden.zones[i];
            INFO("zone " << i);
            REQUIRE(got.trackA == want.a);
            REQUIRE(got.trackB == want.b);
            REQUIRE(got.band == want.band);
            REQUIRE(got.windowLo == want.windowLo);
            REQUIRE(got.windowHi == want.windowHi);
            REQUIRE_THAT(got.freqLoHz, WithinAbs(want.freqLo, 1e-9));
            REQUIRE_THAT(got.freqHiHz, WithinAbs(want.freqHi, 1e-9));
            REQUIRE_THAT(got.score, WithinAbs(want.score, 1e-9));
        }
    }

    SECTION("conflict summary") {
        REQUIRE(result.conflicts.size() == golden.conflicts.size());
        int members = 0;
        for (std::size_t i = 0; i < result.conflicts.size(); ++i) {
            const auto& got = result.conflicts[i];
            const auto& want = golden.conflicts[i];
            INFO("conflict " << i);
            REQUIRE(got.trackA == want.a);
            REQUIRE(got.trackB == want.b);
            REQUIRE(got.band == want.band);
            REQUIRE(got.windowLo == want.windowLo);
            REQUIRE(got.windowHi == want.windowHi);
            REQUIRE(got.activeWindows == want.activeWindows);
            REQUIRE(got.occurrences == want.occurrences);
            REQUIRE_THAT(got.freqLoHz, WithinAbs(want.freqLo, 1e-9));
            REQUIRE_THAT(got.freqHiHz, WithinAbs(want.freqHi, 1e-9));
            REQUIRE_THAT(got.score, WithinAbs(want.score, 1e-9));
            REQUIRE(got.zones.size() == static_cast<std::size_t>(got.occurrences));
            members += got.occurrences;
        }
        // The summary layer is additive: every zone belongs to exactly one
        // conflict, none are invented and none are dropped.
        REQUIRE(members == static_cast<int>(result.zones.size()));
    }
}

TEST_CASE("masking detector rejects contradictory parameters") {
    // Params carries no sample rate default, so a bare Params is already invalid.
    REQUIRE_THROWS_AS(tagodsp::MaskingDetector(tagodsp::MaskingDetector::Params{}),
                      std::invalid_argument);

    const auto withSr = [] {
        tagodsp::MaskingDetector::Params p;
        p.sr = kSr;
        return p;
    };
    REQUIRE_NOTHROW(tagodsp::MaskingDetector(withSr()));

    auto p = withSr();
    p.contentionScoreDb = -1.0; // above the exact ceiling
    REQUIRE_THROWS_AS(tagodsp::MaskingDetector(p), std::invalid_argument);

    p = withSr();
    p.contentionHitDb = -5.0;
    p.contentionScoreDb = -9.75; // a region could never reach a score its cells cannot have
    REQUIRE_THROWS_AS(tagodsp::MaskingDetector(p), std::invalid_argument);

    p = withSr();
    p.fLo = 20000.0;
    p.fHi = 20.0;
    REQUIRE_THROWS_AS(tagodsp::MaskingDetector(p), std::invalid_argument);

    p = withSr();
    p.nBands = 1;
    REQUIRE_THROWS_AS(tagodsp::MaskingDetector(p), std::invalid_argument);
}

TEST_CASE("silence produces no zones") {
    tagodsp::MaskingDetector::Params p;
    p.sr = kSr;
    p.nFft = kNFft;
    p.hop = kHop;
    const tagodsp::MaskingDetector detector(p);
    const std::vector<tagodsp::Track> tracks{
        tagodsp::Track{"a", std::vector<double>(kNSamples, 0.0)},
        tagodsp::Track{"b", std::vector<double>(kNSamples, 0.0)},
    };
    const tagodsp::MaskingResult result = detector.analyze(tracks);
    REQUIRE(result.zones.empty());
    REQUIRE(result.conflicts.empty());
}
