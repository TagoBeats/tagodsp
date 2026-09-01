#pragma once

// Frequency masking (conflict) detection between two or more tracks.
//
// Port of the validated part of `tagodsp.analysis.masking`. The Python module
// carries three scorings; only "contention" is here, and that is a decision,
// not an omission:
//
//   - "relative" is the X-Ray prototype's shipped detector and was refuted on
//     2026-08-31. Over a 60 dB synthetic level sweep it returns the same score
//     (0.959 from -30 to +30 dB): it measures "both tracks have energy here",
//     not masking. Porting a refuted detector into the product core would only
//     create a second way to be wrong.
//   - "collision" was the intermediate candidate, superseded by "contention"
//     before the listening tests.
//
// Both stay in the Python workbench, which is where candidates belong.
//
// Pipeline, per track:
//   1. STFT (default n_fft = 4096, hop = 1024) of the mono signal
//   2. power spectrum |S|^2
//   3. bins mapped to `nBands` log-spaced bands between fLo and fHi
//      (edges = geomspace(fLo, fHi, nBands + 1)), power per band is the sum of
//      its bins
//   4. frames aggregated into time windows of `windowSeconds`; the value per
//      (window, band) is the mean of the frames falling into it
//   5. `nWindows` is shared across all tracks, sized to the longest one, so
//      every band-power grid has the same shape and holds absolute power
//
// Scoring, per pair:
//   mixBb[w] = sum over bands of a[w, k] + b[w, k], for the two tracks compared
//   share    = min(a, b) / mixBb[w]              (0 if mixBb[w] <= 0)
//   cell     = 10 * log10(max(share, 1e-12))     in dB, NOT normalized
//
// The reference is the pair, not the full mix, because the target product is a
// plugin with a sidechain input: it only ever sees two signals. A full-mix
// reference made a pair's zones depend on how many other tracks happened to be
// playing. Accepted cost: two quiet tracks contending with each other still
// score even when both sit under the rest of the mix.
//
// min(a, b) is its own audibility gate. One track much quieter than the other,
// both quiet, or one silent all drive share towards zero without a separate
// threshold.
//
// kContentionCeilDb = -3.0: with the pair as reference, min(a, b) can be at
// most half of mixBb, so the bound is exact (-3.01 dB) rather than an estimate.
// Measured over the corpus (51 beats, 2026-09-01) the highest cell anywhere
// reaches -3.13 dB, so the top of the scale is real material, not headroom.
//
// Thresholds and display scale are separate, and this is load-bearing for this
// port: clustering decides on the dB value directly, so a change to the display
// scale cannot move the detection, and C++ inherits no hidden mapping. The
// reported Zone::score is normalized afterwards over
// [contentionScoreDb, kContentionCeilDb]. The defaults -11.1 / -9.75 dB are the
// exact dB equivalents of the old hit_min = 0.7 / score_min = 0.75 pair on the
// -30 dB scale, so the reported zone set is the one Phase 0 validated by ear.
//
// On top of the zones sits summarizeConflicts(): a reported pair breaks into
// median 3 and up to 64 zones, and measured, that split runs along time rather
// than frequency (58.8 % of the zone relations inside a pair overlap in
// frequency and only sit apart in time; a genuine second frequency region at
// the same time is 1.9 %). The layer merges a pair's zones by overlapping
// frequency range and turns time into a count. It is purely additive: `zones`
// is untouched and every Conflict keeps its members.
//
// Source of truth for the numbers: docs/concepts/masking_detector.md and
// docs/measurements/masking_*_2026-09-01.md.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <deque>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include <tagodsp/matrix.hpp>
#include <tagodsp/stft.hpp>

namespace tagodsp {

/// Exact upper bound of min(a, b) / mixBb with the pair as reference: -3.01 dB.
inline constexpr double kContentionCeilDb = -3.0;

struct Track {
    std::string name;
    std::vector<double> x; ///< mono; the caller mono-sums, a plugin gets channels anyway
};

struct Zone {
    std::string trackA;
    std::string trackB;
    int band = 0;
    double freqLoHz = 0.0;
    double freqHiHz = 0.0;
    int windowLo = 0;
    int windowHi = 0;
    double score = 0.0; ///< always in [0, 1]
};

/// One recurring dispute between two tracks: a frequency region, N occurrences.
///
/// A summary over Zones, not a replacement for them. `score` is the maximum
/// over the members, not the mean: it is the worst moment of the dispute, and
/// the only aggregate that does not move when the same material fragments into
/// more or fewer zones. Persistence stays a separate axis (`occurrences`,
/// `activeWindows`) so a UI can rank by severity or by how much of the track is
/// affected without mixing the two into one number.
struct Conflict {
    std::string trackA;
    std::string trackB;
    int band = 0;          ///< the band the conflict spends the most window-time in
    double freqLoHz = 0.0; ///< union over the member zones
    double freqHiHz = 0.0;
    int windowLo = 0; ///< first and last window the conflict touches
    int windowHi = 0;
    int activeWindows = 0; ///< windows actually covered, gaps between occurrences excluded
    int occurrences = 0;   ///< number of member zones
    double score = 0.0;
    std::vector<Zone> zones; ///< the members, for drill-down
};

struct TrackGrid {
    std::string name;
    Matrix grid;
};

struct PairGrid {
    std::string trackA;
    std::string trackB;
    Matrix grid;
};

struct MaskingResult {
    std::vector<Zone> zones;
    std::vector<double> bandEdgesHz;
    int nWindows = 0;
    double windowSeconds = 0.0;
    /// Per track, (nWindows x nBands), absolute power.
    std::vector<TrackGrid> bandPower;
    /// Per pair, (nWindows x nBands), the dB values clustering ran on.
    std::vector<PairGrid> cells;
    /// `zones` grouped into recurring disputes. Purely additive.
    std::vector<Conflict> conflicts;
};

namespace detail {

/// numpy.geomspace(lo, hi, n + 1): log-spaced, endpoints set exactly.
inline std::vector<double> geomspace(double lo, double hi, std::size_t nEdges) {
    std::vector<double> edges(nEdges);
    const double logLo = std::log10(lo);
    const double logHi = std::log10(hi);
    const double step = (logHi - logLo) / static_cast<double>(nEdges - 1);
    for (std::size_t i = 0; i < nEdges; ++i) {
        edges[i] = std::pow(10.0, logLo + step * static_cast<double>(i));
    }
    edges.front() = lo;
    edges.back() = hi;
    return edges;
}

/// numpy.digitize(value, edges) - 1, i.e. the index of the band containing
/// `value`, or out of range when it falls below the first or above the last
/// edge. Those cells are dropped by the caller.
inline std::ptrdiff_t bandIndex(double value, const std::vector<double>& edges) {
    const auto it = std::upper_bound(edges.begin(), edges.end(), value);
    return std::distance(edges.begin(), it) - 1;
}

/// Python's round(): ties go to the even neighbour.
///
/// Spelled out rather than delegated to std::nearbyint, which rounds according
/// to the ambient floating-point mode. That mode is half-to-even by default, but
/// it is settable at runtime and -ffast-math may not respect it, so relying on
/// it would make the agreement with Python a documentary claim instead of a
/// property of this code.
inline long long roundHalfEven(double v) {
    const double lower = std::floor(v);
    const double frac = v - lower;
    const auto n = static_cast<long long>(lower);
    if (frac > 0.5) {
        return n + 1;
    }
    if (frac < 0.5) {
        return n;
    }
    return (n % 2 == 0) ? n : n + 1;
}

} // namespace detail

/// Group a pair's zones into recurring conflicts, merging over frequency overlap.
///
/// Over the corpus this takes 1222 zones down to 289 conflicts: median 1 per
/// reported pair, 83.1 % of the pairs to a single conflict, at most 3.
///
/// The interval merge is transitive and can chain zones that do not overlap
/// each other (A 100-200, B 190-400, C 390-800). Measured, that happens in 33
/// of 289 groups and widens the reported range by at most 0.66 octaves over the
/// widest member, median 0.00. Accepted: the alternative is reporting one
/// dispute as two entries whose ranges touch.
inline std::vector<Conflict> summarizeConflicts(const std::vector<Zone>& zones) {
    // Pairs in first-seen order, mirroring Python's dict insertion order so the
    // output order is identical for equal scores.
    std::vector<std::pair<std::string, std::string>> pairOrder;
    std::vector<std::vector<Zone>> byPair;
    std::unordered_map<std::string, std::size_t> indexOfPair;
    for (const Zone& z : zones) {
        // pairOrder carries the first-seen order the output depends on; the map
        // only answers "have I seen this pair" without rescanning all of them.
        const std::string key = z.trackA + '\0' + z.trackB;
        auto [it, inserted] = indexOfPair.try_emplace(key, pairOrder.size());
        if (inserted) {
            pairOrder.emplace_back(z.trackA, z.trackB);
            byPair.emplace_back();
        }
        byPair[it->second].push_back(z);
    }

    std::vector<Conflict> conflicts;
    for (std::size_t p = 0; p < pairOrder.size(); ++p) {
        std::vector<Zone> members = byPair[p];
        std::stable_sort(members.begin(), members.end(), [](const Zone& a, const Zone& b) {
            if (a.freqLoHz != b.freqLoHz) {
                return a.freqLoHz < b.freqLoHz;
            }
            return a.freqHiHz < b.freqHiHz;
        });

        // Sweep low to high, merging while the next zone starts below the reach
        // of the group built so far.
        std::vector<std::vector<Zone>> groups;
        std::vector<Zone> current;
        double reach = 0.0;
        for (const Zone& z : members) {
            if (!current.empty() && z.freqLoHz < reach) {
                current.push_back(z);
                reach = std::max(reach, z.freqHiHz);
            } else {
                if (!current.empty()) {
                    groups.push_back(current);
                }
                current.assign(1, z);
                reach = z.freqHiHz;
            }
        }
        if (!current.empty()) {
            groups.push_back(current);
        }

        for (std::vector<Zone>& group : groups) {
            Conflict c;
            c.trackA = pairOrder[p].first;
            c.trackB = pairOrder[p].second;
            c.occurrences = static_cast<int>(group.size());
            c.freqLoHz = group.front().freqLoHz;
            c.freqHiHz = group.front().freqHiHz;
            c.windowLo = group.front().windowLo;
            c.windowHi = group.front().windowHi;
            c.score = group.front().score;
            for (const Zone& z : group) {
                c.freqLoHz = std::min(c.freqLoHz, z.freqLoHz);
                c.freqHiHz = std::max(c.freqHiHz, z.freqHiHz);
                c.windowLo = std::min(c.windowLo, z.windowLo);
                c.windowHi = std::max(c.windowHi, z.windowHi);
                c.score = std::max(c.score, z.score);
            }

            std::vector<char> covered(static_cast<std::size_t>(c.windowHi - c.windowLo + 1), 0);
            for (const Zone& z : group) {
                for (int w = z.windowLo; w <= z.windowHi; ++w) {
                    covered[static_cast<std::size_t>(w - c.windowLo)] = 1;
                }
            }
            c.activeWindows = static_cast<int>(std::count(covered.begin(), covered.end(), 1));

            // Dominant band weighted by window-time, not by zone count: a band
            // named by one long zone beats one named by two single-window ones.
            // Tallied into a band-indexed vector and scanned upwards with a
            // strict >, so ties fall to the lower band index on their own.
            int maxBand = 0;
            for (const Zone& z : group) {
                maxBand = std::max(maxBand, z.band);
            }
            std::vector<int> timePerBand(static_cast<std::size_t>(maxBand) + 1, 0);
            for (const Zone& z : group) {
                timePerBand[static_cast<std::size_t>(z.band)] += z.windowHi - z.windowLo + 1;
            }
            c.band = static_cast<int>(std::distance(
                timePerBand.begin(), std::max_element(timePerBand.begin(), timePerBand.end())));

            c.zones = group;
            std::stable_sort(c.zones.begin(), c.zones.end(), [](const Zone& a, const Zone& b) {
                if (a.windowLo != b.windowLo) {
                    return a.windowLo < b.windowLo;
                }
                return a.band < b.band;
            });
            conflicts.push_back(std::move(c));
        }
    }

    std::stable_sort(conflicts.begin(), conflicts.end(),
                     [](const Conflict& a, const Conflict& b) { return a.score > b.score; });
    return conflicts;
}

/// Offline analyzer: construct with Params, then call analyze() on whole
/// buffers. Deliberately not the prepare()/process()/reset() lifecycle
/// CONTRIBUTING prescribes, and it allocates and throws, which that document
/// rules out "in the process path". This is not a process path: it is batch
/// analysis over complete tracks. The streaming shape (block state, a lock-free
/// ring buffer, FFT and clustering off the audio thread) belongs to the plugin
/// that will wrap this, not to the core.
class MaskingDetector {
public:
    struct Params {
        /// No default on purpose: every other field here has a measured or
        /// derived default, but a silently assumed sample rate would mis-scale
        /// every band edge and window without saying so.
        double sr = 0.0;
        std::size_t nBands = 30;
        double fLo = 20.0;
        double fHi = 20000.0;
        double windowSeconds = 0.1;
        int gap = 2;
        int minLen = 3;
        double contentionHitDb = -11.1;
        double contentionScoreDb = -9.75;
        std::size_t nFft = 4096;
        std::size_t hop = 1024;
    };

    explicit MaskingDetector(const Params& p) : p_(p), stft_(p.nFft, p.hop) {
        if (p_.contentionScoreDb >= kContentionCeilDb) {
            throw std::invalid_argument("contentionScoreDb must be < kContentionCeilDb");
        }
        if (p_.contentionHitDb > p_.contentionScoreDb) {
            // Otherwise a region could never reach a score its own cells cannot have.
            throw std::invalid_argument("contentionHitDb must be <= contentionScoreDb");
        }
        if (p_.nBands < 2) {
            throw std::invalid_argument("nBands must be >= 2");
        }
        if (!(p_.fLo > 0.0 && p_.fLo < p_.fHi)) {
            throw std::invalid_argument("need 0 < fLo < fHi");
        }
        if (p_.windowSeconds <= 0.0) {
            throw std::invalid_argument("windowSeconds must be > 0");
        }
        if (p_.gap < 0) {
            throw std::invalid_argument("gap must be >= 0");
        }
        if (p_.minLen < 1) {
            throw std::invalid_argument("minLen must be >= 1");
        }
        if (p_.sr <= 0.0) {
            throw std::invalid_argument("sr must be set and > 0");
        }
        // Everything below depends only on Params, never on the track being
        // analyzed, so it is built once here instead of once per track.
        bandEdges_ = detail::geomspace(p_.fLo, p_.fHi, p_.nBands + 1);
        const std::vector<double> freqs = stft_.freqs(p_.sr);
        binBand_.resize(freqs.size());
        for (std::size_t k = 0; k < freqs.size(); ++k) {
            binBand_[k] = detail::bandIndex(freqs[k], bandEdges_);
        }
    }

    const Params& params() const noexcept { return p_; }

    const std::vector<double>& bandEdges() const noexcept { return bandEdges_; }

    /// Band-power grid of one track, (nWindows x nBands), absolute power.
    Matrix bandPower(const std::vector<double>& x, int nWindows) const {
        return foldFrames(framePower(x), nWindows);
    }

    /// Band power of one frame, bins summed inside a band, written into
    /// `out[0, nBands)`. No window averaging yet.
    void bandPowerFrame(const std::vector<double>& x, std::size_t f, double* out) const {
        spectrumScratch_.resize(stft_.nBins());
        stft_.powerSpectrumFrame(x, f, spectrumScratch_.data());
        std::fill(out, out + p_.nBands, 0.0);
        foldBins(spectrumScratch_.data(), out);
    }

    /// Band power of one already positioned frame of `stft().nFft()` samples.
    ///
    /// The streaming counterpart of bandPowerFrame(): a live caller holds a ring
    /// of recent samples rather than the signal, so it positions the frame
    /// itself and the window and the transform still happen in here.
    void bandPowerOfSamples(const double* samples, double* out) const {
        spectrumScratch_.resize(stft_.nBins());
        stft_.powerSpectrumOfSamples(samples, spectrumScratch_.data());
        std::fill(out, out + p_.nBands, 0.0);
        foldBins(spectrumScratch_.data(), out);
    }

    const Stft& stft() const noexcept { return stft_; }

    /// Per-frame band power of a signal, (frames x nBands).
    ///
    /// This is the representation a live caller wants to hold on to. A frame is
    /// nBands doubles, 30 by default, against 2049 bins of spectrum and 1024
    /// samples of audio per hop: about 10 KB per second of audio, so a whole
    /// song is megabytes. Keeping this instead of the samples is what lets an
    /// analysis follow playback without recomputing the past.
    Matrix framePower(const std::vector<double>& x) const {
        const std::size_t nFrames = stft_.frameCount(x.size());
        Matrix out(nFrames, p_.nBands);
        if (nFrames == 0) {
            return out;
        }
        spectrumScratch_.resize(stft_.nBins());
        for (std::size_t f = 0; f < nFrames; ++f) {
            stft_.powerSpectrumFrame(x, f, spectrumScratch_.data());
            foldBins(spectrumScratch_.data(), &out.at(f, 0));
        }
        return out;
    }

    /// The number of windows a given frame count spans.
    int windowCountForFrames(std::size_t frames) const {
        const double maxTime =
            frames > 0 ? static_cast<double>(frames - 1) * static_cast<double>(p_.hop) / p_.sr : 0.0;
        return std::max<int>(1,
                             static_cast<int>(detail::roundHalfEven(maxTime / p_.windowSeconds)));
    }

    /// Everything downstream of the transform, run on per-frame band power.
    ///
    /// A live caller keeps its own frame rows and calls this; analyze() builds
    /// the rows first and then calls the same function, so there is one code
    /// path from here on and no second way for the two to disagree.
    MaskingResult analyzeFramePower(const std::vector<TrackGrid>& framePowers) const {
        MaskingResult result;
        result.bandEdgesHz = bandEdges();
        result.windowSeconds = p_.windowSeconds;

        std::size_t maxFrames = 0;
        for (const TrackGrid& g : framePowers) {
            maxFrames = std::max(maxFrames, g.grid.rows);
        }
        const int nWindows = windowCountForFrames(maxFrames);
        result.nWindows = nWindows;

        for (const TrackGrid& g : framePowers) {
            result.bandPower.push_back(TrackGrid{g.name, foldFrames(g.grid, nWindows)});
        }

        for (std::size_t i = 0; i < framePowers.size(); ++i) {
            for (std::size_t j = i + 1; j < framePowers.size(); ++j) {
                const Matrix& a = result.bandPower[i].grid;
                const Matrix& b = result.bandPower[j].grid;
                Matrix cell = contention(a, b);
                std::vector<Zone> found =
                    cluster(cell, result.bandEdgesHz, framePowers[i].name, framePowers[j].name);
                result.zones.insert(result.zones.end(), found.begin(), found.end());
                result.cells.push_back(
                    PairGrid{framePowers[i].name, framePowers[j].name, std::move(cell)});
            }
        }

        std::stable_sort(result.zones.begin(), result.zones.end(),
                         [](const Zone& x, const Zone& y) { return x.score > y.score; });
        result.conflicts = summarizeConflicts(result.zones);
        return result;
    }

    MaskingResult analyze(const std::vector<Track>& tracks) const {
        std::vector<TrackGrid> framePowers;
        framePowers.reserve(tracks.size());
        for (const Track& t : tracks) {
            framePowers.push_back(TrackGrid{t.name, framePower(t.x)});
        }
        return analyzeFramePower(framePowers);
    }

private:
    /// One spectrum's bins summed into bands, ascending bin order. Bands with no
    /// bins stay at whatever the caller left in `out`, which is zero.
    void foldBins(const double* spectrum, double* out) const {
        for (std::size_t k = 0; k < binBand_.size(); ++k) {
            const std::ptrdiff_t b = binBand_[k];
            if (b >= 0 && b < static_cast<std::ptrdiff_t>(p_.nBands)) {
                out[static_cast<std::size_t>(b)] += spectrum[k];
            }
        }
    }

    /// Per-frame band power (frames x nBands) folded into (nWindows x nBands),
    /// frames averaged inside a window. Windows no frame falls into stay at
    /// zero.
    ///
    /// The arithmetic here is deliberately untouched from when this was one
    /// function taking the full spectrum: frames are still added in ascending
    /// order inside a window and the division by the frame count still happens
    /// last. Floating point addition is not associative, so any other order
    /// would move the values in the last bits and the golden diff against the
    /// Python workbench would have to be loosened to hide it.
    Matrix foldFrames(const Matrix& framePowers, int nWindows) const {
        Matrix out(static_cast<std::size_t>(nWindows), p_.nBands);
        if (framePowers.rows == 0) {
            return out;
        }
        std::vector<int> counts(static_cast<std::size_t>(nWindows), 0);
        for (std::size_t f = 0; f < framePowers.rows; ++f) {
            const double t = static_cast<double>(f) * static_cast<double>(p_.hop) / p_.sr;
            const auto w = std::min<std::size_t>(static_cast<std::size_t>(t / p_.windowSeconds),
                                                 static_cast<std::size_t>(nWindows - 1));
            counts[w] += 1;
            for (std::size_t b = 0; b < p_.nBands; ++b) {
                out.at(w, b) += framePowers.at(f, b);
            }
        }
        for (std::size_t w = 0; w < out.rows; ++w) {
            if (counts[w] == 0) {
                continue;
            }
            for (std::size_t b = 0; b < p_.nBands; ++b) {
                out.at(w, b) /= static_cast<double>(counts[w]);
            }
        }
        return out;
    }

    Matrix contention(const Matrix& a, const Matrix& b) const {
        Matrix cell(a.rows, a.cols);
        for (std::size_t w = 0; w < a.rows; ++w) {
            double mixBb = 0.0;
            for (std::size_t k = 0; k < a.cols; ++k) {
                mixBb += a.at(w, k) + b.at(w, k);
            }
            for (std::size_t k = 0; k < a.cols; ++k) {
                const double share =
                    mixBb > 0.0 ? std::min(a.at(w, k), b.at(w, k)) / mixBb : 0.0;
                cell.at(w, k) = 10.0 * std::log10(std::max(share, 1e-12));
            }
        }
        return cell;
    }

    /// Map a surviving region value in dB to the reported score in [0, 1]. The
    /// reporting threshold displays as 0.0: it is the least bad thing still
    /// worth showing, not "nothing".
    double zoneScore(double valueDb) const {
        const double span = kContentionCeilDb - p_.contentionScoreDb;
        return std::clamp((valueDb - p_.contentionScoreDb) / span, 0.0, 1.0);
    }

    /// BFS over cells at or above the hit threshold (time gaps up to `gap`
    /// windows, band adjacency of 1), regions shorter than `minLen` windows
    /// dropped, region value = 90th percentile over its cells, regions below
    /// the score threshold dropped.
    std::vector<Zone> cluster(const Matrix& cell, const std::vector<double>& edges,
                              const std::string& nameA, const std::string& nameB) const {
        const auto nWindows = static_cast<std::ptrdiff_t>(cell.rows);
        const auto nBands = static_cast<std::ptrdiff_t>(cell.cols);
        std::vector<char> hit(cell.data.size(), 0);
        for (std::size_t i = 0; i < cell.data.size(); ++i) {
            hit[i] = cell.data[i] >= p_.contentionHitDb ? 1 : 0;
        }
        std::vector<char> seen(cell.data.size(), 0);
        std::vector<Zone> zones;

        // Seed order is row-major (window, then band), matching np.argwhere, so
        // zones that tie on score keep the same order as in Python.
        for (std::ptrdiff_t w0 = 0; w0 < nWindows; ++w0) {
            for (std::ptrdiff_t k0 = 0; k0 < nBands; ++k0) {
                const std::size_t seed = static_cast<std::size_t>(w0 * nBands + k0);
                if (!hit[seed] || seen[seed]) {
                    continue;
                }
                seen[seed] = 1;
                std::deque<std::pair<std::ptrdiff_t, std::ptrdiff_t>> queue{{w0, k0}};
                std::vector<std::pair<std::ptrdiff_t, std::ptrdiff_t>> cells;
                while (!queue.empty()) {
                    const auto [w, k] = queue.front();
                    queue.pop_front();
                    cells.emplace_back(w, k);
                    for (std::ptrdiff_t dw = -p_.gap; dw <= p_.gap; ++dw) {
                        for (std::ptrdiff_t dk = -1; dk <= 1; ++dk) {
                            const std::ptrdiff_t w2 = w + dw;
                            const std::ptrdiff_t k2 = k + dk;
                            if (w2 < 0 || w2 >= nWindows || k2 < 0 || k2 >= nBands) {
                                continue;
                            }
                            const std::size_t idx = static_cast<std::size_t>(w2 * nBands + k2);
                            if (hit[idx] && !seen[idx]) {
                                seen[idx] = 1;
                                queue.emplace_back(w2, k2);
                            }
                        }
                    }
                }

                auto [wMinIt, wMaxIt] = std::minmax_element(
                    cells.begin(), cells.end(),
                    [](const auto& x, const auto& y) { return x.first < y.first; });
                const std::ptrdiff_t wMin = wMinIt->first;
                const std::ptrdiff_t wMax = wMaxIt->first;
                if (wMax - wMin + 1 < p_.minLen) {
                    continue;
                }

                std::vector<double> values;
                values.reserve(cells.size());
                for (const auto& [w, k] : cells) {
                    values.push_back(cell.at(static_cast<std::size_t>(w),
                                             static_cast<std::size_t>(k)));
                }
                std::sort(values.begin(), values.end());
                const auto idx =
                    std::min(static_cast<std::size_t>(static_cast<double>(values.size()) * 0.9),
                             values.size() - 1);
                const double value = values[idx];
                if (value < p_.contentionScoreDb) {
                    continue;
                }

                // Dominant band: the one contributing the most cells, ties
                // broken by their summed strength, then by which band the
                // region reached first. Strength is kept explicit so the choice
                // does not depend on where the zero of the cell scale sits,
                // which matters now that the cells are dB.
                std::vector<int> bandCount(cell.cols, 0);
                std::vector<double> bandStrength(cell.cols, 0.0);
                std::vector<std::size_t> bandFirstSeen(cell.cols, 0);
                std::size_t order = 0;
                std::ptrdiff_t kMin = cells.front().second;
                std::ptrdiff_t kMax = cells.front().second;
                for (const auto& [w, k] : cells) {
                    kMin = std::min(kMin, k);
                    kMax = std::max(kMax, k);
                    const auto band = static_cast<std::size_t>(k);
                    if (bandCount[band] == 0) {
                        bandFirstSeen[band] = order++;
                    }
                    bandCount[band] += 1;
                    bandStrength[band] +=
                        cell.at(static_cast<std::size_t>(w), static_cast<std::size_t>(k));
                }
                const auto rank = [&](std::size_t band) {
                    return std::make_tuple(bandCount[band], bandStrength[band],
                                           -static_cast<std::ptrdiff_t>(bandFirstSeen[band]));
                };
                std::size_t best = static_cast<std::size_t>(kMin);
                for (std::size_t band = 0; band < cell.cols; ++band) {
                    if (bandCount[band] > 0 && rank(band) > rank(best)) {
                        best = band;
                    }
                }

                Zone z;
                z.trackA = nameA;
                z.trackB = nameB;
                z.band = static_cast<int>(best);
                z.freqLoHz = edges[static_cast<std::size_t>(kMin)];
                z.freqHiHz = edges[static_cast<std::size_t>(kMax) + 1];
                z.windowLo = static_cast<int>(wMin);
                z.windowHi = static_cast<int>(wMax);
                z.score = zoneScore(value);
                zones.push_back(std::move(z));
            }
        }
        return zones;
    }

    Params p_;
    Stft stft_;
    /// Scratch, not state. Mutable for the same reason Stft holds its own: a
    /// frame has to be computable through a const reference. One detector is
    /// therefore not safe to share across threads.
    mutable std::vector<double> spectrumScratch_;
    std::vector<double> bandEdges_;
    std::vector<std::ptrdiff_t> binBand_; ///< bin index -> band index, or out of range
};

} // namespace tagodsp
