#pragma once

// Radix-2 Cooley-Tukey FFT, power-of-two sizes only.
//
// Deliberately dependency-free so the library stays header-only: a plugin that
// pulls in tagodsp should not have to bring an FFT library along. The transform
// is the textbook decimation-in-time form (Cooley & Tukey 1965; Oppenheim &
// Schafer, "Discrete-Time Signal Processing", ch. 9): bit-reversal permutation
// followed by log2(n) butterfly stages over precomputed twiddle factors.
//
// `Rfft` is the real-input wrapper the analysis code uses. It runs the full
// complex transform and keeps bins [0, n/2], which is what numpy's rfft
// returns. The half-size packing trick would be about twice as fast and is a
// known optimization, but this core runs offline over whole files, so the
// simpler form that is easy to verify against the Python workbench wins.

#include <cmath>
#include <complex>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace tagodsp {

inline constexpr double kPi = 3.14159265358979323846;

class Fft {
public:
    explicit Fft(std::size_t n) : n_(n) {
        if (n == 0 || (n & (n - 1)) != 0) {
            throw std::invalid_argument("Fft size must be a power of two");
        }
        reversal_.resize(n);
        std::size_t bits = 0;
        while ((std::size_t{1} << bits) < n) {
            ++bits;
        }
        for (std::size_t i = 0; i < n; ++i) {
            std::size_t r = 0;
            for (std::size_t b = 0; b < bits; ++b) {
                if (i & (std::size_t{1} << b)) {
                    r |= std::size_t{1} << (bits - 1 - b);
                }
            }
            reversal_[i] = r;
        }
        // twiddles_[k] = exp(-2*pi*i*k/n) for k < n/2, shared by every stage:
        // a stage of length m uses stride n/m into this table.
        twiddles_.resize(n / 2);
        for (std::size_t k = 0; k < n / 2; ++k) {
            const double angle = -2.0 * kPi * static_cast<double>(k) / static_cast<double>(n);
            twiddles_[k] = std::complex<double>(std::cos(angle), std::sin(angle));
        }
    }

    std::size_t size() const noexcept { return n_; }

    /// In-place forward DFT of `a`, which must hold exactly size() samples.
    void forward(std::vector<std::complex<double>>& a) const {
        if (a.size() != n_) {
            throw std::invalid_argument("Fft::forward expects exactly size() samples");
        }
        for (std::size_t i = 0; i < n_; ++i) {
            const std::size_t r = reversal_[i];
            if (i < r) {
                std::swap(a[i], a[r]);
            }
        }
        for (std::size_t m = 2; m <= n_; m <<= 1) {
            const std::size_t half = m / 2;
            const std::size_t stride = n_ / m;
            for (std::size_t start = 0; start < n_; start += m) {
                for (std::size_t k = 0; k < half; ++k) {
                    const std::complex<double> w = twiddles_[k * stride];
                    const std::complex<double> even = a[start + k];
                    const std::complex<double> odd = w * a[start + k + half];
                    a[start + k] = even + odd;
                    a[start + k + half] = even - odd;
                }
            }
        }
    }

private:
    std::size_t n_;
    std::vector<std::size_t> reversal_;
    std::vector<std::complex<double>> twiddles_;
};

/// Real-input FFT returning the n/2 + 1 non-redundant bins, like numpy.fft.rfft.
class Rfft {
public:
    explicit Rfft(std::size_t n) : fft_(n), n_(n), scratch_(n) {}

    std::size_t size() const noexcept { return n_; }
    std::size_t nBins() const noexcept { return n_ / 2 + 1; }

    /// Transform `x` (size() real samples) into `out` (nBins() complex bins).
    void forward(const double* x, std::complex<double>* out) const {
        for (std::size_t i = 0; i < n_; ++i) {
            scratch_[i] = std::complex<double>(x[i], 0.0);
        }
        fft_.forward(scratch_);
        for (std::size_t k = 0; k < nBins(); ++k) {
            out[k] = scratch_[k];
        }
    }

private:
    Fft fft_;
    std::size_t n_;
    // Per-call working buffer, written and read entirely within one forward().
    // Mutable so the precomputed tables in fft_ do not have to be copied just
    // to obtain a non-const handle on this scratch space.
    mutable std::vector<std::complex<double>> scratch_;
};

} // namespace tagodsp
