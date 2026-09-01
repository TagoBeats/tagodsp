#pragma once

// Forward STFT, framed exactly like the Python workbench's
// `tagodsp.spectral.stft.Stft` so the two implementations can be diffed
// against golden files.
//
// Framing: periodic Hann window (np.hanning(n_fft + 1)[:-1], i.e.
// w[k] = 0.5 - 0.5*cos(2*pi*k/n_fft)), the signal zero-padded by n_fft/2 on
// both sides so frame k is centered on sample k*hop, then rfft per frame.
// Hann with hop = n_fft/4 satisfies COLA (Griffin & Lim 1984).
//
// Only the forward transform lives here. The inverse exists in Python for the
// resynthesis modules; the masking analysis this core was cut for never
// resynthesizes, and an untested inverse in a header is worse than no inverse.

#include <cmath>
#include <complex>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include <tagodsp/fft.hpp>
#include <tagodsp/matrix.hpp>

namespace tagodsp {

class Stft {
public:
    explicit Stft(std::size_t nFft = 1024, std::size_t hop = 256)
        : nFft_(nFft), hop_(hop), rfft_(nFft) {
        if (nFft == 0 || hop == 0 || hop > nFft) {
            throw std::invalid_argument("need 0 < hop <= nFft");
        }
        window_.resize(nFft);
        for (std::size_t k = 0; k < nFft; ++k) {
            window_[k] = 0.5 - 0.5 * std::cos(2.0 * kPi * static_cast<double>(k) /
                                              static_cast<double>(nFft));
        }
    }

    std::size_t nFft() const noexcept { return nFft_; }
    std::size_t hop() const noexcept { return hop_; }
    std::size_t nBins() const noexcept { return nFft_ / 2 + 1; }

    /// Number of frames produced for a signal of `nSamples` samples.
    std::size_t frameCount(std::size_t nSamples) const noexcept {
        const std::size_t padded = nSamples + 2 * (nFft_ / 2);
        if (padded < nFft_) {
            return 0;
        }
        return 1 + (padded - nFft_) / hop_;
    }

    /// Center frequency of each bin in Hz.
    std::vector<double> freqs(double sr) const {
        std::vector<double> f(nBins());
        for (std::size_t k = 0; k < f.size(); ++k) {
            f[k] = static_cast<double>(k) * sr / static_cast<double>(nFft_);
        }
        return f;
    }

    /// Center time of each frame in seconds.
    std::vector<double> times(std::size_t nFrames, double sr) const {
        std::vector<double> t(nFrames);
        for (std::size_t k = 0; k < nFrames; ++k) {
            t[k] = static_cast<double>(k) * static_cast<double>(hop_) / sr;
        }
        return t;
    }

    /// Power spectrum |S|^2 of a mono signal, as a (frames x bins) grid.
    Matrix powerSpectrum(const std::vector<double>& x) const {
        const std::size_t nFrames = frameCount(x.size());
        Matrix out(nFrames, nBins());
        if (nFrames == 0) {
            return out;
        }
        const std::ptrdiff_t pad = static_cast<std::ptrdiff_t>(nFft_ / 2);
        std::vector<double> frame(nFft_);
        std::vector<std::complex<double>> spectrum(nBins());
        for (std::size_t f = 0; f < nFrames; ++f) {
            const std::ptrdiff_t base = static_cast<std::ptrdiff_t>(f * hop_) - pad;
            for (std::size_t k = 0; k < nFft_; ++k) {
                const std::ptrdiff_t idx = base + static_cast<std::ptrdiff_t>(k);
                const double sample =
                    (idx < 0 || idx >= static_cast<std::ptrdiff_t>(x.size()))
                        ? 0.0
                        : x[static_cast<std::size_t>(idx)];
                frame[k] = sample * window_[k];
            }
            rfft_.forward(frame.data(), spectrum.data());
            for (std::size_t k = 0; k < spectrum.size(); ++k) {
                const double re = spectrum[k].real();
                const double im = spectrum[k].imag();
                out.at(f, k) = re * re + im * im;
            }
        }
        return out;
    }

private:
    std::size_t nFft_;
    std::size_t hop_;
    std::vector<double> window_;
    Rfft rfft_;
};

} // namespace tagodsp
