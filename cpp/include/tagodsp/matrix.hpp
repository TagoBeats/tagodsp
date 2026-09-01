#pragma once

// Plain dense 2D buffer, shared by the analysis modules.
//
// It lives in its own header because it belongs to no single module: the STFT
// fills it with (frames x bins), the masking analysis with (windows x bands)
// power and with (windows x bands) dB cells. Anything else needing a flat 2D
// buffer should include this rather than reach into a module header for it.

#include <cstddef>
#include <vector>

namespace tagodsp {

/// Dense row-major 2D array of doubles.
struct Matrix {
    std::size_t rows = 0;
    std::size_t cols = 0;
    std::vector<double> data;

    Matrix() = default;
    Matrix(std::size_t r, std::size_t c) : rows(r), cols(c), data(r * c, 0.0) {}

    double& at(std::size_t r, std::size_t c) { return data[r * cols + c]; }
    double at(std::size_t r, std::size_t c) const { return data[r * cols + c]; }
};

} // namespace tagodsp
