#include "npu/core.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>

namespace npu {
namespace {

[[noreturn]] void fail(const std::string& message) { throw std::invalid_argument(message); }

void require(bool condition, const char* message) {
    if (!condition) throw std::invalid_argument(message);
}

bool in_dimension(std::uint64_t value) { return value >= 1 && value <= max_dimension; }

// Output extent along one axis: floor((in + pads - extent) / stride) + 1, where
// extent = dilation * (kernel - 1) + 1 is the kernel's reach into the input.
std::uint64_t output_extent(std::uint64_t input, std::uint64_t kernel, std::uint64_t stride,
                            std::uint64_t dilation, std::uint64_t pad_low, std::uint64_t pad_high,
                            const char* axis) {
    const auto padded = checked_add(input, checked_add(pad_low, pad_high));
    const auto extent = checked_add(checked_mul(dilation, kernel - 1), 1);
    if (extent > padded)
        fail(std::string("shape.conv kernel ") + axis + " extent " + std::to_string(extent) +
             " exceeds the padded input " + std::to_string(padded));
    return (padded - extent) / stride + 1;
}

std::int64_t to_signed(std::uint64_t value) { return static_cast<std::int64_t>(value); }

// floor(a / b) and ceil(a / b) for b > 0 and any sign of a.
std::int64_t floor_div(std::int64_t a, std::int64_t b) {
    return a >= 0 ? a / b : -((-a + b - 1) / b);
}
std::int64_t ceil_div_signed(std::int64_t a, std::int64_t b) { return -floor_div(-a, b); }

}  // namespace

bool implicit_conv(const ShapeConfig& shape) {
    const auto& c = shape.conv;
    return c.input_h != 0 || c.input_w != 0 || c.input_c != 0;
}

ConvGeometry conv_geometry(const ShapeConfig& shape) {
    const auto& c = shape.conv;
    if (!implicit_conv(shape)) {
        require(c.kernel_h == 1 && c.kernel_w == 1 && c.stride_h == 1 && c.stride_w == 1 &&
                    c.dilation_h == 1 && c.dilation_w == 1 && c.pad_top == 0 && c.pad_left == 0 &&
                    c.pad_bottom == 0 && c.pad_right == 0,
                "shape.conv kernel, stride, dilation and padding need the input extents "
                "--conv-input-h, --conv-input-w and --conv-input-c");
        return {};
    }
    if (!in_dimension(c.input_h) || !in_dimension(c.input_w) || !in_dimension(c.input_c))
        fail("shape.conv input height, width and channels must all be in 1.." +
             std::to_string(max_dimension));
    if (!in_dimension(c.kernel_h) || !in_dimension(c.kernel_w) || !in_dimension(c.stride_h) ||
        !in_dimension(c.stride_w) || !in_dimension(c.dilation_h) || !in_dimension(c.dilation_w))
        fail("shape.conv kernel, stride and dilation must be in 1.." +
             std::to_string(max_dimension));
    if (c.pad_top > max_dimension || c.pad_left > max_dimension || c.pad_bottom > max_dimension ||
        c.pad_right > max_dimension)
        fail("shape.conv padding must be at most " + std::to_string(max_dimension));
    // The NHWC tensor's byte offsets stay exact in 64 bits.
    checked_mul(checked_mul(c.input_h, c.input_w), c.input_c);
    if (shape.groups == 0 || c.input_c % shape.groups != 0)
        fail("shape.conv input channels " + std::to_string(c.input_c) +
             " must be a multiple of groups " + std::to_string(shape.groups));
    ConvGeometry g;
    g.output_h = output_extent(c.input_h, c.kernel_h, c.stride_h, c.dilation_h, c.pad_top,
                               c.pad_bottom, "height");
    g.output_w = output_extent(c.input_w, c.kernel_w, c.stride_w, c.dilation_w, c.pad_left,
                               c.pad_right, "width");
    g.group_channels = c.input_c / shape.groups;
    g.kernel_row_k = checked_mul(c.kernel_w, g.group_channels);
    const auto pixels = checked_mul(g.output_h, g.output_w);
    if (shape.m != pixels)
        fail("with shape.conv, M must equal output height x width = " + std::to_string(g.output_h) +
             " x " + std::to_string(g.output_w) + " = " + std::to_string(pixels) + ", got " +
             std::to_string(shape.m));
    const auto k = checked_mul(c.kernel_h, g.kernel_row_k);
    if (shape.k != k)
        fail("with shape.conv, K must equal kernel_h x kernel_w x input_c / groups = " +
             std::to_string(k) + ", got " + std::to_string(shape.k));
    return g;
}

KernelRows conv_kernel_rows(const Config& config, std::uint64_t block) {
    const auto g = conv_geometry(config.shape);
    require(g.kernel_row_k != 0, "kernel rows need shape.conv");
    const auto kernel_h = config.shape.conv.kernel_h;
    const auto tile_k = config.mapping.tile_k;
    const auto per_block = tile_k == 0 ? kernel_h : tile_k / g.kernel_row_k;
    require(per_block != 0 && (tile_k == 0 || tile_k % g.kernel_row_k == 0),
            "with shape.conv, tile_k must be 0 or a multiple of kernel_w x input_c / groups");
    const auto first = checked_mul(block, per_block);
    require(first < kernel_h, "reduction block is outside K");
    return {first, std::min(per_block, kernel_h - first)};
}

void conv_input_spans(const Config& config, std::uint64_t first_output, std::uint64_t outputs,
                      std::uint64_t block, std::vector<InputSpan>& spans) {
    const auto& c = config.shape.conv;
    const auto g = conv_geometry(config.shape);
    const auto kernel = conv_kernel_rows(config, block);
    spans.clear();
    if (outputs == 0) return;
    require(first_output < config.shape.m && outputs <= config.shape.m - first_output,
            "output pixels are outside M");
    const auto last_output = first_output + outputs - 1;
    const auto oy_first = first_output / g.output_w, oy_last = last_output / g.output_w;
    const auto height = to_signed(c.input_h), width = to_signed(c.input_w);
    const auto stride_h = to_signed(c.stride_h), stride_w = to_signed(c.stride_w);
    const auto dilation_h = to_signed(c.dilation_h), dilation_w = to_signed(c.dilation_w);
    const auto top = to_signed(c.pad_top), left = to_signed(c.pad_left);
    // Every input row the panel can touch, clipped to the image; each entry
    // holds the hull [first, last] of its needed columns, empty while
    // first > last.
    const auto lowest = std::max<std::int64_t>(
        0, to_signed(oy_first) * stride_h + to_signed(kernel.first) * dilation_h - top);
    const auto highest = std::min<std::int64_t>(
        height - 1, to_signed(oy_last) * stride_h +
                        to_signed(kernel.first + kernel.count - 1) * dilation_h - top);
    if (lowest > highest) return;
    struct Hull {
        std::int64_t first, last;
    };
    const auto none =
        Hull{std::numeric_limits<std::int64_t>::max(), std::numeric_limits<std::int64_t>::min()};
    std::vector<Hull> hulls(static_cast<std::size_t>(highest - lowest + 1), none);
    for (auto oy = oy_first; oy <= oy_last; ++oy) {
        const auto a = to_signed(oy == oy_first ? first_output % g.output_w : 0);
        const auto b = to_signed(oy == oy_last ? last_output % g.output_w : g.output_w - 1);
        // Input column ox * stride + kx * dilation - left for ox in [a, b]:
        // per kernel column, the smallest value at or above 0 and the largest
        // at or below width - 1; the hull spans the extremes over all kx.
        auto column = none;
        for (std::uint64_t kx = 0; kx < c.kernel_w; ++kx) {
            const auto offset = to_signed(kx) * dilation_w - left;
            const auto low_ox = std::max(a, ceil_div_signed(-offset, stride_w));
            if (low_ox <= b) {
                const auto value = low_ox * stride_w + offset;
                if (value < width) column.first = std::min(column.first, value);
            }
            const auto high_ox = std::min(b, floor_div(width - 1 - offset, stride_w));
            if (high_ox >= a) {
                const auto value = high_ox * stride_w + offset;
                if (value >= 0) column.last = std::max(column.last, value);
            }
        }
        if (column.first > column.last) continue;
        for (auto ky = kernel.first; ky < kernel.first + kernel.count; ++ky) {
            const auto row = to_signed(oy) * stride_h + to_signed(ky) * dilation_h - top;
            if (row < lowest || row > highest) continue;
            auto& hull = hulls[static_cast<std::size_t>(row - lowest)];
            hull.first = std::min(hull.first, column.first);
            hull.last = std::max(hull.last, column.last);
        }
    }
    for (std::size_t i = 0; i < hulls.size(); ++i) {
        if (hulls[i].first > hulls[i].last) continue;
        spans.push_back({static_cast<std::uint64_t>(lowest) + i,
                         static_cast<std::uint64_t>(hulls[i].first),
                         static_cast<std::uint64_t>(hulls[i].last)});
    }
}

std::vector<std::int8_t> im2col(const Config& config, const std::vector<std::int8_t>& input) {
    const auto& shape = config.shape;
    const auto& c = shape.conv;
    const auto g = conv_geometry(shape);
    require(g.kernel_row_k != 0, "im2col needs shape.conv");
    require(input.size() == checked_mul(checked_mul(c.input_h, c.input_w), c.input_c),
            "input tensor size does not match shape.conv");
    const auto row_k = checked_mul(shape.groups, shape.k);
    std::vector<std::int8_t> operand(static_cast<std::size_t>(checked_mul(shape.m, row_k)));
    const auto padding = static_cast<std::int8_t>(config.quant.input_zero);
    for (std::uint64_t m = 0; m < shape.m; ++m) {
        const auto oy = m / g.output_w, ox = m % g.output_w;
        for (std::uint64_t ky = 0; ky < c.kernel_h; ++ky) {
            const auto iy = to_signed(oy * c.stride_h + ky * c.dilation_h) - to_signed(c.pad_top);
            for (std::uint64_t kx = 0; kx < c.kernel_w; ++kx) {
                const auto ix =
                    to_signed(ox * c.stride_w + kx * c.dilation_w) - to_signed(c.pad_left);
                const bool inside =
                    iy >= 0 && iy < to_signed(c.input_h) && ix >= 0 && ix < to_signed(c.input_w);
                const auto pixel = inside ? (static_cast<std::uint64_t>(iy) * c.input_w +
                                             static_cast<std::uint64_t>(ix)) *
                                                c.input_c :
                                            0;
                for (std::uint64_t group = 0; group < shape.groups; ++group) {
                    const auto column = group * shape.k + (ky * c.kernel_w + kx) * g.group_channels;
                    for (std::uint64_t ci = 0; ci < g.group_channels; ++ci)
                        operand[static_cast<std::size_t>(m * row_k + column + ci)] =
                            inside ? input[static_cast<std::size_t>(
                                         pixel + group * g.group_channels + ci)] :
                                     padding;
                }
            }
        }
    }
    return operand;
}

}  // namespace npu
