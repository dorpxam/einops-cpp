#pragma once

#include "test_tools.hpp"
using namespace einops;

class APITest : public UnitTest
{
public:
    APITest()
        : UnitTest("API")
    {}

    void test_reduce()
    {
        {
            auto x = torch::randn({ 100, 32, 64 });
            {
                // perform max-reduction on the first axis
                auto y = reduce(x, "t b c -> b c", "max");
                CHECK(dump(y.sizes()), dump({ 32, 64 }));
            }
            {
                // same as previous, but with clearer axes meaning
                auto y = reduce(x, "time batch channel -> batch channel", "max");
                CHECK(dump(y.sizes()), dump({ 32, 64 }));
            }
        }
        {
            auto x = torch::randn({ 10, 20, 30, 40 });
            {
                // 2d max-pooling with kernel size = 2 * 2 for image processing
                auto y1 = reduce(x, "b c (h1 h2) (w1 w2) -> b c h1 w1", "max", axis("h2", 2), axis("w2", 2));
                CHECK(dump(y1.sizes()), dump({ 10, 20, 15, 20 }));
            
                // if one wants to go back to the original height and width, depth-to-space trick can be applied
                auto y2 = rearrange(x, "b (c h2 w2) h1 w1 -> b c (h1 h2) (w1 w2)", axis("h2", 2), axis("w2", 2));
                // assert parse_shape(x, 'b _ h w') == parse_shape(y2, 'b _ h w')
            }
            {
                // Adaptive 2d max-pooling to 3 * 4 grid
                auto y = reduce(x, "b c (h1 h2) (w1 w2) -> b c h1 w1", "max", axis("h1", 3), axis("w1", 4));
                CHECK(dump(y.sizes()), dump({ 10, 20, 3, 4 }));
            }
            {
                // Global average pooling
                auto y = reduce(x, "b c h w -> b c", "mean");
                CHECK(dump(y.sizes()), dump({ 10, 20 }));
            }
            {
                // Subtracting mean over batch for each channel
                auto y = x - reduce(x, "b c h w -> () c () ()", "mean");
                CHECK(dump(y.sizes()), dump({ 10, 20, 30, 40 }));
            }
            {
                // Subtracting per-image mean for each channel
                auto y = x - reduce(x, "b c h w -> b c () ()", "mean");
                CHECK(dump(y.sizes()), dump({ 10, 20, 30, 40 }));
            }
        }
    }

    void test_rearrange()
    {
        auto build_random_images = [](std::size_t n, at::IntArrayRef const& rnd)
        {
            std::vector<torch::Tensor> images; images.reserve(n);
            std::generate_n(std::back_inserter(images), n, [=]() mutable { return torch::randn(rnd); });
            return images;
        };
        auto images = build_random_images(32, { 30, 40, 3 });
        {
            // stack along first (batch) axis, output is a single array
            auto y = rearrange(images, "b h w c -> b h w c");
            CHECK(dump(y.sizes()), dump({ 32, 30, 40, 3 }));
        }
        {
            // concatenate images along height (vertical axis), 960 = 32 * 30
            auto y = rearrange(images, "b h w c -> (b h) w c");
            CHECK(dump(y.sizes()), dump({ 960, 40, 3 }));
        }
        {
            // concatenated images along horizontal axis, 1280 = 32 * 40
            auto y = rearrange(images, "b h w c -> h (b w) c");
            CHECK(dump(y.sizes()), dump({ 30, 1280, 3 }));
        }
        {
            // reordered axes to "b c h w" format for deep learning
            auto y = rearrange(images, "b h w c -> b c h w");
            CHECK(dump(y.sizes()), dump({ 32, 3, 30, 40 }));
        }
        {
            // flattened each image into a vector, 3600 = 30 * 40 * 3
            auto y = rearrange(images, "b h w c -> b (c h w)");
            CHECK(dump(y.sizes()), dump({ 32, 3600 }));
        }
        {
            // split each image into 4 smaller (top-left, top-right, bottom-left, bottom-right), 128 = 32 * 2 * 2
            auto y = rearrange(images, "b (h1 h) (w1 w) c -> (b h1 w1) h w c", axis("h1", 2), axis("w1", 2));
            CHECK(dump(y.sizes()), dump({ 128, 15, 20, 3 }));
        }
        {
            // space-to-depth operation
            auto y = rearrange(images, "b (h h1) (w w1) c -> b h w (c h1 w1)", axis("h1", 2), axis("w1", 2));
            CHECK(dump(y.sizes()), dump({ 32, 15, 20, 12 }));
        }
    }

    void test_repeat()
    {
        // a grayscale image (of shape height x width)
        auto image = torch::randn({ 30, 40 });
        {
            // change it to RGB format by repeating in each channel
            auto result = repeat(image, "h w -> h w c", axis("c", 3));
            CHECK(dump(result.sizes()), dump({ 30, 40, 3 }));
        }
        {
            // repeat image 2 times along height (vertical axis)
            auto result = repeat(image, "h w -> (repeat h) w", axis("repeat", 2));
            CHECK(dump(result.sizes()), dump({ 60, 40 }));
        }
        {
            // repeat image 2 time along height and 3 times along width
            auto result = repeat(image, "h w -> (h2 h) (w3 w)", axis("h2", 2), axis("w3", 3));
            CHECK(dump(result.sizes()), dump({ 60, 120 }));
        }
        {
            // convert each pixel to a small square 2x2. Upsample image by 2x
            auto result = repeat(image, "h w -> (h h2) (w w2)", axis("h2", 2), axis("w2", 2));
            CHECK(dump(result.sizes()), dump({ 60, 80 }));
        }
        {
            // pixelate image first by downsampling by 2x, then upsampling
            auto downsampled = reduce(image, "(h h2) (w w2) -> h w", "mean", axis("h2", 2), axis("w2", 2));
            auto result = repeat(downsampled, "h w -> (h h2) (w w2)", axis("h2", 2), axis("w2", 2));
            CHECKT(dump(result.sizes()) == dump({ 30, 40 }));
        }
    }

    void test_einsum()
    {
        {
            auto x = torch::randn({ 20, 20, 20 });
            auto y = torch::randn({ 20, 20, 20 });
            auto z = torch::randn({ 20, 20, 20 });
            auto result = einsum("a b c, c b d, a g k -> a b k", x, y, z);
            CHECK(dump(result.sizes()), dump({ 20, 20, 20 }));
        }
        {
            auto batched_images = torch::randn({ 128, 16, 16 });
            auto filters = torch::randn({ 16, 16, 30 });
            auto result = einsum("batch h w, h w channel -> batch channel", batched_images, filters);
            CHECK(dump(result.sizes()), dump({ 128, 30 }));
        }
        {
            auto data = torch::randn({ 50, 30, 20 });
            auto weights = torch::randn({ 10, 20 });
            auto result = einsum("out_dim in_dim, ... in_dim -> ... out_dim", weights, data);
            CHECK(dump(result.sizes()), dump({ 50, 30, 10 }));
        }
        {
            auto matrix = torch::randn({ 10, 10 });
            auto result = einsum("i i ->", matrix);
            CHECK(dump(result.sizes()), dump({}));
        }
    }

    void test_list() final
    {
        test_reduce();
        test_rearrange();
        test_repeat();
        test_einsum();
    }
};