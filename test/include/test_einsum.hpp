#pragma once

#include "test_tools.hpp"

class EinsumTest : public UnitTest
{
public:
    EinsumTest()
        : UnitTest("Einsum")
    {}

    void test_functional()
    {
        const std::vector<std::tuple<std::string, std::string, std::vector<std::vector<int64_t>>, std::vector<int64_t>>> test_functional_cases =
        {
            { 
                // Basic:
                "b c h w, b w -> b h", 
                "abcd,ad->ac", 
                { { 2, 3, 4, 5 }, { 2, 5 } }, 
                { 2, 4 } 
            },
            {
                // Three tensors:
                "b c h w, b w, b c -> b h",
                "abcd,ad,ab->ac",
                { { 2, 3, 40, 5 }, { 2, 5 }, { 2, 3 } },
                { 2, 40 },
            },
            {
                // Ellipsis, and full names:
                "... one two three, three four five -> ... two five",
                "...abc,cde->...be",
                { { 32, 5, 2, 3, 4 }, { 4, 5, 6 } },
                { 32, 5, 3, 6 },
            },
            {
                // Ellipsis at the end :
                "one two three ..., three four five -> two five ...",
                "abc...,cde->be...",
                { { 2, 3, 4, 32, 5 }, { 4, 5, 6 } },
                { 3, 6, 32, 5 },
            },
            {
                // Ellipsis on multiple tensors :
                "... one two three, ... three four five -> ... two five",
                "...abc,...cde->...be",
                { { 32, 5, 2, 3, 4 }, { 32, 5, 4, 5, 6 } },
                { 32, 5, 3, 6 },
            },
            {
                // One tensor, and underscores:
                "first_tensor second_tensor -> first_tensor",
                "ab->a",
                { { 5, 4 }, },
                { 5, },
            },
            {
                // Trace{repeated index}
                "i i -> ",
                "aa->",
                { { 5, 5 }, },
                { },
            },
            {
                // Too many spaces in string :
                " one  two  ,  three four->two  four  ",
                "ab,cd->bd",
                { { 2, 3 }, { 4, 5 } },
                { 3, 5 },
            },
            {
                // Trace with other indices
                "i middle i -> middle",
                "aba->b",
                { { 5, 10, 5 }, },
                { 10, },
            },
            {
                // Ellipsis in the middle :
                "i ... i -> ...",
                "a...a->...",
                { { 5, 3, 2, 1, 4, 5 }, },
                { 3, 2, 1, 4 },
            },
            {
                // Product of first and last axes :
                "i ... i -> i ...",
                "a...a->a...",
                { { 5, 3, 2, 1, 4, 5 }, },
                { 5, 3, 2, 1, 4 },
            },
            {
                // Triple diagonal
                "one one one -> one",
                "aaa->a",
                { { 5, 5, 5 }, },
                { 5, },
            },
            {
                // Axis swap :
                "i j k -> j i k",
                "abc->bac",
                { { 1, 2, 3 }, },
                { 2, 1, 3 },
            },
            {
                // Identity:
                "... -> ...",
                "...->...",
                { { 5, 4, 3, 2, 1 }, },
                { 5, 4, 3, 2, 1 },
            },
            {
                // Elementwise product of three tensors
                "..., ..., ... -> ...",
                "...,...,...->...",
                { { 3, 2 }, { 3, 2 }, { 3, 2 } },
                { 3, 2 },
            },
            {
                // Basic summation :
                "index ->",
                "a->",
                { { 10, } },
                { { } },
            },
        };
        
        for (auto&& [einops_pattern, true_pattern, in_shapes, out_shape] : test_functional_cases)
        {
            auto predicted_pattern = _compactify_pattern_for_einsum(einops_pattern);
            TESTS(predicted_pattern, true_pattern);
        }
    }

    void test_list() final
    {
        test_functional();
    }
};