#pragma once

#include "test_tools.hpp"

const std::vector<std::string> identity_patterns =
{
    "...->...",
    "a b c d e-> a b c d e",
    "a b c d e ...-> ... a b c d e",
    "a b c d e ...-> a ... b c d e",
    "... a b c d e -> ... a b c d e",
    "a ... e-> a ... e",
    "a ... -> a ... ",
    "a ... c d e -> a (...) c d e",
};

const std::vector<std::pair<std::string, std::string>> equivalent_rearrange_patterns =
{
    { "a b c d e -> (a b) c d e",   "a b ... -> (a b) ... " },
    { "a b c d e -> a b (c d) e",   "... c d e -> ... (c d) e" },
    { "a b c d e -> a b c d e",     "... -> ... " },
    { "a b c d e -> (a b c d e)",   "... ->  (...)" },
    { "a b c d e -> b (c d e) a",   "a b ... -> b (...) a" },
    { "a b c d e -> b (a c d) e",   "a b ... e -> b (a ...) e" },
};

const std::vector<std::pair<std::string, std::string>> equivalent_reduction_patterns =
{
    { "a b c d e -> ",          " ... ->  " },
    { "a b c d e -> (e a)",     "a ... e -> (e a)" },
    { "a b c d e -> d (a e)",   " a b c d e ... -> d (a e) " },
    { "a b c d e -> (a b)",     " ... c d e  -> (...) " },
};

class OpsTest : public UnitTest
{
public:
    OpsTest()
        : UnitTest("Ops")
    {}

    void test_ellipsis_ops()
    {
        auto x = torch::arange(2 * 3 * 4 * 5 * 6).reshape({ 2, 3, 4, 5, 6 });
        
        for (auto&& pattern : identity_patterns)
            TESTB(array_equal(x, rearrange(x, pattern)));

        for (auto&& [pattern1, pattern2] : equivalent_rearrange_patterns)
            TESTB(array_equal(rearrange(x, pattern1), rearrange(x, pattern2)));

        for (auto&& reduction : { "min", "max", "sum" })
            for (auto&& [pattern1, pattern2] : equivalent_reduction_patterns)
                TESTB(array_equal(reduce(x, pattern1, reduction), reduce(x, pattern2, reduction)));
    }

    void test_list() final
    {
        test_ellipsis_ops();
    }
};