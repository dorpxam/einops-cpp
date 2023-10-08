#pragma once

#include "test_tools.hpp"

class EinsumTest : public UnitTest
{
public:
    EinsumTest()
        : UnitTest("Einsum")
    {}

    void test_layer()
    {
        // TODO
    }

    void test_list() final
    {
        test_layer();
    }
};