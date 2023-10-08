#include "test_api.hpp"
#include "test_einsum.hpp"
#include "test_examples.hpp"
#include "test_ops.hpp"
#include "test_parsing.hpp"

int main()
{
    auto check = [](int a, int b)
    {
        return a != 0 ? a : b;
    };

    int out = 0;
        out = check(out,  ParsingTest().run());
        out = check(out,      OpsTest().run());
        out = check(out,   EinsumTest().run());
        out = check(out, ExamplesTest().run());
        out = check(out,      APITest().run());

    return out;
}