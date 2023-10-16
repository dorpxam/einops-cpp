#pragma once

#include "test_tools.hpp"

// additionnal test helpers for the 'Packing' unit

#define TESTT(a,b) { TESTB(comp_type(a, b)); \
                     TESTB(comp_shape(a,b)); \
                     TESTB(comp_all(a,b)); }

#define TESTU(a,b) { auto&& [unpacked, packed, ps2] = a; \
                     TESTB(comp_allclose(packed, b)); }

#define CATCH(a) { try { a; TESTB(false);  } catch(...) { TESTB(true); } }

class PackingTest : public UnitTest
{
public:
    PackingTest()
        : UnitTest("Packing")
    {}

    void test_trivial(int H = 13, int W = 17)
    {
        auto r = rand({ H, W });
        auto g = rand({ H, W });
        auto b = rand({ H, W });

        auto embeddings = rand({ H, W, 32 });

        TESTT(stack({ r, g, b }, 2), pack_t({ r, g, b }, "h w *"));
        TESTT(stack({ r, g, b }, 1), pack_t({ r, g, b }, "h * w"));
        TESTT(stack({ r, g, b }, 0), pack_t({ r, g, b }, "* h w"));

        TESTT(concatenate({ r, g, b }, 1), pack_t({ r, g, b }, "h *"));
        TESTT(concatenate({ r, g, b }, 0), pack_t({ r, g, b }, "* w"));

        TESTT(concatenate_i({ r, g, b, embeddings }, 2), pack_t({ r, g, b, embeddings }, "h w *"));

        CATCH(pack(Tensors{ r, g, b, embeddings }, "h w nonexisting_axis *"));
              pack(Tensors{ r, g, b }, "some_name_for_H some_name_for_w1 *");

        CATCH(pack(Tensors{ r, g, b, embeddings }, "h _w *"));
        CATCH(pack(Tensors{ r, g, b, embeddings }, "h_ w *"));
        CATCH(pack(Tensors{ r, g, b, embeddings }, "1h_ w *"));
        CATCH(pack(Tensors{ r, g, b, embeddings }, "1 w *"));
        CATCH(pack(Tensors{ r, g, b, embeddings }, "h h *"));

        // capital and non-capital are different
        pack(Tensors{ r, g, b, embeddings }, "h H *");
    }

    class UnpackTestCase
    {
    public:
        UnpackTestCase(Shape const& shape, Pattern const& pattern)
            : shape(shape), pattern(pattern)
        {}

        auto dim() const
        {
            return index(splits(pattern, " "), _asterisk);
        }

        auto selfcheck() const
        {
            return shape[dim()] == 5;
        }

        Shape shape;
        Pattern pattern;
    };

    const std::vector<UnpackTestCase> cases =
    {
        UnpackTestCase({ 5, }, "*"),
        UnpackTestCase({ 5, 7 }, "* seven"),
        UnpackTestCase({ 7, 5 }, "seven *"),
        UnpackTestCase({ 5, 3, 4 }, "* three four"),
        UnpackTestCase({ 4, 5, 3 }, "four * three"),
        UnpackTestCase({ 3, 4, 5 }, "three four *"),
    };

    void test_pack_unpack()
    {
        auto unpack_and_pack = [](auto x, std::vector<std::vector<int64_t>> const& ps, std::string const& pattern)
        {
            auto&& unpacked = unpack(x, ps, pattern);
            auto&& [packed, ps2] = pack(unpacked, pattern);
            return std::make_tuple(unpacked, packed, ps2);
        };

        for (auto&& _case : cases)
        {
            auto x = rand(_case.shape);

            // all correct, no minus 1
            TESTU(unpack_and_pack(x, { { 2 }, { 1 }, { 2 } }, _case.pattern), x);

            // no -1, asking for wrong shapes
            CATCH(unpack_and_pack(x, { { 2 }, { 1 }, { 2 } }, _case.pattern + " non_existent_axis"));
            CATCH(unpack_and_pack(x, { { 2 }, { 1 }, { 1 } }, _case.pattern));
            CATCH(unpack_and_pack(x, { { 4 }, { 1 }, { 1 } }, _case.pattern));

            // all correct, with -1
            TESTU(unpack_and_pack(x, { {  2 }, {  1 }, { -1 } }, _case.pattern), x);
            TESTU(unpack_and_pack(x, { {  2 }, { -1 }, {  2 } }, _case.pattern), x);
            TESTU(unpack_and_pack(x, { { -1 }, {  1 }, {  2 } }, _case.pattern), x);

            auto&& [unpacked, packed, ps2] = unpack_and_pack(x, { { 2 }, { 3 }, { -1 } }, _case.pattern);
            auto&& last = unpacked.back();
            TESTB(last.size(_case.dim()) == 0);

            // asking for more elements than available
            CATCH(unpack(x, { {  2 }, { 4 }, { -1 } }, _case.pattern));
            CATCH(unpack(x, { { -1 }, { 1 }, {  5 } }, _case.pattern));

            // asking for more elements, -1 nested
            CATCH(unpack(x, { { -1, 2 }, { 1 }, { 5     } }, _case.pattern));
            CATCH(unpack(x, { {  2, 2 }, { 2 }, { 5, -1 } }, _case.pattern));

            // asking for non-divisible number of elements
            CATCH(unpack(x, { { 2,  1 }, { 1     }, { 3, -1 } }, _case.pattern));
            CATCH(unpack(x, { { 2,  1 }, { 3, -1 }, { 1     } }, _case.pattern));
            CATCH(unpack(x, { { 3, -1 }, { 2,  1 }, { 1     } }, _case.pattern));

            // -1 takes zero
            unpack_and_pack(x, { {  0 }, {  5 }, { -1 } }, _case.pattern);
            unpack_and_pack(x, { {  0 }, { -1 }, {  5 } }, _case.pattern);
            unpack_and_pack(x, { { -1 }, {  5 }, {  0 } }, _case.pattern);

            // -1 takes zero, -1
            unpack_and_pack(x, { { 2, -1 }, { 1, 5 } }, _case.pattern);
        }
    }

    void test_list() final
    {
        test_trivial();
        test_pack_unpack();
    }
};