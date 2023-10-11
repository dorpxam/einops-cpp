#pragma once

#include "test_tools.hpp"

class ExamplesTest : public UnitTest
{
public:
	ExamplesTest()
		: UnitTest("Examples")
	{}

	void test_rearrange_examples()
	{
		auto test1 = [this](auto x)
		{
			auto y = ::rearrange(x, "b c h w -> b h w c");
			TESTS(dump(y), dump({ 10, 30, 40, 20 }));
		};

		auto test2 = [this](auto x)
		{
			auto y = ::rearrange(x, "b c h w -> b (c h w)");
			TESTS(dump(y), dump({ 10, 20 * 30 * 40 }));
		};

		auto test3 = [this](auto x)
		{
			auto y = ::rearrange(x, "b (c h1 w1) h w -> b c (h h1) (w w1)", axis("h1", 2), axis("w1", 2));
			TESTS(dump(y), dump({ 10, 5, 30 * 2, 40 * 2 }));
		};

		auto test4 = [this](auto x)
		{
			auto y = ::rearrange(x, "b c (h h1) (w w1) -> b (h1 w1 c) h w", axis("h1", 2), axis("w1", 2));
			TESTS(dump(y), dump({ 10, 20 * 4, 30 / 2, 40 / 2 }));
		};

		auto test5 = [this](auto x)
		{
			auto y = ::rearrange(x, "b1 sound b2 letter -> b1 b2 sound letter");
			TESTS(dump(y), dump({ 10, 30, 20, 40 }));
		};
		
		auto test6 = [this](auto x)
		{
			auto t = ::rearrange(x, "b c h w -> (b h w) c");
			TESTS(dump(t), dump({ 10 * 30 * 40, 10 }));
			auto y = ::rearrange(t, "(b h w) c2 -> b c2 h w");
			TESTS(dump(y), dump({ 10, 10, 30, 40 }));
		};
		
		auto test7 = [this](auto x)
		{
			auto y = ::rearrange(x, "b (c g) h w -> g b c h w", axis("g", 2));
			TESTS(dump(y), dump({ 10, 10, 30, 40 }));
		};
	
		auto test8 = [this](auto x)
		{
			auto y = ::reduce(x, "b c (h h1) (w w1) -> b c h w", "max", axis("h1", 2), axis("w1", 2));
			TESTS(dump(y), dump({ 10, 20, 30 / 2, 40 / 2 }));
		};
	
		auto test9 = [this](auto x)
		{
			auto y = ::reduce(x, "b c h w -> b c () ()", "max");
			TESTS(dump(y), dump({ 10, 20, 1, 1 }));
			y = ::rearrange(y, "b c () () -> c b");
			TESTS(dump(y), dump({ 20, 10 }));
		};

		auto test10 = [this](auto x)
		{
			auto y = ::rearrange(x, "b c h w -> b h w c");
			TESTS(dump(y), dump({ 10, 30, 40, 20 }));
		};

		auto test11 = [this](auto x)
		{
			auto y = ::rearrange(x, "b c h w -> h (b w) c");
			TESTS(dump(y), dump({ 30, 10 * 40, 20 }));
		};

		auto x = arange_and_reshape({ 10 * 20 * 30 * 40 }, { 10, 20, 30, 40 });

		test1(x);	// transpose
		test2(x);	// view / reshape
		test3(x);	// depth-to-space
		test4(x);	// space-to-depth
		test5(x);	// simple transposition
		//test6(x);	// parsing parameters
		//test7(x);	// split of embedding into groups
		test8(x);	// max-pooling
		test9(x);	// squeeze - unsqueeze
		test10(x);	// stack
		test11(x);	// concatenate
	}

	void test_list() final
	{
		test_rearrange_examples();
	}
};