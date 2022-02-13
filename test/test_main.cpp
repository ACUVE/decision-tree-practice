#include <algorithm>
#include <cstdint>
#include <dlfcn.h>
#include <fstream>
#include <iostream>
#include <iterator>
#include <optional>
#include <random>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include "benchmark.hpp"
#include "decision.hpp"

using namespace std::string_literals;
using namespace std::string_view_literals;

template <typename Gen>
decision_tree_data random_decision_tree(Gen &&gen, std::size_t const leaf_size)
{
    static std::normal_distribution<double> dist(0.0, 1.0);
    static std::uniform_int_distribution<std::uint8_t> dist_bool(0, 1);
    static std::uniform_int_distribution<std::uint8_t> dist_02(0, 2);

    std::vector<std::uint16_t> feature_index(leaf_size - 1);
    std::generate(feature_index.begin(), feature_index.end(), [i = 0]() mutable
                  { return i++; });
    std::shuffle(feature_index.begin(), feature_index.end(), gen);

    std::vector<std::uint8_t> flag(leaf_size - 1, 0);
    for (std::size_t i = 0u; i < leaf_size - 1; ++i)
    {
        flag[i] = set_missing_type(flag[i], dist_02(gen));
        flag[i] = set_decision_type(flag[i], decision_type_flag::decision_type_default_left, dist_bool(gen));
    }

    std::vector<double> threshold(leaf_size - 1);
    std::generate(threshold.begin(), threshold.end(), [&]
                  { return dist(gen); });

    std::vector<int> nodes;
    std::generate_n(std::back_inserter(nodes), leaf_size, [i = 0]() mutable
                    { return ~(i++); });
    std::generate_n(std::back_inserter(nodes), leaf_size - 2, [i = 0]() mutable
                    { return i++ + 1; });

    std::shuffle(nodes.begin(), nodes.end(), gen);
    std::vector<int> left_node(&nodes[0], &nodes[leaf_size - 1]);
    std::vector<int> right_node(&nodes[leaf_size - 1], &nodes[2 * leaf_size - 2]);

    std::vector<double> leaf_value(leaf_size);
    std::generate(leaf_value.begin(), leaf_value.end(), [&]
                  { return dist(gen); });

    return {
        std::move(feature_index),
        std::move(flag),
        std::move(threshold),
        std::move(left_node),
        std::move(right_node),
        std::move(leaf_value)};
}

void try_one()
{
    std::mt19937_64 gen;
    std::optional<decision_tree_data> maxlength_decision_tree = std::nullopt;
    std::optional<std::string> maxlength_str = std::nullopt;
    std::optional<std::size_t> maxlength_index = std::nullopt;
    {
        std::size_t const leaf_size = 128u;
        std::ofstream ofs("random_decision.cpp");
        ofs << "#include <cmath>"sv << std::endl;
        for (std::size_t i = 0u; i < 100u; ++i)
        {
            auto decision_tree = random_decision_tree(gen, leaf_size);
            auto os_opt = decision_tree_source(decision_tree, "func" + std::to_string(i + 1), "float"sv, "features"sv);
            if (os_opt)
            {
                auto str = std::move(os_opt).value().str();
                ofs << str << std::endl;
                if (!maxlength_str || maxlength_str->length() < str.length())
                {
                    maxlength_decision_tree = std::move(decision_tree);
                    maxlength_str = std::move(str);
                    maxlength_index = i;
                }
            }
        }
    }
    {
        std::cout << "maxlength_index = "sv << maxlength_index.value() << ", func"sv << maxlength_index.value() + 1 << std::endl;
        auto dlhandel = dlopen("./random_decision.so", RTLD_NOW);
        auto func = (double (*)(float const *))dlsym(dlhandel, ("func" + std::to_string(*maxlength_index + 1)).c_str());
        std::cout << (void *)dlhandel << ", " << (void *)func << std::endl;
        std::normal_distribution<double> dist(0.0, 1.0);
        auto max_index = *std::max_element(maxlength_decision_tree->feature_index.begin(), maxlength_decision_tree->feature_index.end());
        std::vector<std::vector<float>> features_list(2000u, std::vector<float>(max_index, float{}));
        auto dt = std::move(maxlength_decision_tree).value();
        auto dt2 = decision_tree_convert(dt).value();
        auto dt3 = decision_tree_v2_convert(dt2).value();
        for (auto &features : features_list)
        {
            std::generate(features.begin(), features.end(), [&]
                          { return dist(gen); });
        }

        std::cout << benchmark([&]
                               {for(auto const &features : features_list){
                                        decision_tree_run(features.data(), dt);
                                    } },
                               10000)
                  << std::endl;
        std::cout << benchmark([&]
                               {for(auto const &features : features_list){
                                        decision_tree_run_nocheck(features.data(), dt);
                                    } },
                               10000)
                  << std::endl;
        std::cout << benchmark([&]
                               {for(auto const &features : features_list){
                                        decision_tree_v2_run(features.data(), dt2);
                                    } },
                               10000)
                  << std::endl;
        std::cout << benchmark([&]
                               {for(auto const &features : features_list){
                                        decision_tree_v2_run_nocheck(features.data(), dt2);
                                    } },
                               10000)
                  << std::endl;
        std::cout << benchmark([&]
                               {for(auto const &features : features_list){
                                        decision_tree_v2_run_v2(features.data(), dt2);
                                    } },
                               10000)
                  << std::endl;
        std::cout << benchmark([&]
                               {for(auto const &features : features_list){
                                        decision_tree_v2_run_nocheck_v2(features.data(), dt2);
                                    } },
                               10000)
                  << std::endl;
        if (func != nullptr)
        {
            std::cout << benchmark([&]
                                   {for(auto const &features : features_list){
                                        func(features.data());
                                    } },
                                   10000)
                      << std::endl;
        }

        for (auto const &features : features_list)
        {
            double rets[] = {
                decision_tree_run(features.data(), dt),
                decision_tree_run_nocheck(features.data(), dt),
                decision_tree_v2_run(features.data(), dt2),
                decision_tree_v2_run_nocheck(features.data(), dt2),
                decision_tree_v2_run_v2(features.data(), dt2),
                decision_tree_v2_run_nocheck_v2(features.data(), dt2),
                func(features.data()),
                decision_tree_v3_run(features.data(), dt3),
            };
            if (!std::all_of(std::begin(rets), std::end(rets), [&](double a)
                             { return a == rets[0]; }))
            {
                bool first = true;
                for (auto const &r : rets)
                {
                    if (!std::exchange(first, false))
                    {
                        std::cout << ", ";
                    }
                    std::cout << r;
                }
                std::cout << std::endl;
            }
        }
    }
}

int main()
{
    try_one();
}
