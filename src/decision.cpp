#include <cmath>
#include <iomanip>
#include <limits>
#include <map>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <xbyak/xbyak.h>

#include "decision.hpp"

using namespace std::string_literals;
using namespace std::string_view_literals;

static void decision_leaf_source(std::ostringstream &oss, double value)
{
    oss << "return "sv << value << ";"sv;
}

static void decision_node_source(
    std::ostringstream &oss,
    int node,
    decision_tree_data const &decision_tree,
    std::string_view features_name)
{
    if (node < 0)
    {
        decision_leaf_source(oss, decision_tree.leaf_value.at(~node));
    }
    else
    {
        auto const missing_type = get_missing_type(decision_tree.flag.at(node));
        auto const default_left = get_decision_type(decision_tree.flag.at(node), decision_type_flag::decision_type_default_left);
        oss << "auto fval = static_cast<double>("sv << features_name << "["sv << decision_tree.feature_index.at(node) << "]);"sv;
        if (missing_type != missing_type_flag::NaN)
        {
            oss << "if(std::isnan(fval)){fval=0.0;}"sv;
        }
        auto const normal_decision = [&]
        {
            oss << "(fval<="sv << decision_tree.threshold.at(node) << ")"sv;
        };
        auto const zero_decision = [&]
        {
            oss << "("sv << -ZERO_THRESHOLD << "<=fval&&fval<="sv << ZERO_THRESHOLD << ")"sv;
        };
        auto const nan_decision = [&]
        {
            oss << "std::isnan(fval)"sv;
        };
        oss << "if("sv;
        normal_decision();
        switch (missing_type)
        {
        case missing_type_flag::None:
        default:
            break;
        case missing_type_flag::Zero:
            if (default_left)
            {
                oss << "||"sv;
                zero_decision();
            }
            else
            {
                oss << "&&!"sv;
                zero_decision();
            }
            break;
        case missing_type_flag::NaN:
            if (default_left)
            {
                oss << "||"sv;
                nan_decision();
            }
            else
            {
                oss << "&&!"sv;
                nan_decision();
            }
            break;
        }
        oss << "){"sv;
        decision_node_source(oss, decision_tree.left_node.at(node), decision_tree, features_name);
        oss << "}else{"sv;
        decision_node_source(oss, decision_tree.right_node.at(node), decision_tree, features_name);
        oss << "}"sv;
    }
}

__attribute__((visibility("default"))) bool
decision_tree_validate(
    decision_tree_data const &decision_tree) noexcept
{
    auto const leaf_num = decision_tree.leaf_value.size();
    if (leaf_num == 0u)
    {
        return false;
    }
    auto const decision_num = leaf_num - 1u;
    if (decision_num != decision_tree.feature_index.size() ||
        decision_num != decision_tree.threshold.size() ||
        decision_num != decision_tree.left_node.size() ||
        decision_num != decision_tree.right_node.size())
    {
        return false;
    }
    return true;
}

__attribute__((visibility("default")))
std::optional<std::ostringstream>
decision_tree_source(
    decision_tree_data const &decision_tree,
    std::string_view function_name,
    std::string_view feature_type,
    std::string_view features_name)
try
{
    if (function_name.empty() ||
        feature_type.empty() ||
        features_name.empty() ||
        decision_tree.leaf_value.empty())
    {
        return std::nullopt;
    }
    auto const leaf_num = decision_tree.leaf_value.size();
    if (!decision_tree_validate(decision_tree))
    {
        return std::nullopt;
    }
    std::ostringstream oss;
    oss << std::setprecision(std::numeric_limits<double>::digits10 + 2) << std::scientific;
    oss << "extern \"C\" double "sv << function_name << "("sv << feature_type << " const *"sv << features_name << ")noexcept{"sv;
    if (leaf_num == 1u)
    {
        decision_leaf_source(oss, decision_tree.leaf_value.at(0));
    }
    else
    {
        decision_node_source(oss, 0, decision_tree, features_name);
    }
    oss << "}"sv;
    return std::make_optional(std::move(oss));
}
catch (...)
{
    return std::nullopt;
}

__attribute__((visibility("default"))) double
decision_tree_run(
    float const *features,
    decision_tree_data const &decision_tree)
{
    if (decision_tree.leaf_value.empty())
    {
        return 0.0;
    }
    else if (decision_tree.leaf_value.size() == 1u)
    {
        return decision_tree.leaf_value.at(0);
    }
    int node = 0;
    while (node >= 0)
    {
        auto const missing_type = get_missing_type(decision_tree.flag.at(node));
        double fval = features[decision_tree.feature_index.at(node)];
        if (missing_type != missing_type_flag::NaN &&
            std::isnan(fval))
        {
            fval = 0.0;
        }
        bool choose_left;
        if ((missing_type == missing_type_flag::Zero && is_zero(fval)) ||
            (missing_type == missing_type_flag::NaN && std::isnan(fval)))
        {
            choose_left = get_decision_type(decision_tree.flag.at(node), decision_type_flag::decision_type_default_left);
        }
        else
        {
            choose_left = fval <= decision_tree.threshold.at(node);
        }
        node = choose_left ? decision_tree.left_node.at(node) : decision_tree.right_node.at(node);
    }
    return decision_tree.leaf_value.at(~node);
}

__attribute__((visibility("default"))) double
decision_tree_run_nocheck(
    float const *features,
    decision_tree_data const &decision_tree)
{
    if (decision_tree.leaf_value.empty())
    {
        return 0.0;
    }
    else if (decision_tree.leaf_value.size() == 1u)
    {
        return decision_tree.leaf_value[0];
    }
    int node = 0;
    while (node >= 0)
    {
        auto const missing_type = get_missing_type(decision_tree.flag[node]);
        double fval = features[decision_tree.feature_index[node]];
        if (missing_type != missing_type_flag::NaN &&
            std::isnan(fval))
        {
            fval = 0.0;
        }
        bool choose_left;
        if ((missing_type == missing_type_flag::Zero && is_zero(fval)) ||
            (missing_type == missing_type_flag::NaN && std::isnan(fval)))
        {
            choose_left = get_decision_type(decision_tree.flag[node], decision_type_flag::decision_type_default_left);
        }
        else
        {
            choose_left = fval <= decision_tree.threshold[node];
        }
        node = choose_left ? decision_tree.left_node[node] : decision_tree.right_node[node];
    }
    return decision_tree.leaf_value[~node];
}

__attribute__((visibility("default")))
std::optional<decision_tree_data_v2>
decision_tree_convert(
    decision_tree_data const &from)
try
{
    if (!decision_tree_validate(from))
    {
        return std::nullopt;
    }
    auto const leaf_num = from.leaf_value.size();
    auto const decision_num = leaf_num - 1u;
    decision_tree_data_v2 to;
    to.leaf_value = from.leaf_value;
    to.nodes.resize(decision_num);
    for (std::size_t i = 0; i < decision_num; ++i)
    {
        auto &node = to.nodes.at(i);
        node.feature_index = from.feature_index.at(i);
        node.threshold = from.threshold.at(i);
        node.left = from.left_node.at(i);
        node.right = from.right_node.at(i);
        node.flag = from.flag.at(i);
    }

    return std::make_optional(std::move(to));
}
catch (...)
{
    return std::nullopt;
}

__attribute__((visibility("default"))) bool
decision_tree_v2_validate(
    decision_tree_data_v2 const &decision_tree) noexcept
{
    auto const leaf_num = decision_tree.leaf_value.size();
    if (leaf_num == 0u)
    {
        return false;
    }
    auto const decision_num = decision_tree.nodes.size();
    return leaf_num - 1 == decision_num;
}

__attribute__((visibility("default"))) double
decision_tree_v2_run(
    float const *features,
    decision_tree_data_v2 const &decision_tree)
{
    if (decision_tree.leaf_value.empty())
    {
        return 0.0;
    }
    else if (decision_tree.leaf_value.size() == 1u)
    {
        return decision_tree.leaf_value.at(0);
    }
    int node = 0;
    while (node >= 0)
    {
        auto const &n = decision_tree.nodes.at(node);
        auto const missing_type = get_missing_type(n.flag);
        double fval = features[n.feature_index];
        if (missing_type != missing_type_flag::NaN &&
            std::isnan(fval))
        {
            fval = 0.0;
        }
        bool choose_left;
        if ((missing_type == missing_type_flag::Zero && is_zero(fval)) ||
            (missing_type == missing_type_flag::NaN && std::isnan(fval)))
        {
            choose_left = get_decision_type(n.flag, decision_type_flag::decision_type_default_left);
        }
        else
        {
            choose_left = fval <= n.threshold;
        }
        node = choose_left ? n.left : n.right;
    }
    return decision_tree.leaf_value.at(~node);
}

__attribute__((visibility("default"))) double
decision_tree_v2_run_nocheck(
    float const *features,
    decision_tree_data_v2 const &decision_tree)
{
    if (decision_tree.leaf_value.empty())
    {
        return 0.0;
    }
    else if (decision_tree.leaf_value.size() == 1u)
    {
        return decision_tree.leaf_value[0];
    }
    int node = 0;
    while (node >= 0)
    {
        auto const &n = decision_tree.nodes[node];
        auto const missing_type = get_missing_type(n.flag);
        double fval = features[n.feature_index];
        if (missing_type != missing_type_flag::NaN &&
            std::isnan(fval))
        {
            fval = 0.0;
        }
        bool choose_left;
        if ((missing_type == missing_type_flag::Zero && is_zero(fval)) ||
            (missing_type == missing_type_flag::NaN && std::isnan(fval)))
        {
            choose_left = get_decision_type(n.flag, decision_type_flag::decision_type_default_left);
        }
        else
        {
            choose_left = fval <= n.threshold;
        }
        node = choose_left ? n.left : n.right;
    }
    return decision_tree.leaf_value[~node];
}

__attribute__((visibility("default"))) double
decision_tree_v2_run_v2(
    float const *features,
    decision_tree_data_v2 const &decision_tree)
{
    if (decision_tree.leaf_value.empty())
    {
        return 0.0;
    }
    else if (decision_tree.leaf_value.size() == 1u)
    {
        return decision_tree.leaf_value.at(0);
    }
    int node = 0;
    while (node >= 0)
    {
        auto const &n = decision_tree.nodes.at(node);
        auto const missing_type = get_missing_type(n.flag);
        double fval = features[n.feature_index];
        bool choose_left;
        switch (missing_type)
        {
        case missing_type_flag::None:
        default:
            if (std::isnan(fval))
            {
                fval = 0.0;
            }
            choose_left = fval <= n.threshold;
            break;
        case missing_type_flag::Zero:
            if (std::isnan(fval))
            {
                fval = 0.0;
            }
            choose_left = is_zero(fval) ? get_decision_type(n.flag, decision_type_flag::decision_type_default_left) : fval <= n.threshold;
            break;
        case missing_type_flag::NaN:
            choose_left = std::isnan(fval) ? get_decision_type(n.flag, decision_type_flag::decision_type_default_left) : fval <= n.threshold;
            break;
        }
        node = choose_left ? n.left : n.right;
    }
    return decision_tree.leaf_value.at(~node);
}

__attribute__((visibility("default"))) double
decision_tree_v2_run_nocheck_v2(
    float const *features,
    decision_tree_data_v2 const &decision_tree)
{
    if (decision_tree.leaf_value.empty())
    {
        return 0.0;
    }
    else if (decision_tree.leaf_value.size() == 1u)
    {
        return decision_tree.leaf_value[0];
    }
    int node = 0;
    while (node >= 0)
    {
        auto const &n = decision_tree.nodes[node];
        auto const missing_type = get_missing_type(n.flag);
        double fval = features[n.feature_index];
        bool choose_left;
        switch (missing_type)
        {
        case missing_type_flag::None:
        default:
            if (std::isnan(fval))
            {
                fval = 0.0;
            }
            choose_left = fval <= n.threshold;
            break;
        case missing_type_flag::Zero:
            if (std::isnan(fval))
            {
                fval = 0.0;
            }
            choose_left = is_zero(fval) ? get_decision_type(n.flag, decision_type_flag::decision_type_default_left) : fval <= n.threshold;
            break;
        case missing_type_flag::NaN:
            choose_left = std::isnan(fval) ? get_decision_type(n.flag, decision_type_flag::decision_type_default_left) : fval <= n.threshold;
            break;
        }
        node = choose_left ? n.left : n.right;
    }
    return decision_tree.leaf_value[~node];
}

struct decision_tree_data_v3::internal_data : Xbyak::CodeGenerator
{
private:
    struct build_data
    {
        std::list<std::tuple<double, Xbyak::Label>> num_to_label;

        Xbyak::Label &double_num(double num)
        {
            auto it = std::find_if(this->num_to_label.begin(), this->num_to_label.end(), [&](auto &v)
                                   { return std::get<double>(v) == num; });
            if (it != this->num_to_label.end())
            {
                return std::get<Xbyak::Label>(*it);
            }
            this->num_to_label.emplace_back(num, Xbyak::Label{});
            return std::get<Xbyak::Label>(this->num_to_label.back());
        }
    };

public:
    internal_data(decision_tree_data_v2 const &from)
        : Xbyak::CodeGenerator(4096, Xbyak::AutoGrow)
    {
        if (!decision_tree_v2_validate(from))
        {
            pxor(xmm0, xmm0);
            ret();
            return;
        }
        build_data data;
        if (from.leaf_value.size() == 1u)
        {
            movsd(xmm0, qword[rip + data.double_num(from.leaf_value.at(0))]);
            ret();
        }
        else
        {
            if (std::any_of(from.nodes.begin(), from.nodes.end(), [](auto const &node)
                            { return get_missing_type(node.flag) == missing_type_flag::Zero; }))
            {
                movsd(xmm2, qword[rip + data.double_num(-ZERO_THRESHOLD)]);
                movsd(xmm3, qword[rip + data.double_num(ZERO_THRESHOLD)]);
            }
            this->node(from, 0, data);
        }
        align(8);
        for (auto &[key, value] : data.num_to_label)
        {
            union
            {
                double d;
                std::uint64_t i;
            } tmp;
            tmp.d = key;
            L(value);
            dq(tmp.i);
        }
    }

    void node(decision_tree_data_v2 const &from, int node, build_data &data)
    {
        if (node < 0)
        {
            movsd(xmm0, qword[rip + data.double_num(from.leaf_value.at(~node))]);
            ret();
            return;
        }
        else
        {
            auto const &n = from.nodes.at(node);
            switch (get_missing_type(n.flag))
            {
            case missing_type_flag::None:
            default:
            {
                Xbyak::Label label1;
                movss(xmm1, dword[rdi + sizeof(float) * n.feature_index]);
                // nan かチェック
                ucomiss(xmm1, xmm1);
                xorpd(xmm0, xmm0);
                // nan ならジャンプ
                jp(label1);
                // nan でないなら、double に変換
                xorps(xmm0, xmm0);
                cvtss2sd(xmm0, xmm1);
                L(label1);
                ucomisd(xmm0, qword[rip + data.double_num(n.threshold)]);
                if (n.left < 0 && n.right < 0)
                {
                    lea(rcx, ptr[rip + data.double_num(from.leaf_value.at(~n.left))]);
                    lea(rdx, ptr[rip + data.double_num(from.leaf_value.at(~n.right))]);
                    cmovae(rcx, rdx);
                    movsd(xmm0, qword[rcx]);
                    ret();
                }
                else
                {
                    Xbyak::Label label2;
                    jae(label2, T_NEAR);
                    this->node(from, n.left, data);
                    L(label2);
                    this->node(from, n.right, data);
                }
            }
            break;
            case missing_type_flag::Zero:
            {
            }
            break;
            case missing_type_flag::NaN:
            {
            }
            break;
            }
        }
    }
};

__attribute__((visibility("default")))
std::optional<decision_tree_data_v3>
decision_tree_v2_convert(decision_tree_data_v2 const &from)
try
{
    auto data = std::make_shared<decision_tree_data_v3::internal_data>(from);
    std::cout << "make_shared" << std::endl;
    data->readyRE();
    return decision_tree_data_v3{std::move(data)};
}
catch (std::exception &e)
{
    std::cout << "nullopt: " << e.what() << std::endl;
    return std::nullopt;
}
catch (...)
{
    std::cout << "nullopt" << std::endl;
    return std::nullopt;
}

__attribute__((visibility("default"))) double
decision_tree_v3_run(
    float const *features,
    decision_tree_data_v3 const &decision_tree)
{
    return decision_tree.data->getCode<double (*)(float const *)>()(features);
}
