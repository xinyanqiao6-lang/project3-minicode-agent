"""Bug 基准集：5 类 × 6 例 = 30 个用例，用于量化 Agent 的自愈能力边界。

## 为什么从 5 个扩到 30 个
v1 只有 5 个 case，样本量太小，任何结论都只是"观察性"的——60% / 40%
这两个数字经不起追问。30 个 case 按 5 类分布，才能得出"哪类能修、哪类修不了"
这样的结构性结论。

## 五类划分（难度递进）
1. `syntax`      语法错误——模型靠 stderr 就能定位，通常能修
2. `runtime`     运行时异常——同上，但需要正确理解异常信息
3. `logic`       逻辑错误——**不报错但结果错**，最考验模型是否真的理解意图
4. `dependency`  依赖/接口错误——调用不存在的函数、参数用错
5. `boundary`    边界条件——空列表、除零、极端输入，考验防御性编程意识

第 3~5 类是 v1 已发现的"小模型只求不报错、不求语义对"的集中区，
样本扩大后能定量回答"哪些能修、哪些不能"。

## 判定方式（关键）
**用事实判定，不信模型自报**：Agent 结束后，直接跑脚本，
比对 stdout 是否包含期望输出。这与 v1 保持一致，是整个评测的信任基础。
"""
from __future__ import annotations

# 每类 6 个用例
CASES: list[dict] = [
    # ---------- 1. 语法错误 ----------
    {
        "id": "syntax_missing_colon", "category": "syntax",
        "name": "函数定义缺冒号",
        "code": "def greet(name)\n    return f'Hello, {name}'\n\nprint(greet('World'))\n",
        "expect": "Hello, World",
        "task": "运行 workspace 里的 case_syntax_missing_colon.py，如果报错请修复，让它输出 'Hello, World'",
    },
    {
        "id": "syntax_unclosed_paren", "category": "syntax",
        "name": "括号未闭合",
        "code": "def area(w, h):\n    return w * h\n\nprint(area(3, 4)\n",
        "expect": "12",
        "task": "运行 workspace 里的 case_syntax_unclosed_paren.py，报错请修复，让它输出 12",
    },
    {
        "id": "syntax_bad_indent", "category": "syntax",
        "name": "缩进不一致",
        "code": "def total(nums):\n    s = 0\n        for n in nums:\n        s += n\n    return s\n\nprint(total([1,2,3]))\n",
        "expect": "6",
        "task": "运行 workspace 里的 case_syntax_bad_indent.py，报错请修复，让它输出 6",
    },
    {
        "id": "syntax_missing_quote", "category": "syntax",
        "name": "字符串引号未闭合",
        "code": "name = 'Python\nprint(f'I love {name}')\n",
        "expect": "I love Python",
        "task": "运行 workspace 里的 case_syntax_missing_quote.py，报错请修复，让它正确输出 'I love Python'",
    },
    {
        "id": "syntax_wrong_operator", "category": "syntax",
        "name": "使用未定义运算符顺序（= 写成 +）",
        "code": "x = 5\nx + = 3\nprint(x)\n",
        "expect": "8",
        "task": "运行 workspace 里的 case_syntax_wrong_operator.py，报错请修复，让它输出 8",
    },
    {
        "id": "syntax_missing_comma", "category": "syntax",
        "name": "列表字面量缺逗号",
        "code": "nums = [1, 2 3, 4]\nprint(sum(nums))\n",
        "expect": "10",
        "task": "运行 workspace 里的 case_syntax_missing_comma.py，报错请修复，让它输出 10",
    },

    # ---------- 2. 运行时异常 ----------
    {
        "id": "runtime_zero_division", "category": "runtime",
        "name": "除零异常",
        "code": "def divide(a, b):\n    return a / b\n\nprint(divide(10, 0))\n",
        "expect": "除数不能为零",
        "task": "运行 workspace 里的 case_runtime_zero_division.py，如果报错请诊断并修复，让它不再报错",
    },
    {
        "id": "runtime_index_error", "category": "runtime",
        "name": "列表越界 IndexError",
        "code": "nums = [1, 2, 3]\nprint(nums[5])\n",
        "expect": "5",
        "task": "运行 workspace 里的 case_runtime_index_error.py，报错请修复，让它输出 5",
    },
    {
        "id": "runtime_key_error", "category": "runtime",
        "name": "字典 KeyError",
        "code": "d = {'a': 1, 'b': 2}\nprint(d['c'])\n",
        "expect": "0",
        "task": "运行 workspace 里的 case_runtime_key_error.py，报错请修复，让它在键缺失时输出 0",
    },
    {
        "id": "runtime_type_error", "category": "runtime",
        "name": "类型错误 TypeError",
        "code": "s = 'abc'\nprint(s + 1)\n",
        "expect": "abc1",
        "task": "运行 workspace 里的 case_runtime_type_error.py，报错请修复，让它输出 abc1",
    },
    {
        "id": "runtime_name_error", "category": "runtime",
        "name": "变量未定义 NameError",
        "code": "def calc():\n    result = total + 10\n    return result\n\nprint(calc())\n",
        "expect": "30",
        "task": "运行 workspace 里的 case_runtime_name_error.py，报错请修复，让程序输出 30",
    },
    {
        "id": "runtime_attr_error", "category": "runtime",
        "name": "属性不存在 AttributeError",
        "code": "s = 'hello'\nprint(s.uppercase())\n",
        "expect": "HELLO",
        "task": "运行 workspace 里的 case_runtime_attr_error.py，报错请修复，让它输出 HELLO",
    },

    # ---------- 3. 逻辑错误（不报错但结果错） ----------
    {
        "id": "logic_wrong_op", "category": "logic",
        "name": "加法写成减法",
        "code": "def add(a, b):\n    return a - b\n\nprint(add(3, 5))\n",
        "expect": "8",
        "task": "workspace 里的 case_logic_wrong_op.py 计算结果不对，请找出并修复，让 add(3,5) 输出 8",
    },
    {
        "id": "logic_off_by_one", "category": "logic",
        "name": "区间边界差一",
        "code": "def count_upto(n):\n    return range(n)\n\nprint(len(list(count_upto(5))))\n",
        "expect": "5",
        "task": "case_logic_off_by_one.py 应输出 5 但输出了 4，请修复这个 off-by-one 问题",
    },
    {
        "id": "logic_swapped_vars", "category": "logic",
        "name": "两个变量写反",
        "code": "def swap(a, b):\n    return a, b\n\nx, y = swap(1, 2)\nprint(x * 10 + y)\n",
        "expect": "21",
        "task": "case_logic_swapped_vars.py 输出不符合预期，请修复让 x=1,y=2 时打印 21",
    },
    {
        "id": "logic_wrong_return", "category": "logic",
        "name": "return 了中间变量",
        "code": "def middle(nums):\n    return nums[0] + nums[-1]\n\nprint(middle([3, 10, 20, 4]))\n",
        "expect": "14",
        "task": "case_logic_wrong_return.py 应输出 14（中间两个元素之和），请修复",
    },
    {
        "id": "logic_inverted_cond", "category": "logic",
        "name": "条件取反",
        "code": "def is_even(n):\n    if n % 2 != 0:\n        return True\n    return False\n\nprint(is_even(4))\n",
        "expect": "True",
        "task": "case_logic_inverted_cond.py 中 is_even(4) 应返回 True，请修复判断逻辑",
    },
    {
        "id": "logic_accumulate", "category": "logic",
        "name": "累加写成覆盖",
        "code": "def total(nums):\n    s = 0\n    for n in nums:\n        s = n\n    return s\n\nprint(total([1, 2, 3]))\n",
        "expect": "6",
        "task": "case_logic_accumulate.py 求和结果错误，应输出 6，请修复",
    },

    # ---------- 4. 依赖/接口错误 ----------
    {
        "id": "dep_missing_import", "category": "dependency",
        "name": "忘记 import",
        "code": "d = {'x': 1}\nprint(d['x'])\n",
        "expect": "1",
        "task": "case_dep_missing_import.py 报 NameError，请修复让输出 1",
    },
    {
        "id": "dep_wrong_func_name", "category": "dependency",
        "name": "调用不存在的函数",
        "code": "s = 'a,b,c'\nparts = s.split_by_comma()\nprint(len(parts))\n",
        "expect": "3",
        "task": "case_dep_wrong_func_name.py 调用了不存在的方法，请修复让输出 3",
    },
    {
        "id": "dep_wrong_arg_count", "category": "dependency",
        "name": "参数个数不对",
        "code": "def add(a, b):\n    return a + b\n\nprint(add(1, 2, 3))\n",
        "expect": "3",
        "task": "case_dep_wrong_arg_count.py 报参数错误，请修复让输出 3",
    },
    {
        "id": "dep_wrong_import_name", "category": "dependency",
        "name": "导入不存在的模块成员",
        "code": "from math import square_root\nprint(square_root(16))\n",
        "expect": "4.0",
        "task": "case_dep_wrong_import_name.py 导入失败，请用标准库正确实现并输出 4.0",
    },
    {
        "id": "dep_string_vs_number", "category": "dependency",
        "name": "字符串与数字混用",
        "code": "def double(n):\n    return n * 2\n\nprint(double('3'))\n",
        "expect": "6",
        "task": "case_dep_string_vs_number.py 期望输出 6，请修复",
    },
    {
        "id": "dep_wrong_method", "category": "dependency",
        "name": "列表方法用错",
        "code": "nums = [3, 1, 2]\nnums.sort_reverse()\nprint(nums)\n",
        "expect": "[1, 2, 3]",
        "task": "case_dep_wrong_method.py 使用了不存在的方法，请修复让 nums 升序",
    },

    # ---------- 5. 边界条件 ----------
    {
        "id": "boundary_empty_list", "category": "boundary",
        "name": "空列表求最大值",
        "code": "def max_of(nums):\n    return max(nums)\n\nprint(max_of([]))\n",
        "expect": "0",
        "task": "case_boundary_empty_list.py 传入空列表时应输出 0，请修复",
    },
    {
        "id": "boundary_negative_index", "category": "boundary",
        "name": "负数索引处理",
        "code": "nums = [1, 2, 3]\nprint(nums[-1])\n",
        "expect": "1",
        "task": "case_boundary_negative_index.py 期望输出 1，请修复",
    },
    {
        "id": "boundary_division_remainder", "category": "boundary",
        "name": "取模负数边界",
        "code": "def mod(a, b):\n    return a % b\n\nprint(mod(-7, 3))\n",
        "expect": "2",
        "task": "case_boundary_division_remainder.py 期望输出 2（数学意义上的非负余数），请修复",
    },
    {
        "id": "boundary_zero_input", "category": "boundary",
        "name": "输入为 0 的特判",
        "code": "def safe_sqrt(n):\n    return n ** 0.5\n\nprint(round(safe_sqrt(0)))\n",
        "expect": "0",
        "task": "case_boundary_zero_input.py 输入 0 时应输出 0，请确认并修复",
    },
    {
        "id": "boundary_long_string", "category": "boundary",
        "name": "长字符串截断",
        "code": "def shorten(s, n):\n    return s[:n]\n\nprint(shorten('abcdefghij', 3))\n",
        "expect": "abc",
        "task": "case_boundary_long_string.py 期望输出 abc，请修复",
    },
    {
        "id": "boundary_division_by_input", "category": "boundary",
        "name": "输入值作为除数",
        "code": "def half(n):\n    return 100 / n\n\nprint(half(0))\n",
        "expect": "0",
        "task": "case_boundary_division_by_input.py 除数为 0 时应输出 0 而不是报错，请修复",
    },
]

# 保持向后兼容：v1 的 self_heal_bench.py 按 CASES[:n] 用
ALL_CASES = CASES

CATEGORY_ORDER = ["syntax", "runtime", "logic", "dependency", "boundary"]
CATEGORY_DESC = {
    "syntax": "语法错误（stderr 直接指明，修起来最容易）",
    "runtime": "运行时异常（需正确理解异常信息）",
    "logic": "逻辑错误（不报错但结果错，最考验意图理解）",
    "dependency": "依赖/接口错误（调用不存在或用错 API）",
    "boundary": "边界条件（空、除零、负数、极端输入）",
}


def cases_by_category() -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {c: [] for c in CATEGORY_ORDER}
    for c in CASES:
        out[c["category"]].append(c)
    return out


def subset(n_per_category: int) -> list[dict]:
    """每类各取 n 个用例（用于快速冒烟）。"""
    picked = []
    for c, items in cases_by_category().items():
        picked.extend(items[:n_per_category])
    return picked