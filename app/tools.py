"""工具层：Agent 可调用的全部工具，统一返回 (ok, output) 结构。

设计要点（面试讲法）：
- 每个工具都是纯函数，独立可单测，不依赖 LLM；
- 统一返回 (ok, output)，output 是给模型看的观察结果，ok 用于记录成败；
- 危险操作（写文件到 workspace 之外）在工具层就拦截，不靠 Prompt 约束。
"""
import os
import subprocess
from pathlib import Path

from . import config


def _workspace() -> Path:
    """解析工作目录为绝对路径，保证沙箱边界可控。"""
    p = Path(config.WORKSPACE).resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p


def _safe_path(name: str) -> Path:
    """把相对路径约束在 workspace 内，拒绝 ../ 逃逸。"""
    base = _workspace()
    target = (base / name).resolve()
    if base not in target.parents and target != base:
        raise ValueError(f"路径越界：{name} 超出 workspace 沙箱")
    return target


# ---------- 文件工具 ----------

def list_files(path: str = "") -> tuple[bool, str]:
    """列出 workspace（或子目录）下的文件结构。"""
    try:
        target = _safe_path(path) if path else _workspace()
        if not target.exists():
            return False, f"路径不存在：{path}"
        if target.is_file():
            return True, f"{path}（文件）"
        lines = []
        for p in sorted(target.rglob("*")):
            if p.is_file():
                rel = p.relative_to(_workspace())
                lines.append(f"  {rel}  ({p.stat().st_size} B)")
        return True, "workspace 文件列表：\n" + ("\n".join(lines) or "  （空）")
    except (ValueError, OSError) as e:
        return False, str(e)


def read_file(name: str) -> tuple[bool, str]:
    """读取 workspace 内文件内容，带行号。"""
    try:
        target = _safe_path(name)
        if not target.exists():
            return False, f"文件不存在：{name}"
        content = target.read_text(encoding="utf-8", errors="replace")
        numbered = "\n".join(f"{i+1:4d} | {ln}" for i, ln in enumerate(content.splitlines()))
        return True, f"{name} 内容：\n{numbered}"
    except (ValueError, OSError) as e:
        return False, str(e)


def write_file(name: str, content: str) -> tuple[bool, str]:
    """在 workspace 内创建 / 覆盖文件。"""
    try:
        target = _safe_path(name)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return True, f"已写入 {name}（{len(content)} 字符）"
    except (ValueError, OSError) as e:
        return False, str(e)


def edit_file(name: str, old: str, new: str) -> tuple[bool, str]:
    """局部替换：把文件中的 old 片段替换为 new（单次替换，避免误改多处）。"""
    try:
        target = _safe_path(name)
        if not target.exists():
            return False, f"文件不存在：{name}"
        content = target.read_text(encoding="utf-8")
        if old not in content:
            return False, f"未找到目标片段：{old[:50]}..."
        if content.count(old) > 1:
            return False, f"目标片段出现 {content.count(old)} 次，请提供更精确的上下文"
        target.write_text(content.replace(old, new, 1), encoding="utf-8")
        return True, f"已替换 {name} 中的 1 处片段"
    except (ValueError, OSError) as e:
        return False, str(e)


def search_code(keyword: str, path: str = "") -> tuple[bool, str]:
    """按关键词搜索 workspace 内文件内容，返回 文件名:行号 + 匹配行。"""
    try:
        base = _safe_path(path) if path else _workspace()
        if not base.exists():
            return False, f"路径不存在：{path}"
        hits = []
        for p in base.rglob("*"):
            if not p.is_file() or p.suffix not in (".py", ".txt", ".md", ".js", ".json"):
                continue
            try:
                for i, ln in enumerate(p.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                    if keyword in ln:
                        hits.append(f"  {p.relative_to(_workspace())}:{i}  {ln.strip()[:80]}")
            except OSError:
                continue
        if not hits:
            return True, f"未找到含「{keyword}」的内容"
        return True, f"匹配「{keyword}」：\n" + "\n".join(hits[:50])
    except (ValueError, OSError) as e:
        return False, str(e)


# ---------- 执行工具 ----------

def run_python(name: str) -> tuple[bool, str]:
    """在 workspace 内用 subprocess 执行一个 Python 文件，捕获 stdout/stderr/returncode。"""
    try:
        target = _safe_path(name)
        if not target.exists():
            return False, f"文件不存在：{name}"
        proc = subprocess.run(
            ["python", str(target)],
            capture_output=True, text=True, timeout=config.EXEC_TIMEOUT,
            cwd=str(_workspace()),
        )
        out = proc.stdout.strip()
        err = proc.stderr.strip()
        if proc.returncode == 0:
            return True, f"执行成功（exit 0）\n{out}"
        # 失败：把 stderr + returncode 一起返回，让模型据此修复
        return False, f"执行失败（exit {proc.returncode}）\nstdout: {out}\nstderr: {err}"
    except subprocess.TimeoutExpired:
        return False, f"执行超时（>{config.EXEC_TIMEOUT}s），可能死循环，请检查代码"
    except (ValueError, OSError) as e:
        return False, str(e)


def run_command(cmd: str) -> tuple[bool, str]:
    """执行一条 shell 命令（用于运行测试、查看状态等）。"""
    # 白名单命令，降低误执行风险
    allowed_prefix = ("python ", "pytest ", "pip ", "ls ", "dir ", "cat ", "echo ")
    if not cmd.startswith(allowed_prefix):
        return False, f"命令被拦截：仅允许 {allowed_prefix} 开头的命令"
    try:
        proc = subprocess.run(
            cmd, shell=True, capture_output=True, text=True,
            timeout=config.EXEC_TIMEOUT, cwd=str(_workspace()),
        )
        out = (proc.stdout + proc.stderr).strip()
        status = "成功" if proc.returncode == 0 else f"失败(exit {proc.returncode})"
        return proc.returncode == 0, f"{status}\n{out[:2000]}"
    except subprocess.TimeoutExpired:
        return False, f"命令超时（>{config.EXEC_TIMEOUT}s）"
    except OSError as e:
        return False, str(e)


# ---------- 工具注册表 ----------
# 每个工具：name -> (函数, 参数 schema, 描述)。描述会写进 system prompt 供模型参考。
TOOLS = {
    "list_files": (list_files, ["path"], "列出 workspace 下的文件结构"),
    "read_file": (read_file, ["name"], "读取文件内容（带行号）"),
    "write_file": (write_file, ["name", "content"], "创建或覆盖一个文件"),
    "edit_file": (edit_file, ["name", "old", "new"], "局部替换文件中的片段"),
    "search_code": (search_code, ["keyword", "path"], "按关键词搜索文件内容"),
    "run_python": (run_python, ["name"], "执行 workspace 内的 Python 文件"),
    "run_command": (run_command, ["cmd"], "执行白名单 shell 命令（如 pytest）"),
}


def call_tool(name: str, args: dict) -> tuple[bool, str]:
    """统一工具调用入口：按名字分发，返回 (ok, observation)。"""
    if name not in TOOLS:
        return False, f"未知工具：{name}，可用工具：{', '.join(TOOLS)}"
    fn, _, _ = TOOLS[name]
    try:
        return fn(**args)
    except TypeError as e:
        return False, f"参数错误：{e}"
