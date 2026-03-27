import re
import shlex
from dataclasses import dataclass
from typing import Any, Optional

_VAR_RE = re.compile(
    r'\$\{([A-Za-z_][A-Za-z0-9_]*)\}|\$([A-Za-z_][A-Za-z0-9_]*)')


@dataclass(frozen=True)
class PretrainScriptConfig:
    variables: dict[str, str]
    blocks: dict[str, str]
    argv: list[str]
    flags: dict[str, Any]


def _strip_quotes(v: str) -> str:
    v = v.strip()
    if len(v) >= 2 and ((v[0] == v[-1] == '"') or (v[0] == v[-1] == "'")):
        return v[1:-1]
    return v


def _expand_vars(s: str, vars_: dict[str, str]) -> str:

    def repl(m: re.Match) -> str:
        name = m.group(1) or m.group(2)
        if name in vars_:
            return str(vars_[name])
        return m.group(0)

    return _VAR_RE.sub(repl, s)


def _parse_assignments(lines: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    in_block = False
    for ln in lines:
        line = ln.strip()
        if not line or line.startswith('#'):
            continue
        if in_block:
            if line == '"':
                in_block = False
            continue
        if re.match(r'^[A-Za-z_][A-Za-z0-9_]*\s*=\s*"\s*$', line):
            in_block = True
            continue
        if '=' not in line:
            continue
        if line.startswith('export '):
            line = line[len('export '):].strip()
        k, v = line.split('=', 1)
        k = k.strip()
        if not re.match(r'^[A-Za-z_][A-Za-z0-9_]*$', k):
            continue
        v = v.strip()
        if v.endswith(';'):
            v = v[:-1].strip()
        out[k] = _strip_quotes(v)
    return out


def _parse_quoted_blocks(lines: list[str]) -> dict[str, str]:
    blocks: dict[str, str] = {}
    i = 0
    while i < len(lines):
        ln = lines[i]
        m = re.match(r'^([A-Za-z_][A-Za-z0-9_]*)\s*=\s*"\s*$', ln.strip())
        if not m:
            i += 1
            continue
        name = m.group(1)
        i += 1
        buf: list[str] = []
        while i < len(lines):
            cur = lines[i]
            if cur.strip() == '"':
                break
            buf.append(cur)
            i += 1
        blocks[name] = '\n'.join(buf)
        while i < len(lines) and lines[i].strip() != '"':
            i += 1
        i += 1
    return blocks


def _extract_torchrun_section(lines: list[str]) -> list[str]:
    out: list[str] = []
    in_cmd = False
    for ln in lines:
        if not in_cmd and 'torchrun' in ln and 'pretrain_gpt.py' in ln:
            in_cmd = True
        if in_cmd:
            out.append(ln)
            if 'tee' in ln and 'TRAIN_LOG_PATH' in ln:
                break
    return out


def _normalize_arg_text(s: str) -> str:
    s = s.replace('\\\n', '\n')
    s = s.replace('\\t', '\t')
    s = s.replace('\\n', '\n')
    s = s.replace('\\', ' ')
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def parse_pretrain_script(path: str) -> PretrainScriptConfig:
    with open(path) as f:
        lines = f.read().splitlines()

    variables = _parse_assignments(lines)
    blocks = _parse_quoted_blocks(lines)
    torchrun_lines = _extract_torchrun_section(lines)
    torchrun_text = '\n'.join(torchrun_lines)

    used_block_names: set[str] = set()
    for m in re.finditer(r'\$([A-Za-z_][A-Za-z0-9_]*)', torchrun_text):
        used_block_names.add(m.group(1))
    for m in re.finditer(r'\$\{([A-Za-z_][A-Za-z0-9_]*)\}', torchrun_text):
        used_block_names.add(m.group(1))

    # Replace block variable references with block content
    for bn in used_block_names:
        if bn in blocks:
            torchrun_text = torchrun_text.replace(f'${bn}', blocks[bn])
            torchrun_text = torchrun_text.replace(f'${{{bn}}}', blocks[bn])

    merged = torchrun_text
    merged = _expand_vars(merged, variables)
    merged = _normalize_arg_text(merged)

    argv = shlex.split(merged, posix=True)
    argv = [a for a in argv if a != 'torchrun' and a != 'pretrain_gpt.py']
    argv = [a for a in argv if not a.startswith('2>&1')]
    argv = [a for a in argv if a != '|' and a != 'tee']

    flags: dict[str, Any] = {}
    i = 0
    while i < len(argv):
        a = argv[i]
        if not a.startswith('--'):
            i += 1
            continue
        if i + 1 < len(argv) and not argv[i + 1].startswith('--'):
            key = a
            val = argv[i + 1]
            if key in flags:
                if isinstance(flags[key], list):
                    flags[key].append(val)
                else:
                    flags[key] = [flags[key], val]
            else:
                flags[key] = val
            i += 2
            continue
        flags[a] = True
        i += 1

    return PretrainScriptConfig(variables=variables,
                                blocks=blocks,
                                argv=argv,
                                flags=flags)


def get_flag(cfg: PretrainScriptConfig,
             name: str,
             default: Optional[Any] = None) -> Any:
    return cfg.flags.get(name, default)


def get_int(cfg: PretrainScriptConfig,
            name: str,
            default: Optional[int] = None) -> Optional[int]:
    v = get_flag(cfg, name, default)
    if v is None or v is True:
        return default
    try:
        return int(v)
    except Exception:
        return default


def get_float(cfg: PretrainScriptConfig,
              name: str,
              default: Optional[float] = None) -> Optional[float]:
    v = get_flag(cfg, name, default)
    if v is None or v is True:
        return default
    try:
        return float(v)
    except Exception:
        return default


def get_bool(cfg: PretrainScriptConfig,
             name: str,
             default: Optional[bool] = None) -> Optional[bool]:
    v = get_flag(cfg, name, None)
    if v is None:
        return default
    if v is True:
        return True
    if isinstance(v, str):
        if v.lower() in ('1', 'true', 'yes', 'y', 'on'):
            return True
        if v.lower() in ('0', 'false', 'no', 'n', 'off'):
            return False
    return default
