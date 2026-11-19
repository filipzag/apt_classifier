"""Exact procedure artefacts in report and ATT&CK text (development study, docs 21.3.32).

Registry keys, file paths and names, named pipes, mutexes, service and task names, debug (PDB)
paths, user agents and DLL export names are pulled out of a statement and normalised, so the same
artefact written by two authors compares equal: hive spellings unified, user and environment
folders folded into ``%appdata%``-style variables, GUIDs, SIDs and long hex runs replaced by
placeholders, case folded. The ``ArtefactIndex`` credits each artefact to the groups whose ATT&CK
texts carry it and scores a report by the rarity (inverse group frequency) of what it shares with
each group. Nothing in the pipeline imports this module.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

KINDS = (
    "registry",
    "path",
    "file",
    "pipe",
    "mutex",
    "service",
    "task",
    "pdb",
    "user_agent",
    "export",
)

_QUOTES = "\"'`“”‘’«»"
_CODE_TAG = re.compile(r"</?code>", re.I)
_HIVES = {
    "hkey_local_machine": "hklm",
    "hkey_current_user": "hkcu",
    "hkey_classes_root": "hkcr",
    "hkey_users": "hku",
    "hkey_current_config": "hkcc",
}
_GUID = re.compile(r"\{?[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\}?", re.I)
_SID = re.compile(r"\bs-1-5-[0-9-]{6,}\b", re.I)
_HEX = re.compile(r"\b[0-9a-f]{12,}\b", re.I)
_DIGITS = re.compile(r"\d{5,}")
_FOLDERS = (
    (re.compile(r"^[a-z]:\\users\\[^\\]+\\appdata\\roaming(?=\\|$)"), "%appdata%"),
    (re.compile(r"^[a-z]:\\users\\[^\\]+\\appdata\\local\\temp(?=\\|$)"), "%temp%"),
    (re.compile(r"^[a-z]:\\users\\[^\\]+\\appdata\\local(?=\\|$)"), "%localappdata%"),
    (re.compile(r"^[a-z]:\\users\\public(?=\\|$)"), "%public%"),
    (re.compile(r"^[a-z]:\\users\\[^\\]+(?=\\|$)"), "%userprofile%"),
    (re.compile(r"^[a-z]:\\programdata(?=\\|$)"), "%programdata%"),
    (re.compile(r"^[a-z]:\\windows\\system32(?=\\|$)"), "%system32%"),
    (re.compile(r"^[a-z]:\\windows\\syswow64(?=\\|$)"), "%syswow64%"),
    (re.compile(r"^[a-z]:\\windows\\temp(?=\\|$)"), "%windir%\\temp"),
    (re.compile(r"^[a-z]:\\windows(?=\\|$)"), "%windir%"),
    (re.compile(r"^[a-z]:\\program files x86(?=\\|$)"), "%programfiles(x86)%"),
    (re.compile(r"^[a-z]:\\program files(?=\\|$)"), "%programfiles%"),
    (re.compile(r"^%tmp%(?=\\|$)"), "%temp%"),
    (re.compile(r"^%systemroot%(?=\\|$)"), "%windir%"),
    (re.compile(r"^%windir%\\system32(?=\\|$)"), "%system32%"),
    (re.compile(r"^%userprofile%\\appdata\\roaming(?=\\|$)"), "%appdata%"),
    (re.compile(r"^%userprofile%\\appdata\\local\\temp(?=\\|$)"), "%temp%"),
    (re.compile(r"^%userprofile%\\appdata\\local(?=\\|$)"), "%localappdata%"),
    (re.compile(r"^%localappdata%\\temp(?=\\|$)"), "%temp%"),
)
_EXTENSIONS = (
    "dll|exe|ps1|psm1|bat|cmd|vbs|vbe|js|jse|wsf|hta|lnk|scr|sys|dat|tmp|bin|log|ini|cfg|conf|"
    "xml|json|py|sh|elf|so|jar|zip|rar|7z|iso|img|vhd|vhdx|docm|dotm|doc|docx|xls|xlsm|xlsx|"
    "ppt|pptx|rtf|chm|msi|cab|cpl|ocx|pif|db|sqlite|ps|dmg|pkg|plist|apk|php|aspx|asp|jsp|sct|"
    "inf|reg|txt|pdf|cr|sxd|key|crt|pem|enc|dmp|lock|pyc|ko"
)
# one path segment: no separators, quotes or sentence punctuation (dots stay for file names)
_SEGMENT = r"[^\\/\s\"'`“”‘’<>|,;:()\[\]{}*?]+"
_SPACED = (
    r"(?:\s(?!(?:to|and|or|in|on|at|for|with|from|as|is|was|the|a|an|which|that|via|by)\b)"
    r"(?![a-zA-Z]:)(?=\S{2})" + _SEGMENT + r")*"
)
_WIN_PATH = re.compile(
    r"(?<![\w%])((?:[a-zA-Z]:|%[A-Za-z_()0-9]+%|\\\\\?)(?:\\" + _SEGMENT + _SPACED + r")+\\?)"
)
_NIX_PATH = re.compile(
    r"(?<![\w/.~])((?:~|/(?:tmp|var|etc|usr|home|dev|opt|lib|bin|sbin|root|private|Library|"
    r"Users|Applications|System|proc|run|mnt))(?:/[^\s\"'`“”‘’<>|,;:()\[\]{}*?]+)+)"
)
_FILE = re.compile(r"(?<![\w\\/.%-])([\w][\w.$~-]{0,80}\.(?:" + _EXTENSIONS + r"))(?![\w-])", re.I)
_REGISTRY = re.compile(
    r"(?<![\w])((?:HKEY_[A-Z_]+|HKLM|HKCU|HKCR|HKU|HKCC)(?::)?(?:\\{1,2}"
    + _SEGMENT
    + _SPACED
    + r")+)",
    re.I,
)
_PIPE = re.compile(r"\\{1,2}(?:\.|\?)?\\{1,2}pipe\\{1,2}([^\s\"'`“”‘’<>|,;()]+)", re.I)
_PIPE_NAMED = re.compile(
    r"\bnamed pipe (?:called |named )?["
    + _QUOTES
    + r"]([^"
    + _QUOTES
    + r"]{2,80})["
    + _QUOTES
    + r"]",
    re.I,
)
_MUTEX = re.compile(
    r"\bmutex(?:es|e)?\b(?:\s+(?:name|names|named|called|of|value|string|such as|like|with the name|"
    r"object|is|was))*\s*[:,]?\s*[" + _QUOTES + r"]([^" + _QUOTES + r"]{2,100})[" + _QUOTES + r"]",
    re.I,
)
_MUTEX_PREFIXED = re.compile(r"(?<![\w\\])((?:Global|Local)\\[^\s\"'`“”‘’<>|,;()]{2,100})")
_SERVICE = re.compile(
    r"\bservices?\b(?:\s+(?:name|names|named|called|with the name))+\s*[:,]?\s*"
    r"(?:[" + _QUOTES + r"]([^" + _QUOTES + r"]{2,80})[" + _QUOTES + r"]|([A-Za-z][\w.-]{3,60}))",
    re.I,
)
_SERVICE_COMMAND = re.compile(
    r"\b(?:sc(?:\.exe)?\s+create|New-Service\s+(?:-Name\s+)?)\s*[" + _QUOTES + r"]?([\w.-]{2,60})",
    re.I,
)
_TASK = re.compile(
    r"\b(?:scheduled\s+)?tasks?\b(?:\s+(?:name|names|named|called|with the name))+\s*"
    r"[:,]?\s*(?:["
    + _QUOTES
    + r"]([^"
    + _QUOTES
    + r"]{2,80})["
    + _QUOTES
    + r"]|([A-Za-z][\w.\\-]{3,60}))",
    re.I,
)
_TASK_COMMAND = re.compile(
    r"/tn\s+(?:[" + _QUOTES + r"]([^" + _QUOTES + r"]{2,80})[" + _QUOTES + r"]|([^\s/]{2,80}))",
    re.I,
)
# folder segments exclude the separators, or a line full of slashes backtracks exponentially
_PDB = re.compile(
    r"((?:[a-zA-Z]:\\|\\\\|/)?(?:[^\s\"'`“”‘’<>|,;()\\/]+[\\/]+)*[^\s\"'`“”‘’<>|,;()\\/]+\.pdb)\b",
    re.I,
)
_USER_AGENT = re.compile(r"(Mozilla/\d\.\d \([^)]{4,200}\)(?: (?:[\w.-]+/[\w.]+|\([^)]{1,60}\)))*)")
_EXPORT = re.compile(
    r"\b(?:export(?:ed)?(?: function)?(?: named| called)?|exports? (?:a |the )?(?:function )?(?:named |called )?|"
    r"entry ?point(?: named| called)?)\s+["
    + _QUOTES
    + r"]?([A-Za-z_#][\w@#]{1,60})["
    + _QUOTES
    + r"]?",
    re.I,
)
_EXPORT_RUNDLL = re.compile(
    r"rundll32(?:\.exe)?\s+["
    + _QUOTES
    + r"]?[^\s,]+\.\w{2,4}["
    + _QUOTES
    + r"]?\s*,\s*#?([A-Za-z_#][\w@#]{1,60})",
    re.I,
)
# names too common in benign Windows and in everyone's tradecraft to be an artefact of anyone
_COMMON_FILES = frozenset(
    [
        "cmd.exe",
        "powershell.exe",
        "pwsh.exe",
        "rundll32.exe",
        "regsvr32.exe",
        "mshta.exe",
        "wmic.exe",
        "schtasks.exe",
        "certutil.exe",
        "bitsadmin.exe",
        "vssadmin.exe",
        "netsh.exe",
        "whoami.exe",
        "reg.exe",
        "net.exe",
        "net1.exe",
        "nltest.exe",
        "psexec.exe",
        "wscript.exe",
        "cscript.exe",
        "explorer.exe",
        "svchost.exe",
        "lsass.exe",
        "services.exe",
        "winlogon.exe",
        "csrss.exe",
        "smss.exe",
        "taskhost.exe",
        "taskhostw.exe",
        "conhost.exe",
        "dllhost.exe",
        "msiexec.exe",
        "tasklist.exe",
        "taskkill.exe",
        "ipconfig.exe",
        "systeminfo.exe",
        "ping.exe",
        "nslookup.exe",
        "arp.exe",
        "route.exe",
        "netstat.exe",
        "quser.exe",
        "query.exe",
        "at.exe",
        "sc.exe",
        "wevtutil.exe",
        "bcdedit.exe",
        "ntdsutil.exe",
        "esentutl.exe",
        "expand.exe",
        "makecab.exe",
        "forfiles.exe",
        "mstsc.exe",
        "notepad.exe",
        "calc.exe",
        "iexplore.exe",
        "chrome.exe",
        "firefox.exe",
        "msedge.exe",
        "outlook.exe",
        "winword.exe",
        "excel.exe",
        "powerpnt.exe",
        "eqnedt32.exe",
        "installutil.exe",
        "regasm.exe",
        "regsvcs.exe",
        "msbuild.exe",
        "cmstp.exe",
        "odbcconf.exe",
        "wuauclt.exe",
        "rdpclip.exe",
        "userinit.exe",
        "ntdll.dll",
        "kernel32.dll",
        "user32.dll",
        "advapi32.dll",
        "ws2_32.dll",
        "wininet.dll",
        "winhttp.dll",
        "shell32.dll",
        "ole32.dll",
        "comsvcs.dll",
        "dbghelp.dll",
        "version.dll",
        "ntds.dit",
        "sam",
        "system",
        "security",
        "lsass.dmp",
        "hosts",
        "readme.txt",
        "desktop.ini",
        "thumbs.db",
        "update.exe",
        "setup.exe",
        "install.exe",
        "installer.exe",
        "svc.exe",
        "service.exe",
        "test.exe",
        "a.exe",
        "1.exe",
        "run.bat",
        "start.bat",
        "test.bat",
        "1.bat",
        "a.bat",
        "payload.exe",
        "payload.dll",
        "malware.exe",
    ]
)
_COMMON_EXPORTS = frozenset(
    [
        "dllregisterserver",
        "dllunregisterserver",
        "dllmain",
        "dllentrypoint",
        "dllinstall",
        "dllgetclassobject",
        "dllcanunloadnow",
        "control_rundll",
        "main",
        "start",
        "run",
        "entry",
        "init",
        "export",
    ]
)
# folders every intruder writes to: a path artefact needs more than these
_GENERIC_DIRS = frozenset(
    [
        "/tmp",
        "/var/tmp",
        "/var/log",
        "/etc",
        "/usr/bin",
        "/usr/local/bin",
        "/usr/sbin",
        "/bin",
        "/sbin",
        "/dev/shm",
        "/root",
        "/home",
        "/opt",
        "/var",
        "/usr",
        "/lib",
        "/etc/init.d",
        "/etc/cron.d",
        "/etc/systemd/system",
        "/lib/systemd/system",
        "/dev/null",
        "/bin/sh",
        "/bin/bash",
        "c:\\temp",
        "c:\\users",
        "%temp%",
        "%appdata%",
        "%localappdata%",
        "%programdata%",
        "%public%",
        "%userprofile%",
        "%windir%",
        "%system32%",
        "%syswow64%",
        "%programfiles%",
        "%programfiles(x86)%",
        "%windir%\\temp",
        "%appdata%\\microsoft\\windows\\start",
        "menu\\programs\\startup",
        "%programdata%\\microsoft\\windows\\start",
        "menu\\programs\\startup",
    ]
)
_GENERIC_NAMES = frozenset(
    [
        "the",
        "a",
        "an",
        "this",
        "that",
        "it",
        "its",
        "these",
        "those",
        "such",
        "their",
        "his",
        "her",
        "our",
        "malware",
        "adversary",
        "operation",
        "backdoor",
        "tool",
        "payload",
        "service",
        "task",
        "file",
    ]
)
# prose words a file-name pattern catches before an extension ("and.log", "in.zip")
_STOP_STEMS = frozenset(
    ["and", "or", "in", "on", "at", "to", "of", "the", "for", "is", "as", "by", "via", "from"]
)


@dataclass(frozen=True)
class Artefact:
    kind: str
    value: str

    def key(self) -> str:
        return f"{self.kind}:{self.value}"


def _placeholders(value: str) -> str:
    value = _GUID.sub("<guid>", value)
    value = _SID.sub("<sid>", value)
    value = _HEX.sub("<hex>", value)
    return _DIGITS.sub("<n>", value)


def _strip(value: str) -> str:
    return value.strip().strip(_QUOTES).rstrip(".,;:)]}>").strip()


def normalise_registry(raw: str) -> str:
    value = _strip(raw).replace("/", "\\").lower()
    value = re.sub(r"\\{2,}", r"\\", value).rstrip("\\")
    value = value.replace(":\\", "\\", 1) if re.match(r"^hk\w+:\\", value) else value
    head, _, rest = value.partition("\\")
    head = _HIVES.get(head, head)
    rest = re.sub(r"(?<=\\)wow6432node\\", "", "\\" + rest)[1:] if rest else rest
    rest = re.sub(r"^software\\wow6432node\\", "software\\\\", rest)
    return _placeholders(f"{head}\\{rest}" if rest else head)


def normalise_path(raw: str) -> str:
    value = _strip(raw)
    if "\\" not in value:
        return _placeholders(re.sub(r"/{2,}", "/", value))  # Unix: case matters
    value = re.sub(r"\\{2,}", r"\\", value.lower()).rstrip("\\")
    value = re.sub(r"^\\\\?\?\\", "", value)
    for pattern, replacement in _FOLDERS:
        value = pattern.sub(lambda _m, r=replacement: r, value)
    return _placeholders(value)


def _name(value: str) -> str | None:
    value = re.sub(r"\s+", " ", _strip(value)).lower()
    if len(value) < 3 or value in _GENERIC_NAMES or value.split(" ")[0] in _GENERIC_NAMES:
        return None
    return _placeholders(value)


def _file_name(leaf: str) -> bool:
    stem = leaf.split(".")[0]
    return leaf not in _COMMON_FILES and len(stem) >= 2 and stem not in _STOP_STEMS


def _specific_path(value: str) -> bool:
    separator = "\\" if "\\" in value else "/"
    parts = [p for p in value.split(separator) if p]
    return len(parts) >= 2 and value.rstrip(separator) not in _GENERIC_DIRS


def extract_artefacts(text: str) -> set[Artefact]:
    """The normalised artefacts one statement names (empty for most prose)."""
    text = _CODE_TAG.sub(" ", text or "")
    text = re.sub(r"program files \(x86\)", "Program Files x86", text, flags=re.I)
    out: set[Artefact] = set()
    registry_spans: list[tuple[int, int]] = []
    for match in _REGISTRY.finditer(text):
        value = normalise_registry(match.group(1))
        if "\\" in value:
            out.add(Artefact("registry", value))
            registry_spans.append(match.span(1))

    def inside_registry(start: int) -> bool:
        return any(a <= start < b for a, b in registry_spans)

    for pattern in (_WIN_PATH, _NIX_PATH):
        for match in pattern.finditer(text):
            if inside_registry(match.start(1)):
                continue
            raw = match.group(1)
            if _strip(raw).lower().endswith(".pdb"):
                continue
            value = normalise_path(raw)
            separator = "\\" if "\\" in value else "/"
            parts = [p for p in value.split(separator) if p]
            if _specific_path(value):
                out.add(Artefact("path", value))
            leaf = parts[-1] if parts else ""
            if re.search(r"\.(?:" + _EXTENSIONS + r")$", leaf, re.I):
                parent = ("/" if value.startswith("/") else "") + separator.join(parts[:-1])
                if _specific_path(parent):
                    out.add(Artefact("path", parent))
                if _file_name(leaf.lower()):
                    out.add(Artefact("file", _placeholders(leaf.lower())))
    for match in _FILE.finditer(text):
        if inside_registry(match.start(1)):
            continue
        name = match.group(1).lower()
        if name.endswith(".pdb") or not _file_name(name):
            continue
        out.add(Artefact("file", _placeholders(name)))
    for pattern in (_PIPE, _PIPE_NAMED):
        for match in pattern.finditer(text):
            name = _name(match.group(1).replace("\\", "/").split("/")[-1])
            if name:
                out.add(Artefact("pipe", name))
    for match in _MUTEX.finditer(text):
        name = _name(match.group(1))
        if name:
            out.add(Artefact("mutex", name))
    if re.search(r"\bmutex", text, re.I):
        for match in _MUTEX_PREFIXED.finditer(text):
            name = _name(match.group(1).replace("\\\\", "\\"))
            if name:
                out.add(Artefact("mutex", name))
    for pattern in (_SERVICE, _SERVICE_COMMAND):
        for match in pattern.finditer(text):
            name = _name(next(g for g in match.groups() if g))
            if name:
                out.add(Artefact("service", name))
    for pattern in (_TASK, _TASK_COMMAND):
        for match in pattern.finditer(text):
            name = _name(next(g for g in match.groups() if g))
            if name:
                out.add(Artefact("task", name))
    for match in _PDB.finditer(text) if ".pdb" in text.lower() else ():
        value = normalise_path(match.group(1))
        out.add(Artefact("pdb", value))
        leaf = re.split(r"[\\/]", value)[-1]
        if leaf != value:
            out.add(Artefact("pdb", leaf))
    for match in _USER_AGENT.finditer(text):
        out.add(Artefact("user_agent", re.sub(r"\s+", " ", _strip(match.group(1)))))
    for pattern in (_EXPORT, _EXPORT_RUNDLL):
        for match in pattern.finditer(text):
            whole = match.group(0)
            # an unquoted word after "exports" is usually prose ("exports the requests ...")
            if pattern is _EXPORT and not (
                whole[-1] in _QUOTES or re.search(r"\b(?:named|called)\b", whole, re.I)
            ):
                continue
            name = match.group(1).lower().lstrip("#")
            if len(name) >= 2 and name not in _COMMON_EXPORTS and name not in _GENERIC_NAMES:
                out.add(Artefact("export", name))
    return out


class ArtefactIndex:
    """Artefact -> the groups whose ATT&CK texts carry it, scored by rarity.

    ``records`` are the text member's corpus records (``text`` and the credited ``groups``); an
    artefact credited to more than ``max_groups`` groups is everyone's and never scores."""

    def __init__(
        self,
        records: Iterable[Mapping[str, object]],
        group_ids: Sequence[str],
        *,
        max_groups: int = 8,
        cache: dict[str, frozenset[Artefact]] | None = None,
    ) -> None:
        self.group_ids = tuple(group_ids)
        self.max_groups = max_groups
        groups_of: dict[str, set[str]] = defaultdict(set)
        cache = cache if cache is not None else {}
        for record in records:
            credited = [g for g in record.get("groups") or () if isinstance(g, str)]
            if not credited:
                continue
            text = str(record.get("text") or "")
            found = cache.get(text)
            if found is None:
                found = cache[text] = frozenset(extract_artefacts(text))
            for artefact in found:
                groups_of[artefact.key()].update(credited)
        self.groups_of = {k: frozenset(v) for k, v in groups_of.items()}
        total = max(1, len(self.group_ids))
        self.weight = {
            key: math.log(total / len(groups))
            for key, groups in self.groups_of.items()
            if len(groups) <= max_groups
        }

    def matches(
        self, artefacts: Iterable[Artefact]
    ) -> list[tuple[Artefact, float, frozenset[str]]]:
        """The report artefacts the index knows, with their weight and groups."""
        out = []
        for artefact in sorted(set(artefacts), key=Artefact.key):
            weight = self.weight.get(artefact.key())
            if weight is not None:
                out.append((artefact, weight, self.groups_of[artefact.key()]))
        return out

    def scores(self, artefacts: Iterable[Artefact]) -> dict[str, float]:
        """Per group, the summed rarity of the report artefacts it shares (only groups > 0)."""
        totals: dict[str, float] = defaultdict(float)
        for _artefact, weight, groups in self.matches(artefacts):
            for group in groups:
                totals[group] += weight
        return dict(totals)
