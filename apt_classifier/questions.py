import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from math import log2
from typing import Any

from apt_classifier.evidence.models import EvidenceStatus
from apt_classifier.llm import OpenRouterClient, supports_seed
from apt_classifier.platforms import PlatformScope

INTERVIEW_REVISION = 3

_TACTIC_GUIDANCE = {
    "initial-access": "Review email security, identity, VPN, and internet-facing application logs for the initial entry path.",
    "execution": "Review process creation, command-line, script-block, application-control, and EDR execution telemetry.",
    "persistence": "Inspect services, scheduled tasks, autoruns, registry changes, startup items, and identity persistence mechanisms.",
    "privilege-escalation": "Review token, account, service, driver, and privileged process activity around the incident window.",
    "defense-evasion": "Inspect security-control changes, deleted or modified logs, obfuscated execution, and trusted-binary abuse.",
    "credential-access": "Review authentication, process-access, credential-store, directory-service, and secret-access telemetry.",
    "discovery": "Review shell, process, directory, cloud-control-plane, network-enumeration, and host-discovery activity.",
    "lateral-movement": "Inspect remote authentication, SMB, RDP, WinRM, SSH, remote service, and administrative-share telemetry.",
    "collection": "Review file access, staging directories, mailbox, database, screenshot, archive, and clipboard telemetry.",
    "command-and-control": "Review DNS, proxy, firewall, TLS, endpoint network, and beaconing telemetry for anomalous destinations.",
    "exfiltration": "Inspect proxy, firewall, DLP, cloud-storage, archive, and large outbound-transfer telemetry.",
    "impact": "Review file modification, encryption, deletion, service interruption, recovery, and destructive-action telemetry.",
}
_TECHNIQUE_COLLECTION_GUIDANCE = {
    "T1003.001": (
        "LSASS access — Microsoft-Windows-Sysmon/Operational Event ID 10; filter TargetImage for lsass.exe and inspect SourceImage, GrantedAccess, user, host, and call trace where collected.",
        "Execution context — Security Event ID 4688 and Sysmon Event ID 1; search for dump tools, suspicious rundll32/comsvcs usage, unusual parent processes, command lines, and output files.",
        "EDR/Defender — review process-handle access, memory-read, credential-theft, and dump-file alerts; correlate source process hashes and signer status.",
    ),
    "T1003.002": (
        "SAM access — Security Event IDs 4656 and 4663 where object-access auditing and SACLs are enabled; inspect access to SAM/SYSTEM hives and shadow copies.",
        "Execution context — Security 4688 and Sysmon 1; search for reg.exe save, secretsdump-like tooling, volume-shadow-copy utilities, suspicious command lines, and resulting hive files.",
    ),
    "T1003.003": (
        "Directory Services — Security Event ID 4662 on domain controllers where Directory Service Access auditing is enabled; inspect replication-right GUID access by non-DC accounts.",
        "Execution context — Security 4688 and Sysmon 1; search for ntdsutil, esentutl, vssadmin, diskshadow, or suspicious tooling accessing NTDS.dit and SYSTEM hive material.",
    ),
    "T1041": (
        "Network egress — Sysmon Event IDs 3 and 22 plus Security 5156 where Filtering Platform Connection auditing is enabled; correlate process, destination, DNS, bytes, and timing.",
        "Proxy, firewall, NDR, and DLP — look for unusual upload volume, rare destinations, long-lived sessions, cloud-storage APIs, and transfers following collection or archive creation.",
    ),
    "T1047": (
        "WMI activity — Microsoft-Windows-WMI-Activity/Operational Event IDs 5857-5861; correlate client process ID, user, namespace, query, consumer, filter, and binding details.",
        "Process execution — Security 4688 and Sysmon 1; inspect wmiprvse.exe children and remote process creation associated with unusual accounts or source hosts.",
    ),
    "T1053.005": (
        "Task creation/change — Security Event IDs 4698, 4699, 4700, 4701, and 4702; inspect task name/XML, action command, arguments, trigger, author, principal, and source account.",
        "Task Scheduler — Microsoft-Windows-TaskScheduler/Operational Event IDs 106, 140, 141, 200, and 201; correlate registration, update/deletion, action start, and completion.",
        "Process execution — Security 4688 and Sysmon 1; inspect schtasks.exe and taskeng.exe/taskhostw.exe process trees, commands, users, and payload paths.",
    ),
    "T1055": (
        "Process injection — Sysmon Event IDs 8, 10, and 25 for CreateRemoteThread, ProcessAccess, and ProcessTampering; inspect source/target images, users, access mask, start address, and call trace where available.",
        "Image/process context — Sysmon Event IDs 1 and 7 plus EDR process-tree and memory telemetry; look for unsigned modules, cross-process memory operations, hollowing, and anomalous parent/child relationships.",
    ),
    "T1059": (
        "Process execution — Security Event ID 4688 and Sysmon Event ID 1; inspect executable, full command line, parent process, user, integrity/elevation, hashes, signer, and host.",
        "EDR — reconstruct the process tree and correlate script, file, registry, and network activity within the same process lineage and incident time window.",
    ),
    "T1059.001": (
        "PowerShell — Microsoft-Windows-PowerShell/Operational Event IDs 4103 and 4104; inspect module/pipeline details, ScriptBlockText, HostApplication, user, and script-block ID. These require module/script-block logging to be enabled.",
        "Process context — Security 4688 and Sysmon 1; filter powershell.exe/pwsh.exe and inspect full command line, parent process, user, host, hashes, encoded content, and download cradles.",
        "AMSI, Defender, and EDR — correlate script content, AMSI detections, child processes, file writes, registry changes, and network connections from the PowerShell process tree.",
    ),
    "T1059.003": (
        "Windows command shell — Security 4688 and Sysmon 1; filter cmd.exe and inspect command line, parent process, user, integrity, child processes, redirected output, and network/file activity.",
    ),
    "T1070.001": (
        "Log clearing — Security Event ID 1102 and System Event ID 104; identify the account, logon session, host, cleared channel, and activity immediately before collection stopped.",
        "Execution context — Security 4688 and Sysmon 1; search for wevtutil, PowerShell Clear-EventLog, or direct event-log service manipulation.",
    ),
    "T1070.004": (
        "File deletion — Sysmon Event IDs 23 and 26 where enabled, plus Security 4663 with object-access auditing; inspect process, user, target path, hash/archive status, and delete access.",
        "Filesystem/EDR — correlate USN journal or MFT changes, mass deletion, secure-delete utilities, process tree, and files created shortly before deletion.",
    ),
    "T1071": (
        "Network communications — Sysmon Event IDs 3 and 22 and Security 5156 where enabled; correlate process, user, DNS query, destination, port, protocol, signer, and connection cadence.",
        "Proxy, firewall, DNS, TLS, and NDR — look for rare domains, new certificates, unusual user agents, beaconing intervals, low-prevalence destinations, and protocol misuse.",
    ),
    "T1071.004": (
        "DNS — Microsoft-Windows-Sysmon/Operational Event ID 22; inspect QueryName, QueryResults, process image, user, host, query volume, entropy, and repeated timing.",
        "Resolver, passive DNS, firewall, and NDR — identify rare/new domains, TXT or oversized queries, high-entropy labels, tunneling patterns, and endpoint-to-query correlation.",
    ),
    "T1078": (
        "Authentication — Security Event IDs 4624, 4625, 4648, and 4672; inspect logon type, source address/workstation, target account, explicit credentials, elevated privileges, and logon ID.",
        "Domain authentication — domain-controller Security Event IDs 4768, 4769, and 4776; correlate Kerberos/NTLM activity, service names, ticket encryption, source host, and unusual account timing.",
        "Identity provider — review successful/failed sign-ins, MFA result, device, IP/ASN, impossible travel, token issuance, consent, and session-risk events.",
    ),
    "T1105": (
        "Download and execution — Sysmon Event IDs 3, 11, and 22 plus Security 4688; correlate network connection/DNS, created file, process command line, parent, hash, signer, and destination.",
        "Proxy, firewall, EDR, and web gateway — search for low-prevalence downloads, mismatched extensions/content types, newly written executables/scripts, and immediate execution.",
    ),
    "T1112": (
        "Registry modification — Sysmon Event IDs 12, 13, and 14 and Security 4657 where registry auditing is enabled; inspect process, user, key/value path, old/new data, and host.",
        "Process context — Security 4688 and Sysmon 1; search for reg.exe, PowerShell registry cmdlets, regsvr32, or unusual processes modifying sensitive keys.",
    ),
    "T1136": (
        "Account creation/change — Security Event IDs 4720, 4722, 4724, 4728, 4732, and 4756 as applicable; inspect creator, new account, group membership, domain, and logon ID.",
        "Identity provider and cloud audit — review user/service-principal creation, role assignment, credential addition, MFA changes, actor, source IP, and correlation ID.",
    ),
    "T1218": (
        "Proxy execution — Security 4688 and Sysmon 1; inspect trusted Microsoft binary command lines, parent/child processes, users, remote content, unusual DLL/script arguments, and network activity.",
        "Image/file evidence — Sysmon 7 and 11 plus EDR signer/prevalence data; correlate loaded modules and newly written payloads with the trusted-binary process tree.",
    ),
    "T1486": (
        "File encryption impact — Security 4663 where object-access auditing is enabled, Sysmon 11/23/26, EDR file telemetry, and storage telemetry; look for rapid high-volume rewrites, extensions, and deletions.",
        "Process context — Security 4688 and Sysmon 1; identify the responsible process tree, account, host spread, command line, binary hash, ransom-note creation, and preceding service/recovery changes.",
    ),
    "T1489": (
        "Service interruption — System Event IDs 7036 and 7040; identify stopped services, start-type changes, host, timing, and preceding process/account activity.",
        "Execution context — Security 4688 and Sysmon 1; search for sc.exe, net.exe, PowerShell service cmdlets, taskkill, or custom binaries stopping security/business services.",
    ),
    "T1490": (
        "Recovery inhibition — Security 4688 and Sysmon 1; search for vssadmin, wbadmin, bcdedit, wmic shadowcopy, diskshadow, PowerShell, and backup-agent commands.",
        "Backup, storage, and EDR — review shadow-copy deletion, backup-policy/job changes, recovery configuration, process tree, account, and host coverage.",
    ),
    "T1543.003": (
        "Service installation — System Event ID 7045 and Security Event ID 4697 where enabled; inspect service name, image path, account, start type, host, and installing subject.",
        "Service/process activity — System 7036/7040 plus Security 4688 and Sysmon 1; correlate service state/start-type changes and services.exe child processes with payload hashes.",
    ),
    "T1547.001": (
        "Run keys/startup folder — Sysmon Event IDs 12, 13, and 14 and Security 4657 where registry auditing is enabled; inspect autorun key/value, data, process, user, and host.",
        "Payload execution — Security 4688 and Sysmon 1/11; correlate newly written startup payloads with logon-time process creation and parent process.",
    ),
    "T1560": (
        "Archive creation — Security 4688 and Sysmon 1/11; search for 7z, rar, tar, zip, PowerShell compression, unusual command lines, archive paths, size, user, and staging directory.",
        "EDR/DLP/file telemetry — correlate high-volume reads followed by one large archive and subsequent network or removable-media activity.",
    ),
    "T1562.001": (
        "Security-control changes — Microsoft-Windows-Windows Defender/Operational Event IDs 5001 and 5007, Security 4719, and System 7036/7040; inspect actor, setting, service, and time.",
        "Execution context — Security 4688 and Sysmon 1; search for Defender preference changes, service-control commands, firewall/audit-policy changes, EDR tampering, and exclusions.",
    ),
}


@dataclass(frozen=True)
class QuestionCandidate:
    technique_id: str
    technique_name: str
    information_gain: float
    probability_yes: float
    default_question: str
    profile_support: float = 0.0
    supporting_group_count: int = 0
    candidate_group_count: int = 0
    investigation_priority: float = 0.0
    hypothesis: str = ""
    investigation_steps: tuple[str, ...] = ()


@dataclass(frozen=True)
class SelectedQuestion:
    technique_id: str
    question: str
    information_gain: float
    probability_yes: float
    profile_support: float = 0.0
    supporting_group_count: int = 0
    candidate_group_count: int = 0
    investigation_priority: float = 0.0
    hypothesis: str = ""
    investigation_steps: tuple[str, ...] = ()
    # software questions (software_questions.py): the malware asked about, the top-three place of
    # the one candidate documented for it, the predicted separation change if found, and when the
    # question was selected (UTC); technique_id is "" for them
    kind: str = "technique"
    software_id: str | None = None
    target_rank: int | None = None
    target_group_id: str | None = None
    delta_found: float | None = None
    asked_at: str | None = None


def entropy(probabilities: dict[str, float]) -> float:
    return -sum(value * log2(value) for value in probabilities.values() if value > 0)


def information_gain(
    posterior: dict[str, float],
    group_techniques: dict[str, set[str] | list[str]],
    technique_id: str,
    alpha: float = 0.6,
    beta: float = 0.05,
) -> tuple[float, float]:
    if not posterior:
        return 0.0, 0.0
    likelihood_yes = {
        group_id: alpha if technique_id in set(group_techniques.get(group_id, ())) else beta
        for group_id in posterior
    }
    probability_yes = sum(posterior[group_id] * likelihood_yes[group_id] for group_id in posterior)
    probability_no = 1.0 - probability_yes
    yes_raw = {group_id: posterior[group_id] * likelihood_yes[group_id] for group_id in posterior}
    no_raw = {
        group_id: posterior[group_id] * (1.0 - likelihood_yes[group_id]) for group_id in posterior
    }
    yes_posterior = _normalize(yes_raw)
    no_posterior = _normalize(no_raw)
    expected = probability_yes * entropy(yes_posterior)
    expected += probability_no * entropy(no_posterior)
    return max(entropy(posterior) - expected, 0.0), probability_yes


def shortlist_questions(
    kb: Any,
    posterior: dict[str, float],
    answered: set[str],
    *,
    alpha: float = 0.6,
    beta: float = 0.05,
    top_n: int = 10,
    observable_techniques: set[str] | None = None,
    platforms: Iterable[str] = (),
) -> list[QuestionCandidate]:
    """Unanswered techniques of the candidate groups, ranked by information gain.

    ``platforms`` is the case's Target OS: techniques documented only for other operating
    systems are not asked about (their absence says nothing about a group when the hosts
    cannot show them, and a "no" would count against the groups that document them), and the
    investigation steps name only the platforms still relevant.
    """
    group_techniques = {
        group_id: set(values) for group_id, values in getattr(kb, "group_techniques", {}).items()
    }
    technique_ids = set().union(*group_techniques.values()) if group_techniques else set()
    technique_ids -= answered
    if observable_techniques is not None:
        technique_ids &= observable_techniques
    target = tuple(platforms)
    scope = PlatformScope(target, "case") if target else None
    if scope is not None:
        techniques = getattr(kb, "techniques", {}) or {}
        technique_ids = {
            technique_id
            for technique_id in technique_ids
            if scope.allows(_record_values(techniques.get(technique_id), "platforms"))
        }
    candidates = []
    hypothesis_weights = _normalize(posterior)
    if not hypothesis_weights:
        return candidates
    candidate_count = len(hypothesis_weights)
    for technique_id in technique_ids:
        gain, p_yes = information_gain(
            hypothesis_weights, group_techniques, technique_id, alpha, beta
        )
        supporting_groups = {
            group_id
            for group_id in hypothesis_weights
            if technique_id in group_techniques.get(group_id, set())
        }
        if not supporting_groups:
            continue
        profile_support = sum(hypothesis_weights[group_id] for group_id in supporting_groups)
        name = _technique_name(kb, technique_id)
        priority = gain + 0.25 * p_yes
        candidates.append(
            QuestionCandidate(
                technique_id=technique_id,
                technique_name=name,
                information_gain=gain,
                probability_yes=p_yes,
                default_question=(
                    f"After reviewing the suggested telemetry, did you find evidence consistent "
                    f"with {name} ({technique_id})?"
                ),
                profile_support=profile_support,
                supporting_group_count=len(supporting_groups),
                candidate_group_count=candidate_count,
                investigation_priority=priority,
                hypothesis=(
                    f"Unconfirmed investigation hypothesis: {name} ({technique_id}) may be present "
                    f"but not yet visible. It is documented in {len(supporting_groups)}/"
                    f"{candidate_count} current candidate profiles representing "
                    f"{profile_support:.0%} of current hypothesis weight."
                ),
                investigation_steps=_investigation_steps(kb, technique_id, scope),
            )
        )
    return sorted(
        candidates,
        key=lambda item: (-item.investigation_priority, -item.information_gain, item.technique_id),
    )[:top_n]


def select_question_with_llm(
    client: OpenRouterClient,
    candidates: list[QuestionCandidate],
    case_summary: str,
    *,
    model: str,
    effort: str,
    target_platforms: Iterable[str] = (),
) -> SelectedQuestion | None:
    if not candidates:
        return None
    target = ", ".join(target_platforms)
    # only with a Target OS, so every other case keeps the same prompt
    platform_note = (
        f"The affected hosts run {target}: phrase the question for those operating systems.\n\n"
        if target
        else ""
    )
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "technique_id": {"type": "string"},
            "question": {"type": "string"},
        },
        "required": ["technique_id", "question"],
    }
    prompt = (
        "Choose exactly one unconfirmed investigation lead that is concrete, observable, and useful "
        "to a security analyst. Do not introduce a technique outside the list. Phrase one neutral "
        "yes/no question that asks what the analyst found after checking the supplied investigation "
        "steps. Never assert that the behavior occurred and avoid naming threat groups. Prefer a "
        "lead that balances investigation_priority, information_gain, and practical observability.\n\n"
        f"{platform_note}"
        f"Case observations:\n{case_summary[:12000]}\n\n"
        f"Candidates:\n{[asdict(candidate) for candidate in candidates]}"
    )
    response = client.complete(
        [
            {
                "role": "system",
                "content": "You select neutral incident-response follow-up questions.",
            },
            {"role": "user", "content": prompt},
        ],
        model=model,
        effort=effort,
        schema=schema,
        schema_name="follow_up_question",
        max_tokens=1000,
        seed=41 if supports_seed(model) else None,
    )
    parsed = response.parsed if isinstance(response.parsed, dict) else {}
    selected_id = str(parsed.get("technique_id", ""))
    selected = next(
        (item for item in candidates if item.technique_id == selected_id), candidates[0]
    )
    question = str(parsed.get("question") or selected.default_question).strip()
    return SelectedQuestion(
        technique_id=selected.technique_id,
        question=question,
        information_gain=selected.information_gain,
        probability_yes=selected.probability_yes,
        profile_support=selected.profile_support,
        supporting_group_count=selected.supporting_group_count,
        candidate_group_count=selected.candidate_group_count,
        investigation_priority=selected.investigation_priority,
        hypothesis=selected.hypothesis,
        investigation_steps=selected.investigation_steps,
    )


def select_question_locally(candidates: list[QuestionCandidate]) -> SelectedQuestion | None:
    """Deterministic selection for local mode (no LLM): the top-priority lead, default phrasing.

    The shortlist is already ordered by investigation priority, then information gain; the
    LLM only rephrases and may pick a more observable lead. Without it the first candidate
    and its neutral default question are used, so the interview keeps working offline.
    """
    if not candidates:
        return None
    selected = candidates[0]
    return SelectedQuestion(
        technique_id=selected.technique_id,
        question=selected.default_question,
        information_gain=selected.information_gain,
        probability_yes=selected.probability_yes,
        profile_support=selected.profile_support,
        supporting_group_count=selected.supporting_group_count,
        candidate_group_count=selected.candidate_group_count,
        investigation_priority=selected.investigation_priority,
        hypothesis=selected.hypothesis,
        investigation_steps=selected.investigation_steps,
    )


def parse_interview_answer(value: str) -> EvidenceStatus | None:
    normalized = re.sub(r"\s+", " ", re.sub(r"[^a-z0-9']+", " ", value.casefold())).strip()
    if not normalized:
        return None
    # the software-question answers first, so "not found" can never read as found or as no
    if normalized in {"not found", "nothing found"} or normalized.startswith("not found "):
        return EvidenceStatus.NEGATIVE
    if normalized in {"not checked", "unchecked", "did not check", "didn't check"}:
        return EvidenceStatus.UNKNOWN
    if normalized == "found" or normalized.startswith("found "):
        return EvidenceStatus.POSITIVE
    if normalized.startswith(("not visible", "no visibility", "cannot see", "can't see")):
        return EvidenceStatus.NOT_VISIBLE
    if normalized in {"unknown", "unsure", "i don't know", "dont know", "not sure"}:
        return EvidenceStatus.UNKNOWN
    if normalized in {
        "yes",
        "y",
        "observed",
        "present",
        "confirmed",
        "true",
    } or normalized.startswith(("yes ", "observed ", "confirmed ")):
        return EvidenceStatus.POSITIVE
    if normalized in {"no", "n", "not observed", "absent", "false"} or normalized.startswith(
        ("no ", "not observed ", "absent ")
    ):
        return EvidenceStatus.NEGATIVE
    return None


def interview_stop_reason(
    posterior: dict[str, float],
    candidates: list[QuestionCandidate],
    question_count: int,
    *,
    max_questions: int = 10,
    posterior_threshold: float = 0.6,
    minimum_gain: float = 0.02,
    minimum_priority: float = 0.08,
) -> str | None:
    if question_count >= max_questions:
        return "question_limit"
    if not candidates:
        return "no_candidates"
    if question_count == 0:
        return None
    leading = candidates[0]
    if (
        max(posterior.values(), default=0.0) >= posterior_threshold
        and leading.information_gain < minimum_gain
        and leading.probability_yes < 0.2
    ):
        return "concentrated_low_value"
    if leading.investigation_priority < minimum_priority:
        return "low_priority"
    return None


def should_stop(
    posterior: dict[str, float],
    candidates: list[QuestionCandidate],
    question_count: int,
    *,
    max_questions: int = 10,
    posterior_threshold: float = 0.6,
    minimum_gain: float = 0.02,
    minimum_priority: float = 0.08,
) -> bool:
    return (
        interview_stop_reason(
            posterior,
            candidates,
            question_count,
            max_questions=max_questions,
            posterior_threshold=posterior_threshold,
            minimum_gain=minimum_gain,
            minimum_priority=minimum_priority,
        )
        is not None
    )


# How ATT&CK platform labels read in an investigation step; the host operating systems are
# grouped into one phrase ("Windows and Linux hosts").
_HOST_LABELS = ("Windows", "Linux", "macOS")
_PLATFORM_PHRASES = {
    "ESXi": "ESXi hypervisors",
    "Containers": "containers",
    "Network Devices": "network devices",
    "IaaS": "cloud infrastructure (IaaS)",
    "SaaS": "SaaS applications",
    "Office Suite": "office suites (Microsoft 365, Google Workspace)",
    "Identity Provider": "identity providers",
}
# "PRE" marks reconnaissance and resource development (always alone in ATT&CK 19): it happens
# before access, outside the victim's systems, so there is nothing to scope on hosts.
_PRE_COMPROMISE_STEP = (
    "Pre-compromise activity outside the environment: look for it in threat intelligence and "
    "external records (domain, certificate and hosting registrations, scans or probes seen at "
    "the perimeter) and in the tooling recovered from hosts, not in host logs alone."
)


def _join(items: list[str]) -> str:
    return items[0] if len(items) == 1 else f"{', '.join(items[:-1])} and {items[-1]}"


def _scope_step(platforms: tuple[str, ...]) -> str | None:
    """Where to look, in words: the technique's ATT&CK platforms as places to search."""
    if not platforms:
        return None
    if set(platforms) == {"PRE"}:
        return _PRE_COMPROMISE_STEP
    parts = []
    hosts = [label for label in _HOST_LABELS if label in platforms]
    if hosts:
        parts.append(f"{_join(hosts)} hosts")
    parts += [phrase for label, phrase in _PLATFORM_PHRASES.items() if label in platforms]
    known = {*_HOST_LABELS, *_PLATFORM_PHRASES, "PRE"}
    parts += [label for label in platforms if label not in known]  # a newer ATT&CK label, as is
    return f"Scope the search to {_join(parts)}." if parts else None


def _investigation_steps(
    kb: Any, technique_id: str, scope: PlatformScope | None = None
) -> tuple[str, ...]:
    technique = getattr(kb, "techniques", {}).get(technique_id)
    platforms = _record_values(technique, "platforms")
    if scope is not None:
        platforms = scope.relevant(platforms)
    tactics = _record_values(technique, "tactics")
    detection = str(_record_value(technique, "detection") or "").strip()
    parent_id = technique_id.split(".", 1)[0]
    # Prefer ATT&CK Analytics-derived collection steps when available, falling
    # back to hand-authored guidance. Analytics provide concrete log sources
    # and channels (e.g., Sysmon Event ID 10) tied to the current ATT&CK version.
    steps: list[str] = []
    analytic_steps = _analytic_collection_steps(kb, technique_id, scope)
    if analytic_steps:
        steps.extend(analytic_steps)
    elif scope is None or "Windows" in scope.platforms:
        # the hand-written fallback names Windows telemetry (Sysmon, event IDs, registry)
        collection_steps = _TECHNIQUE_COLLECTION_GUIDANCE.get(
            technique_id, _TECHNIQUE_COLLECTION_GUIDANCE.get(parent_id, ())
        )
        steps.extend(collection_steps)
    scope_step = _scope_step(platforms)
    if scope_step:
        steps.append(scope_step)
    steps.extend(
        _TACTIC_GUIDANCE[tactic.casefold()]
        for tactic in tactics
        if tactic.casefold() in _TACTIC_GUIDANCE
    )
    if detection:
        concise = re.split(r"(?<=[.!?])\s+", detection, maxsplit=1)[0].strip()
        steps.append(f"ATT&CK detection guidance: {concise[:600]}")
    if not steps:
        steps.append(
            "Review endpoint, identity, network, and application telemetry around the incident "
            "window for artifacts consistent with this behavior."
        )
    return tuple(dict.fromkeys(steps))[:8]


def _analytic_collection_steps(
    kb: Any, technique_id: str, scope: PlatformScope | None = None
) -> list[str]:
    """Build collection steps from ATT&CK Analytics when available.

    Returns an empty list if the KB has no parsed detection strategies or
    analytics for the technique, so the caller can fall back to hand-authored
    guidance. Each analytic covers one platform; with a Target OS only the
    analytics for it (and for platforms an OS says nothing about) are kept, so a
    Linux case is not sent to Windows event logs.
    """
    detection_strategies = getattr(kb, "detection_strategies", {})
    analytics = getattr(kb, "analytics", {})
    if not detection_strategies or not analytics:
        return []
    try:
        from apt_classifier.telemetry import technique_collection_steps
    except ImportError:
        return []
    steps = technique_collection_steps(kb, technique_id)
    if scope is not None:
        steps = [step for step in steps if scope.allows(step.platforms)]
    if not steps:
        return []
    result: list[str] = []
    for step in steps:
        sources = ", ".join(source.label for source in step.log_sources if source.name)
        prefix = f"[{step.analytic_id}] "
        if sources:
            prefix += f"Sources: {sources}. "
        description = step.description.strip()
        if description:
            result.append(f"{prefix}{description[:500]}")
    return result


def _record_value(record: Any, name: str, default: Any = "") -> Any:
    if record is None:
        return default
    if isinstance(record, dict):
        return record.get(name, default)
    return getattr(record, name, default)


def _record_values(record: Any, name: str) -> tuple[str, ...]:
    value = _record_value(record, name, ())
    values = [value] if isinstance(value, str) else list(value or ())
    return tuple(dict.fromkeys(str(item).strip() for item in values if str(item).strip()))


def _normalize(values: dict[str, float]) -> dict[str, float]:
    total = sum(values.values())
    if total <= 0:
        count = len(values)
        return {key: 1.0 / count for key in values} if count else {}
    return {key: value / total for key, value in values.items()}


def _technique_name(kb: Any, technique_id: str) -> str:
    technique = getattr(kb, "techniques", {}).get(technique_id)
    if technique is None:
        return technique_id
    if isinstance(technique, dict):
        return str(technique.get("name") or technique_id)
    return str(getattr(technique, "name", technique_id))
