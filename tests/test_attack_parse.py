from apt_classifier.attack.download import attack_bundle_url, normalize_attack_version
from apt_classifier.attack.kb import AttackKnowledgeBase
from apt_classifier.attack.parse import parse_attack_bundle, split_sentences, strip_attack_citations


def attack_object(kind, stix_id, external_id, name, **extra):
    return {
        "type": kind,
        "id": stix_id,
        "name": name,
        "description": extra.pop("description", f"{name} description."),
        "external_references": [{"source_name": "mitre-attack", "external_id": external_id}],
        **extra,
    }


def relationship(source, target, relationship_type="uses", description=""):
    return {
        "type": "relationship",
        "id": f"relationship--{source[-4:]}-{target[-4:]}-{relationship_type}",
        "source_ref": source,
        "target_ref": target,
        "relationship_type": relationship_type,
        "description": description,
    }


def test_parse_resolves_entities_relationships_and_filters_inactive(tmp_path):
    group_stix = "intrusion-set--0001"
    technique_stix = "attack-pattern--0001"
    sub_stix = "attack-pattern--0002"
    software_stix = "malware--0001"
    campaign_stix = "campaign--0001"
    bundle = {
        "type": "bundle",
        "objects": [
            attack_object("intrusion-set", group_stix, "G0001", "Example Group", aliases=["EG"]),
            attack_object(
                "attack-pattern",
                technique_stix,
                "T1000",
                "Example Technique",
                x_mitre_platforms=["Windows"],
                kill_chain_phases=[{"phase_name": "execution"}],
                x_mitre_detection="Detect this. (Citation: Test)",
            ),
            attack_object(
                "attack-pattern",
                sub_stix,
                "T1000.001",
                "Example Sub-technique",
                x_mitre_is_subtechnique=True,
            ),
            attack_object("malware", software_stix, "S0001", "Example Tool"),
            attack_object("campaign", campaign_stix, "C0001", "Example Campaign"),
            attack_object(
                "attack-pattern",
                "attack-pattern--dead",
                "T9999",
                "Deprecated",
                x_mitre_deprecated=True,
            ),
            relationship(
                group_stix,
                technique_stix,
                description=(
                    "Example Group used a signed binary to load its payload into a specific process. "
                    "It then removed the temporary payload. (Citation: Example Report)"
                ),
            ),
            relationship(group_stix, software_stix),
            relationship(software_stix, sub_stix),
            relationship(campaign_stix, technique_stix),
            relationship(campaign_stix, group_stix, "attributed-to"),
            relationship(sub_stix, technique_stix, "subtechnique-of"),
        ],
    }
    kb = parse_attack_bundle(bundle, version="test")
    assert "T9999" not in kb.techniques
    assert kb.group_techniques["G0001"] == ["T1000"]
    assert kb.groups["G0001"].technique_behaviors == {
        "T1000": [
            "Example Group used a signed binary to load its payload into a specific process.",
            "It then removed the temporary payload.",
        ]
    }
    assert kb.group_software["G0001"] == ["S0001"]
    assert kb.software_techniques["S0001"] == ["T1000.001"]
    assert kb.campaign_groups["C0001"] == ["G0001"]
    assert kb.techniques["T1000.001"].parent_id == "T1000"
    path = kb.save(tmp_path / "kb.json")
    loaded = AttackKnowledgeBase.load(path)
    assert loaded.to_dict() == kb.to_dict()


def test_text_cleanup_and_versioned_url():
    assert strip_attack_citations("Action. (Citation: Report)") == "Action."
    assert split_sentences("First action. Second action!") == ["First action.", "Second action!"]
    assert normalize_attack_version("v18.1") == "18.1"
    assert attack_bundle_url("18.1").endswith("enterprise-attack-18.1.json")
