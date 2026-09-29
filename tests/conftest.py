from types import SimpleNamespace

import pytest


@pytest.fixture
def synthetic_kb():
    groups = {
        "G1": SimpleNamespace(
            id="G1",
            name="Alpha",
            aliases=("A-Team",),
            description=(
                "Alpha is a China-nexus espionage group that targets United States government "
                "and defense organizations to steal intellectual property."
            ),
            campaign_ids=("C1",),
        ),
        "G2": SimpleNamespace(
            id="G2",
            name="Bravo",
            aliases=("B-Team",),
            description=(
                "Bravo is a financially motivated group targeting European financial institutions "
                "for ransomware extortion."
            ),
            campaign_ids=("C2",),
        ),
        "G3": SimpleNamespace(
            id="G3",
            name="Charlie",
            aliases=("C-Team",),
            description=(
                "Charlie conducts disruptive attacks against Ukrainian energy organizations "
                "to destroy operational systems."
            ),
            campaign_ids=(),
        ),
    }
    techniques = {
        technique_id: SimpleNamespace(
            id=technique_id, name=name, tactics=("execution",), platforms=("Windows",)
        )
        for technique_id, name in {
            "T1": "One",
            "T2": "Two",
            "T3": "Three",
            "T4": "Four",
        }.items()
    }
    software = {
        "S1": SimpleNamespace(id="S1", name="Tool One", aliases=("RedKit",)),
        "S2": SimpleNamespace(id="S2", name="Tool Two", aliases=()),
    }
    campaigns = {
        "C1": SimpleNamespace(id="C1", name="Operation One", aliases=()),
        "C2": SimpleNamespace(id="C2", name="Operation Two", aliases=()),
    }
    return SimpleNamespace(
        groups=groups,
        techniques=techniques,
        software=software,
        campaigns=campaigns,
        group_techniques={"G1": {"T1", "T2"}, "G2": {"T2", "T3"}, "G3": {"T3", "T4"}},
        group_software={"G1": {"S1"}, "G2": {"S2"}, "G3": set()},
        software_groups={"S1": {"G1"}, "S2": {"G2"}},
        campaign_techniques={"C1": {"T1", "T2"}, "C2": {"T2", "T3"}},
        campaign_groups={"C1": {"G1"}, "C2": {"G2"}},
        idf={"T1": 2.0, "T2": 1.0, "T3": 1.5, "T4": 2.0},
        group_priors={"G1": 1 / 3, "G2": 1 / 3, "G3": 1 / 3},
        prominence={"G1": 10.0, "G2": 3.0, "G3": 1.0},
    )
