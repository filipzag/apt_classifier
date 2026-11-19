from apt_classifier.evidence.artefacts import ArtefactIndex, extract_artefacts


def keys(text: str) -> set[str]:
    return {a.key() for a in extract_artefacts(text)}


def test_registry_hives_and_spaces_normalise_alike():
    long_form = keys(
        "modified `HKEY_CURRENT_USER\\Software\\Microsoft\\Windows NT\\CurrentVersion\\Windows\\Load`"
    )
    short_form = keys(
        "sets <code>HKCU\\Software\\Microsoft\\Windows NT\\CurrentVersion\\Windows\\Load</code>"
    )
    assert (
        "registry:hkcu\\software\\microsoft\\windows nt\\currentversion\\windows\\load" in long_form
    )
    assert long_form == short_form


def test_registry_key_stops_before_a_following_path():
    found = keys("reg save hklm\\sam C:\\ProgramData\\sam.save")
    assert "registry:hklm\\sam" in found
    assert "path:%programdata%\\sam.save" in found


def test_user_folders_fold_into_variables():
    a = keys("dropped to C:\\Users\\bob\\AppData\\Roaming\\NvmFilter\\stu.cr")
    b = keys("dropped to %APPDATA%\\NvmFilter\\stu.cr")
    assert "path:%appdata%\\nvmfilter\\stu.cr" in a
    assert a == b


def test_generic_folders_and_system_binaries_are_not_artefacts():
    assert keys("wrote a log to %TEMP% and ran cmd.exe and powershell.exe") == set()
    assert keys("files under /var/log/ were deleted") == set()


def test_named_objects():
    assert "mutex:loadupongunsbringyourfriends" in keys(
        "hardcoded mutex name of “LoadUpOnGunsBringYourFriends” using CreateMutexW()"
    )
    assert "pipe:testpipe" in keys("a named pipe (“\\\\.\\pipe\\testPipe”)")
    assert "service:wmdmpmsp" in keys("it creates a Windows service named WmdmPmSp")
    assert "task:microsoftedgeupdatetaskmachine" in keys(
        'schtasks /create /tn "MicrosoftEdgeUpdateTaskMachine" /tr x'
    )
    assert "export:starts" in keys("a DLL with a single export named Starts.")
    assert keys("cloud services such as Dropbox") == set()


def test_pdb_path_and_leaf():
    found = keys("It contains the PDB path D:\\Proj\\Release\\x64\\Finder.pdb.")
    assert found == {"pdb:d:\\proj\\release\\x64\\finder.pdb", "pdb:finder.pdb"}


def test_user_agent_ends_at_the_product_tokens():
    found = keys(
        "User-Agent: Mozilla/5.0 (Windows NT 6.1; WOW64; rv:52.0) Gecko/20100101 Firefox/52.0 Host: x"
    )
    assert found == {
        "user_agent:Mozilla/5.0 (Windows NT 6.1; WOW64; rv:52.0) Gecko/20100101 Firefox/52.0"
    }


def test_index_weights_rare_artefacts_and_drops_common_ones():
    shared = "the adversary used the mutex “Common1”."
    records = [{"text": shared, "groups": [f"G{i}"]} for i in range(5)]
    records.append({"text": "the adversary created the mutex “Rare42”.", "groups": ["G9"]})
    index = ArtefactIndex(records, [f"G{i}" for i in range(10)], max_groups=3)
    report = extract_artefacts("It creates mutex “Rare42” and mutex “Common1”.")
    assert index.scores(report) == {"G9": index.weight["mutex:rare42"]}


def test_long_separator_runs_stay_fast():
    import time

    text = "see " + "/".join(f"seg{i}" for i in range(3000)) + " and " + "\\x" * 400 + " done"
    start = time.perf_counter()
    extract_artefacts(text)
    assert time.perf_counter() - start < 1.0


def test_prose_is_not_a_file_or_an_export():
    assert keys("results saved in.zip archives and.log files") == set()
    assert keys("the DLL exports the requests module") == set()
    assert keys("redirected output to /dev/null and wrote to C:\\Temp") == set()
