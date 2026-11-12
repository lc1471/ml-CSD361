import os, sys, shutil, getpass, time, zipfile, requests

SERVER_URL = "https://borealisdata.ca"
PERSISTENT_ID = "doi:10.5683/SP3/PSWY62"
OUT_DIR = r"C:\dev\data\avcaffe"
INSTITUTION = "Shiv Nadar Institution of Eminence"
RETRIES = 3

def fetch(fid, headers, tmp):
    body = {"guestbookResponse": {"institution": INSTITUTION, "answers": []}}
    r = requests.post(f"{SERVER_URL}/api/access/datafile/{fid}", headers=headers, json=body, timeout=60)
    if r.status_code != 200:
        raise RuntimeError(f"link HTTP {r.status_code}: {r.text[:200]}")
    signed = (r.json().get("data") or {}).get("signedUrl")
    if not signed:
        raise RuntimeError("no signedUrl")
    with requests.get(signed, stream=True, timeout=120) as g:
        if g.status_code != 200:
            raise RuntimeError(f"download HTTP {g.status_code}: {g.text[:200]}")
        expected = int(g.headers.get("Content-Length", 0))
        enc = g.headers.get("Content-Encoding")
        got = 0
        with open(tmp, "wb") as f:
            for chunk in g.iter_content(1024 * 1024):
                f.write(chunk)
                got += len(chunk)
    if expected and not enc and got != expected:
        raise RuntimeError(f"incomplete: got {got} of {expected} bytes")

def is_done(path):
    if path.endswith(".zip"):
        final = path[:-4]
        return os.path.isdir(final) and any(os.scandir(final))
    return os.path.exists(path)

def main():
    folders = set(sys.argv[1:])
    all_files = not folders
    headers = {"X-Dataverse-key": getpass.getpass("token: ")}
    url = f"{SERVER_URL}/api/datasets/:persistentId//versions/:latest/files?persistentId={PERSISTENT_ID}"
    resp = requests.get(url, headers=headers)
    if resp.status_code != 200:
        print("Cannot get the dataset:", resp.status_code, resp.text[:200])
        return
    ok = skipped = 0
    failed = []
    for item in resp.json()["data"]:
        fid = item["dataFile"]["id"]
        fname = item["dataFile"]["filename"]
        label = item.get("directoryLabel")
        if label is not None:
            wanted = all_files or any(label == f or label.startswith(f + "/") for f in folders)
            dr = os.path.join(OUT_DIR, *label.split("/"))
        else:
            wanted = all_files or "." in folders
            dr = OUT_DIR
        if not wanted:
            continue
        os.makedirs(dr, exist_ok=True)
        path = os.path.join(dr, fname)
        final = path[:-4] if path.endswith(".zip") else path
        rel = os.path.relpath(final, OUT_DIR)
        if is_done(path):
            skipped += 1
            continue
        tmp = path + ".part"
        for attempt in range(1, RETRIES + 1):
            try:
                fetch(fid, headers, tmp)
                if path.endswith(".zip"):
                    if not zipfile.is_zipfile(tmp):
                        with open(tmp, "rb") as f:
                            head = f.read(120)
                        raise RuntimeError(f"not a zip, starts with {head!r}")
                    os.makedirs(final, exist_ok=True)
                    with zipfile.ZipFile(tmp) as z:
                        z.extractall(final)
                    os.remove(tmp)
                else:
                    os.replace(tmp, path)
                ok += 1
                print(rel)
                break
            except Exception as e:
                print(f"{rel} attempt {attempt} failed: {e}")
                if os.path.exists(tmp):
                    os.remove(tmp)
                if path.endswith(".zip") and os.path.isdir(final):
                    shutil.rmtree(final, ignore_errors=True)
                time.sleep(5 * attempt)
        else:
            failed.append(rel)
    print(f"done: {ok} downloaded, {skipped} skipped, {len(failed)} failed")
    for x in failed:
        print("FAILED:", x)

if __name__ == "__main__":
    main()
