import json, urllib.request, time

DOIS = ["10.1137/24M1681458", "10.1137/24M1702246", "10.1137/18M1226312",
        "10.1137/20M1314355", "10.1137/21M1465032", "10.1093/imanum/drac012",
        "10.1137/24M1679586", "10.1145/3840284", "10.1007/s00211-023-01370-y",
        "10.1109/78.492531", "10.1109/TVLSI.2019.2913958", "10.1109/18.256489",
        "10.1109/TCOM.1964.1088973", "10.1137/1.9780898718027", "10.1109/DAC63849.2025.11132978",
        "10.1609/aaai.v32i1.11334", "10.1109/FPL.2018.00056"]

def get(u):
    r = urllib.request.Request(u, headers={"User-Agent": "audit/1.0 (mailto:a@b.org)"})
    return json.loads(urllib.request.urlopen(r, timeout=20).read())

print(f"{'DOI':<32} {'YEAR':<6} {'VOL/ISS':<10} {'PAGES':<14} UPDATED-BY / UPDATE-TO")
print("-" * 120)
for d in DOIS:
    try:
        m = get("https://api.crossref.org/works/" + d)["message"]
        yr = m.get("issued", {}).get("date-parts", [["?"]])[0][0]
        vi = f"{m.get('volume','-')}/{m.get('issue','-')}"
        pp = m.get("page") or m.get("article-number") or "-"
        ut = ",".join(x.get("DOI", "") for x in m.get("update-to", [])) or "-"
        ub = ",".join(x.get("DOI", "") for x in m.get("updated-by", [])) or "-"
        rel = [f"{x.get('type')}:{x.get('id','')}" for x in m.get("relation", {}).get("is-retracted-by", [])]
        print(f"{d:<32} {yr:<6} {vi:<10} {pp:<14} to={ut} | by={ub} | retracted-by={rel or '-'}")
    except Exception as e:
        print(f"{d:<32} ERROR {e}")
    time.sleep(0.3)
