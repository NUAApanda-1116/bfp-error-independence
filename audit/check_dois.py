import json, urllib.request, urllib.parse, time

DOIS = [
 ("abdelfattah2025", None, "arXiv:2506.11277"),
 ("carson2024", "10.1137/24M1702246", None),
 ("carson2025", "10.1145/3840284", None),
 ("croci2022", "10.1093/imanum/drac012", None),
 ("elarar2024", "10.1137/24M1681458", None),
 ("fan2018", "10.1109/FPL.2018.00056", None),
 ("fasi2023", "10.1137/21M1465032", None),
 ("gray1993dithered", "10.1109/18.256489", None),
 ("hallman2023", "10.1007/s00211-023-01370-y", None),
 ("han2025", "10.1109/DAC63849.2025.11132978", None),
 ("higham2002", "10.1137/1.9780898718027", None),
 ("higham2019", "10.1137/18M1226312", None),
 ("higham2022", "10.1137/20M1314355", None),
 ("kalliojarvi1996", "10.1109/78.492531", None),
 ("lian2019", "10.1109/TVLSI.2019.2913958", None),
 ("schuchman1964", "10.1109/TCOM.1964.1088973", None),
 ("song2018", "10.1609/aaai.v32i1.11334", None),
 ("fox2024zfp", "10.1137/24M1703513", None),
]

ARXIV = ["2506.11277","2603.19559","2404.12556","2411.18747","2410.06319","2504.07835",
         "2605.09825","2010.16225","2410.10517","2408.03069","2603.06060","2407.01826",
         "2203.15928","2504.15721","2607.18758","2605.20402","2604.10494","2606.20381"]

def get(url, timeout=15):
    req = urllib.request.Request(url, headers={"User-Agent":"ref-audit/1.0 (mailto:audit@example.org)"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", "replace")

CACHE = r"E:\paper\_audit\doi_arxiv.json"
try:
    with open(CACHE, encoding="utf-8") as f:
        out = json.load(f)
except Exception:
    out = {}

def save():
    with open(CACHE, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

for key, doi, arx in DOIS:
    if key in out:
        continue
    if doi:
        url = "https://api.crossref.org/works/" + urllib.parse.quote(doi)
        try:
            j = json.loads(get(url))
            m = j["message"]
            rec = {
                "found": True,
                "title": (m.get("title") or [""])[0],
                "container": (m.get("container-title") or [""])[0],
                "short": (m.get("short-container-title") or [""])[0],
                "volume": m.get("volume"), "issue": m.get("issue"),
                "page": m.get("page"), "article-number": m.get("article-number"),
                "issued": m.get("issued", {}).get("date-parts"),
                "published-print": m.get("published-print", {}).get("date-parts"),
                "published-online": m.get("published-online", {}).get("date-parts"),
                "type": m.get("type"),
                "publisher": m.get("publisher"),
                "authors": [" ".join(filter(None,[a.get("given"), a.get("family")])) for a in m.get("author",[])],
                "DOI": m.get("DOI"),
            }
        except Exception as e:
            rec = {"found": False, "error": repr(e), "doi": doi}
        out[key] = rec
        print("="*90)
        print(key, "|", doi)
        print(json.dumps(rec, ensure_ascii=False, indent=1))
        save()
        time.sleep(0.3)

print("\n\n########## arXiv ##########")
for a in ARXIV:
    if "arXiv:"+a in out:
        continue
    url = "http://export.arxiv.org/api/query?id_list=" + a
    try:
        x = get(url)
    except Exception as e:
        print(a, "ERROR", e); continue
    import re
    def grab(tag):
        m = re.search(r"<%s>(.*?)</%s>" % (tag, tag), x, re.S)
        return re.sub(r"\s+", " ", m.group(1)).strip() if m else None
    title = grab("title")
    # first entry title only
    titles = re.findall(r"<title>(.*?)</title>", x, re.S)
    title = re.sub(r"\s+"," ", titles[1]).strip() if len(titles)>1 else None
    authors = re.findall(r"<name>(.*?)</name>", x, re.S)
    published = re.search(r"<published>(.*?)</published>", x)
    updated = re.search(r"<updated>(.*?)</updated>", x)
    prim = re.search(r'<arxiv:primary_category[^>]*term="([^"]+)"', x)
    comment = re.search(r'<arxiv:comment[^>]*>(.*?)</arxiv:comment>', x, re.S)
    jref = re.search(r'<arxiv:journal_ref[^>]*>(.*?)</arxiv:journal_ref>', x, re.S)
    doi = re.search(r'<arxiv:doi[^>]*>(.*?)</arxiv:doi>', x, re.S)
    rec = {"title": title, "authors": authors,
           "published": published.group(1) if published else None,
           "updated": updated.group(1) if updated else None,
           "primary": prim.group(1) if prim else None,
           "comment": re.sub(r"\s+"," ",comment.group(1)).strip() if comment else None,
           "journal_ref": re.sub(r"\s+"," ",jref.group(1)).strip() if jref else None,
           "doi": doi.group(1) if doi else None}
    out["arXiv:"+a] = rec
    print("="*90)
    print("arXiv:", a)
    print(json.dumps(rec, ensure_ascii=False, indent=1))
    save()
    time.sleep(0.3)

save()
print("ALL DONE")
