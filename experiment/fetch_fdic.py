"""E1 cases: public FDIC call-report aggregates at 2022-12-31 for banks with assets $10-300bn.
Output: cases_20221231.csv (all banks in range) -- source https://api.fdic.gov/banks/financials
Field titles checked against https://api.fdic.gov/banks/docs/risview_properties.yaml (23 Oct 2026):
  RBC1AAJ leverage ratio-PCA; RBC1RWAJ tier 1 RBC ratio-PCA; RBCRWAJ total RBC ratio-PCA;
  IDT1CER common equity tier 1 ratio; DEPUNINS estimated uninsured deposits, domestic offices;
  DEPDOM deposits in domestic offices; SCHA/SCHF HTM amortised cost/fair value; SCAA/SCAF AFS amortised cost/fair value.
Dollar amounts are in $ thousands in the API; stored here in $ bn."""
import json, csv, urllib.request, urllib.parse, os
HERE = os.path.dirname(os.path.abspath(__file__))
F = "CERT,REPDTE,NAMEFULL,ASSET,DEP,DEPDOM,DEPUNINS,SCHA,SCHF,SCAA,SCAF,EQ,RBC1AAJ,RBC1RWAJ,IDT1CER,RBCRWAJ,ROA"
q = {"filters": "REPDTE:20221231 AND ASSET:[10000000 TO 300000000]", "fields": F, "limit": "500", "sort_by": "ASSET", "sort_order": "DESC"}
url = "https://api.fdic.gov/banks/financials?" + urllib.parse.urlencode(q)
req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
d = json.load(urllib.request.urlopen(req, timeout=120))
fails = json.load(urllib.request.urlopen(urllib.request.Request(
    "https://api.fdic.gov/banks/failures?" + urllib.parse.urlencode({"filters": 'FAILDATE:["2023-01-01" TO "2025-12-31"]', "fields": "CERT,FAILDATE,NAME", "limit": "100"}),
    headers={"User-Agent": "Mozilla/5.0"}), timeout=120))
failed = {x["data"]["CERT"]: x["data"]["FAILDATE"] for x in fails["data"]}
bn = lambda v: round((v or 0) / 1e6, 2)
rows = []
for x in d["data"]:
    r = x["data"]
    if None in (r.get("DEPDOM"), r.get("DEPUNINS"), r.get("EQ")) or not r["DEPDOM"] or not r["EQ"]:
        continue
    unreal = ((r.get("SCHA") or 0) - (r.get("SCHF") or 0)) + ((r.get("SCAA") or 0) - (r.get("SCAF") or 0))
    rows.append(dict(cert=r["CERT"], name=r["NAMEFULL"], assets_bn=bn(r["ASSET"]),
                     dep_dom_bn=bn(r["DEPDOM"]), dep_unins_bn=bn(r["DEPUNINS"]),
                     equity_bn=bn(r["EQ"]), unreal_loss_bn=bn(unreal),
                     uninsured_share=round(r["DEPUNINS"] / r["DEPDOM"], 3),
                     unreal_loss_to_equity=round(unreal / r["EQ"], 3),
                     leverage=r.get("RBC1AAJ"), tier1_rbc=r.get("RBC1RWAJ"),
                     cet1=r.get("IDT1CER"), total_rbc=r.get("RBCRWAJ"),
                     roa=r.get("ROA"), failed=failed.get(r["CERT"], "")))
with open(os.path.join(HERE, "cases_20221231.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print(len(rows), "banks;", sum(1 for r in rows if r["failed"]), "later failed")
for r in rows:
    if r["failed"]: print(r)
