"""Gold passages for E1 (pilot 3 design): for each rule family, three independent source wordings of the same rule.
Verbatim spans are cut from the source files by start/end markers (reproducible); the two FDIC RMS tables are
transcribed from the layout rendering of PDF pages 11 and 12 (values unchanged; table cells joined into sentences).
Output: gold_passages.json."""
import json, os, re, subprocess
import e1_lib as L

flat = lambda s: re.sub(r"\s+", " ", s).strip()


def span(text, start, end, include_end=False):
    i = text.index(start)
    j = text.index(end, i)
    return flat(text[i:j + (len(end) if include_end else 0)])


ecfr_pca = flat(L._text(os.path.join(L.RAW, "ecfr_324H_PCA.xml")))
ecfr_252 = flat(L._text(os.path.join(L.RAW, "ecfr_252_5_categories.xml")))
fr = flat(L._text(os.path.join(L.RAW, "fr_84FR59230.txt")))
cbem = flat(subprocess.run(["pdftotext", "-f", "784", "-l", "785", os.path.join(L.RAW, "fed_cbem_wb20230120.pdf"), "-"],
                           capture_output=True, text=True).stdout)

# Fed CBEM PCA section: two-column extraction interleaves footnotes; stitch the five category descriptions in order.
cb1 = span(cbem, "The following are descriptions of the five PCA capital categories", "PCA direc", include_end=True)
cb1 = re.sub(r"or greater,4 and", "or greater, and", cb1)
cb1b = span(cbem, "tive to meet and maintain a specific capital level", "A qualifying community banking organization")
cb2 = span(cbem, "2. Adequately capitalized.", "4. Significantly undercapitalized.") + " 4. Significantly undercapitalized. " \
      "The bank has a total risk-based capital ratio that is less than"
cb2 = re.sub(r"bank\.6 ", "bank. ", cb2); cb2 = re.sub(r"growth\.7 ", "growth. ", cb2)
cb3 = span(cbem, "6.0 percent, a tier 1 risk-based capital ratio that is less than 4.0", "EXAMINATION CONSIDERATIONS")
cb3 = re.sub(r"2\.0 percent\.8", "2.0 percent.", cb3)
pca_cbem = flat(f"{cb1}{cb1b} {cb2} {cb3}")  # "PCA direc" + "tive to meet": the word is split across the page column

GOLD = {
    "pca": {
        "ecfr": dict(cite="12 CFR 324.403(b)(1)-(2), eCFR as of 31 Dec 2022 (verbatim)",
                     text=span(ecfr_pca, "(b) Capital categories.", "(3) “Undercapitalized”")),
        "fdic_rms": dict(cite="FDIC RMS Manual 2.1 (8/22), p. 2.1-11 table (transcribed; values verbatim)",
                         text="The following table summarizes the PCA categories for non-CBLR institutions. "
                              "Well Capitalized: Total RBC Ratio ≥ 10%; Tier 1 RBC Ratio ≥ 8%; Common Equity Tier 1 RBC Ratio ≥ 6.5%; Tier 1 Leverage Ratio ≥ 5%. "
                              "Adequately Capitalized: Total RBC Ratio ≥ 8%; Tier 1 RBC Ratio ≥ 6%; Common Equity Tier 1 RBC Ratio ≥ 4.5%; Tier 1 Leverage Ratio ≥ 4%. "
                              "Undercapitalized: Total RBC Ratio < 8%; Tier 1 RBC Ratio < 6%; Common Equity Tier 1 RBC Ratio <4.5%; Tier 1 Leverage Ratio < 4%. "
                              "Significantly Undercapitalized: Total RBC Ratio < 6%; Tier 1 RBC Ratio < 4%; Common Equity Tier 1 RBC Ratio < 3%; Tier 1 Leverage Ratio < 3%. "
                              "Critically Undercapitalized: Tangible Equity/Total Assets ≤ 2%."),
        "fed_cbem": dict(cite="Fed Commercial Bank Examination Manual, Prompt Corrective Action (Nov 2020), pp. 2-3 (verbatim, footnote markers removed)",
                         text=pca_cbem),
    },
    "tailoring": {
        "fr": dict(cite="84 FR 59230 (1 Nov 2019), preamble section on the four categories (verbatim, footnote markers removed)",
                   text=re.sub(r"\\2[45]\\", "", " ".join([
                       span(fr, "The second set of standards", "Like Category I standards"),
                       span(fr, "The third set of standards", "Category III standards reflect"),
                       span(fr, "The fourth set of standards", "Banking organizations in Category IV")]))),
        "ecfr": dict(cite="12 CFR 252.5(d)(1) and (e)(1), eCFR as of 31 Dec 2022 (verbatim)",
                     text=span(ecfr_252, "(d) Category III. (1)", "(2) After meeting the criteria in paragraph (d)(1)") + " " +
                          span(ecfr_252, "(e) Category IV. (1)", "(2) After meeting the criteria in paragraph (e)(1)")),
        "fdic_rms": dict(cite="FDIC RMS Manual 2.1 (8/22), p. 2.1-12 table (transcribed; values verbatim)",
                         text="The table below summarizes the additional capital requirements for Category I – IV institutions. "
                              "Category I: U.S. Global Systemically Important Banks (GSIBs). "
                              "Category II: Banking organizations with $700 billion or more in total assets or $75 billion or more in cross-jurisdictional activity that are not GSIBs. "
                              "Category III: Banking organizations that are not subject to Category I or Category II thresholds and that have either: "
                              "$250 billion or more in total assets; or $100 billion but less than $250 billion in total assets and $75 billion or more of any "
                              "of the following nonbank assets, weighted short-term wholesale funding (STWF), or off-balance-sheet exposures. "
                              "Category IV: Banking organizations that are U.S. depository institution holding companies or U.S. intermediate holding "
                              "companies with at least $100 billion in total assets that do not meet any of the thresholds specified for Categories I-III."),
    },
}
# Which rule family each item needs.
FAMILY = {"R1": "pca", "R2": "pca", "R3": "pca", "C1": "pca", "R4": "tailoring", "R5": "tailoring", "C2": "tailoring"}
FAMILY.update({f"R{k}": "pca" for k in range(6, 17)})        # R6-R16 (added 24 Oct 2026)
FAMILY.update({f"R{k}": "tailoring" for k in range(17, 26)})  # R17-R25

if __name__ == "__main__":
    for fam, d in GOLD.items():
        for src, g in d.items():
            print(f"== {fam}/{src} ({len(g['text'].split())} words) {g['cite']}\n{g['text']}\n")
    json.dump(dict(gold=GOLD, family=FAMILY), open(os.path.join(L.HERE, "gold_passages.json"), "w"), indent=1, ensure_ascii=False)
