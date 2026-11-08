r"""
build_L0.py -- pipeline smoke test deck (EN + RO): title, one chart frame, abbreviations, references.
Run:  python3 lectures/L0_smoke/build_L0.py   (exit code 0 only if every check of lectures/STYLE.md passes)
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
from nn_build import Deck, Numbers, abbreviation_frames, reference_frames, run_build  # noqa: E402

ABBR = {
    "MLP": ("multilayer perceptron", "perceptron multistrat"),
    "ASE": ("Bucharest University of Economic Studies", "Academia de Studii Economice din București"),
}
IGNORE = set()
REFS = [
    "Bishop, C. M. \\& Bishop, H. (2024). \\textit{Deep Learning: Foundations and Concepts}. Springer. "
    "\\href{https://doi.org/10.1007/978-3-031-45468-4}{doi:10.1007/978-3-031-45468-4}",
]


def build(lang, H=None):
    V = Numbers(os.path.join(HERE, "L0_numbers.json"), lang)
    D = Deck(0, "Pipeline smoke test" if lang == "en" else "Testul de funcționare al fluxului de lucru", V, lang, heights=H)
    t = D.t
    D.parts.append("% END-FRONT-MATTER\n")
    abbreviation_frames(D, ABBR)
    D.section(t("A real chart", "Un grafic pe date reale"))
    D.figframe(
        t("Equity losses come in long episodes", "Pierderile pe acțiuni vin în episoade lungi"),
        t("a multilayer perceptron (MLP) trained on prices must cope with drawdowns like these.",
          "un perceptron multistrat (MLP) antrenat pe prețuri trebuie să facă față unor scăderi ca acestea."),
        "\\[ D_t=\\frac{P_t}{\\max_{s\\le t}P_s}-1 \\]",
        "sp500_drawdown",
        bullets=[t("Maximum drawdown @{sp500.maxdd_pct|.2f}\\,\\% on @{sp500.maxdd_date|date}; annualised volatility "
                   "@{sp500.vol_ann_pct|.2f}\\,\\% over @{data.n_days|,d} daily log returns.",
                   "Scăderea maximă: @{sp500.maxdd_pct|.2f}\\,\\%, la @{sp500.maxdd_date|date}; volatilitatea anualizată: "
                   "@{sp500.vol_ann_pct|.2f}\\,\\%, pe @{data.n_days|,d} de randamente logaritmice zilnice.")],
        height="0.62\\textheight", cite=t("(own computation)", "(calcul propriu)"))
    reference_frames(D, REFS)
    D.mark_main_end()
    return D


if __name__ == "__main__":
    langs = [a for a in sys.argv[1:] if a in ("en", "ro")] or ["en", "ro"]
    sys.exit(0 if run_build(build, HERE, 0, ABBR, IGNORE, langs=langs, exempt_first_use={"ASE"}) else 1)
