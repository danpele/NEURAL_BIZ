<!--
Web version of the smoke test (EN). Built into web/L0.html by lectures/common/build_web_lecture.py.
-->

## @intro

This page only tests the pipeline: one chart computed from real data, its source line, its Quantlet and the link to the slide.

@goals
- **Check** that the deck, the notebook, the Quantlet and the web page are built from the same numbers.

# A real chart

The chart below is computed from the daily closing values of the S&P 500 index.

## Equity losses come in long episodes {h=780afcb2}

The drawdown $D_t = P_t / \max_{s \le t} P_s - 1$ measures how far the index is below its previous peak. The maximum drawdown is −56.78 % on 9 March 2009.

@chart sp500_drawdown
Drawdown of the S&P 500 from its previous peak.

@practice A model trained only on a calm period has never seen a drawdown of this size.

## @end

The same numbers appear on the slide, in the notebook and in the Quantlet.
