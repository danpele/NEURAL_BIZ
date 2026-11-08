"""
fitgrow.py -- figure auto-fit for the decks of Lectures 1-3: the shared nn_build.autofit only shrinks figures, and with
width-limited charts (keepaspectratio) it can overshoot and leave a frame half empty. This version shrinks the figures of
overflowing frames as before, then grows every figure back in small steps for as long as its frame still fits.
Installed by the builders with `fitgrow.install(nn_build)`; lectures/common is not modified.
"""


def install(nn_build):
    compile_tex = nn_build.compile_tex

    def frame_of(D, title):
        tt = title.strip().lstrip("{").rstrip("}")[:60]
        return next((k for k in D.frame_figs if k.startswith(tt) or tt.startswith(k[:60])), None)

    def autofit(build_fn, out_path, lang, max_iter=8, textheight_pt=230.0, min_h=0.18, step=0.03, max_h=0.66):
        H = {}
        D = tex = res = None
        # 1. shrink until nothing overflows
        for _ in range(max_iter):
            D = build_fn(lang, H)
            tex = D.write(out_path)
            res = compile_tex(out_path)
            changed = False
            for kind, amt, title in res["where"]:
                if kind != "v":
                    continue
                key = frame_of(D, title)
                figs = D.frame_figs.get(key) if key else None
                if not figs:
                    continue
                cut = (amt + 3.0) / textheight_pt / len(figs)
                for n in figs:
                    new = max(min_h, D.used_h[n] - max(cut, step))
                    if new < D.used_h[n] - 1e-6:
                        H[n] = round(new, 3)
                        changed = True
            if not changed:
                break
        # 2. grow back: every figure whose frame fits gets +step until its frame overflows (then it stays one step back)
        cur = dict(D.used_h)
        frozen = set()
        for _ in range(12):
            grow = [n for n in cur if n not in frozen and cur[n] + step <= max_h]
            if not grow:
                break
            trial = dict(H)
            trial.update({n: cur[n] for n in cur})
            for n in grow:
                trial[n] = round(cur[n] + step, 3)
            D2 = build_fn(lang, trial)
            D2.write(out_path)
            res2 = compile_tex(out_path)
            bad_figs = set()
            for kind, amt, title in res2["where"]:
                key = frame_of(D2, title)
                bad_figs |= set(D2.frame_figs.get(key, []) if key else [])
            frozen |= bad_figs & set(grow)
            for n in grow:
                if n not in bad_figs:
                    cur[n] = trial[n]
                    H[n] = trial[n]
        # final build with the accepted heights
        D = build_fn(lang, H)
        tex = D.write(out_path)
        res = compile_tex(out_path)
        return D, tex, res, H

    nn_build.autofit = autofit
