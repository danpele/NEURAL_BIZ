#!/usr/bin/env python3
"""
check_site.py -- local check of the course site (not published): serves the repository with python -m http.server,
opens each page in headless Google Chrome through the DevTools protocol and reports
  * console errors and uncaught exceptions,
  * failed network requests (HTTP >= 400 or network errors), local files only,
  * relative links (href/src) of the rendered page that do not resolve to a file of the repository.
Usage:  python3 tools/check_site.py [page ...]      (default: index.html index_ro.html and every web/*.html)
Exit code 0 only if every page is clean.
"""
import glob
import json
import os
import socket
import subprocess
import sys
import time
import urllib.parse
import urllib.request

import websocket

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def check_page(ws_url, url, wait=6.0):
    ws = websocket.create_connection(ws_url, timeout=30)
    n = [0]

    def send(method, params=None):
        n[0] += 1
        ws.send(json.dumps({"id": n[0], "method": method, "params": params or {}}))
        return n[0]

    for m in ("Runtime.enable", "Log.enable", "Network.enable", "Page.enable"):
        send(m)
    send("Page.navigate", {"url": url})
    errors, failed, reqs = [], [], {}
    t0 = time.time()
    ws.settimeout(0.5)
    while time.time() - t0 < wait:
        try:
            msg = json.loads(ws.recv())
        except websocket.WebSocketTimeoutException:
            continue
        meth, p = msg.get("method"), msg.get("params", {})
        if meth == "Runtime.consoleAPICalled" and p.get("type") in ("error", "assert"):
            errors.append("console: " + " ".join(str(a.get("value", a.get("description", ""))) for a in p.get("args", [])))
        elif meth == "Runtime.exceptionThrown":
            d = p.get("exceptionDetails", {})
            errors.append("exception: " + (d.get("exception", {}).get("description") or d.get("text", ""))[:300])
        elif meth == "Log.entryAdded" and p.get("entry", {}).get("level") == "error":
            e = p["entry"]
            errors.append(f"log: {e.get('text', '')} {e.get('url', '')}")
        elif meth == "Network.requestWillBeSent":
            reqs[p["requestId"]] = p["request"]["url"]
        elif meth == "Network.responseReceived":
            r = p["response"]
            if r.get("status", 200) >= 400 and r["url"].startswith("http://127.0.0.1"):
                failed.append(f"HTTP {r['status']} {r['url']}")
        elif meth == "Network.loadingFailed":
            u = reqs.get(p["requestId"], "?")
            if u.startswith("http://127.0.0.1") and not p.get("canceled"):
                failed.append(f"failed {p.get('errorText')} {u}")
    # links of the rendered DOM
    i = send("Runtime.evaluate", {"expression": "JSON.stringify([...document.querySelectorAll('[href],[src]')].map(e => e.getAttribute('href') || e.getAttribute('src')))",
                                  "returnByValue": True})
    links = []
    ws.settimeout(10)
    while True:
        msg = json.loads(ws.recv())
        if msg.get("id") == i:
            links = json.loads(msg["result"]["result"]["value"])
            break
    ws.close()
    return errors, failed, links


def bad_links(page, links):
    base = os.path.dirname(os.path.join(ROOT, page))
    bad = []
    for l in set(links):
        if not l or l.startswith(("http:", "https:", "mailto:", "#", "data:", "javascript:")):
            continue
        path = urllib.parse.unquote(l.split("#")[0].split("?")[0])
        if not path:
            continue
        if not os.path.exists(os.path.normpath(os.path.join(base, path))):
            bad.append(l)
    return sorted(bad)


def main(pages):
    port, dport = free_port(), free_port()
    srv = subprocess.Popen([sys.executable, "-m", "http.server", str(port), "--bind", "127.0.0.1"], cwd=ROOT,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    prof = os.path.join(ROOT, "tools", ".chrome-profile")
    chrome = subprocess.Popen([CHROME, "--headless=new", f"--remote-debugging-port={dport}", f"--user-data-dir={prof}",
                               "--no-first-run", "--disable-gpu", "--window-size=1400,1000", "about:blank"],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    ok = True
    try:
        for _ in range(50):
            try:
                tabs = json.load(urllib.request.urlopen(f"http://127.0.0.1:{dport}/json"))
                if any(t.get("type") == "page" for t in tabs):
                    break
            except OSError:
                pass
            time.sleep(0.2)
        ws_url = next(t["webSocketDebuggerUrl"] for t in tabs if t.get("type") == "page")
        for page in pages:
            errors, failed, links = check_page(ws_url, f"http://127.0.0.1:{port}/{page}")
            bl = bad_links(page, links)
            clean = not (errors or failed or bl)
            ok &= clean
            print(f"{'OK  ' if clean else 'FAIL'} {page}: {len(errors)} console errors, {len(failed)} failed requests, "
                  f"{len(links)} links ({len(bl)} broken)")
            for x in errors + failed + [f"broken link: {b}" for b in bl]:
                print("     ", x)
    finally:
        chrome.terminate()
        srv.terminate()
    return ok


if __name__ == "__main__":
    pages = sys.argv[1:] or (["index.html", "index_ro.html"] +
                             sorted(os.path.relpath(p, ROOT) for p in glob.glob(os.path.join(ROOT, "web", "**", "*.html"), recursive=True)))
    sys.exit(0 if main(pages) else 1)
