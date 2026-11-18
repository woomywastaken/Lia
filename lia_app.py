"""Starts Lia: runs the server on this computer and opens the page in its own window.

Used three ways:   python lia_app.py      start_lia.bat      Lia.exe (built by build_exe.bat)

Options:
  --demo      canned answers, no AI model needed (same as LIA_BACKEND=fake)
  --browser   open in your normal browser instead of an app window
  --port N    use another port (default 8000)

LM Studio stays a separate program: open it, load MedGemma, and start its local server.
Lia connects to it by itself, and shows "Model offline" in the corner until it can.
"""
import argparse, json, os, socket, subprocess, sys, threading, time, urllib.request, webbrowser
from pathlib import Path

APP_DIR = Path(sys.executable).parent if getattr(sys, "frozen", False) else Path(__file__).parent

def port_in_use(port):
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0

def lia_health(port):
    """The /health answer if a Lia server is on this port, else None."""
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=6) as r:
            d = json.load(r)
        return d if "backend" in d else None
    except Exception:
        return None

def find_browser():
    """Edge or Chrome, for a clean app window with no address bar. Edge is on every Windows 10/11 computer."""
    env = os.environ.get
    spots = [(env("ProgramFiles(x86)"), r"Microsoft\Edge\Application\msedge.exe"),
             (env("ProgramFiles"), r"Microsoft\Edge\Application\msedge.exe"),
             (env("ProgramFiles"), r"Google\Chrome\Application\chrome.exe"),
             (env("ProgramFiles(x86)"), r"Google\Chrome\Application\chrome.exe"),
             (env("LocalAppData"), r"Google\Chrome\Application\chrome.exe"),
             ("/Applications/Google Chrome.app/Contents/MacOS", "Google Chrome"),
             ("/Applications/Microsoft Edge.app/Contents/MacOS", "Microsoft Edge")]
    for base, rest in spots:
        if base and (Path(base) / rest).exists():
            return str(Path(base) / rest)
    import shutil
    for name in ("microsoft-edge", "google-chrome", "chromium", "chromium-browser"):
        if shutil.which(name):
            return shutil.which(name)
    return None

def open_window(url, app_dir, use_browser):
    """Opens Lia. Returns the window's process when we can tell when it closes, else None."""
    exe = None if use_browser else find_browser()
    if exe:
        try:
            profile = app_dir / "lia_window"           # Lia's saved checks and camera/mic permission live here
            return subprocess.Popen([exe, f"--app={url}", f"--user-data-dir={profile}", "--no-first-run",
                                     "--no-default-browser-check", "--window-size=1320,880"],
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass
    webbrowser.open(url)
    return None

def main():
    try:
        sys.stdout.reconfigure(line_buffering=True)     # show messages right away
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="Start Lia")
    ap.add_argument("--demo", action="store_true", help="canned answers, no AI model")
    ap.add_argument("--browser", action="store_true", help="open in the normal browser")
    ap.add_argument("--port", type=int, default=int(os.environ.get("LIA_PORT", "8000")))
    args = ap.parse_args()
    if args.demo:
        os.environ["LIA_BACKEND"] = "fake"

    port = args.port
    if port_in_use(port):
        old = lia_health(port)
        if old:
            print(f"Lia is already running on this computer (mode: {old['backend']}). Opening it.")
            print("If that is an old copy in the wrong mode, close its window first, then start Lia again.")
            open_window(f"http://127.0.0.1:{port}", APP_DIR, args.browser)
            time.sleep(4)
            return
        while port_in_use(port):
            port += 1
        print(f"Port {args.port} is used by another program, so Lia will use {port}. "
              "(Saved checks are kept per port, so earlier ones will not show.)")

    import uvicorn
    import server
    url = f"http://127.0.0.1:{port}"
    srv = uvicorn.Server(uvicorn.Config(server.app, host="127.0.0.1", port=port, log_level="warning"))

    def after_start():
        while not srv.started:
            time.sleep(0.1)
        threading.Thread(target=server.warm_up, daemon=True).start()   # loads voice and speech models in the background
        print("\n  Lia is running at", url)
        h = lia_health(port) or {}
        if h.get("backend") == "fake":
            print("  DEMO MODE: canned answers, no AI model.")
        elif h.get("ok"):
            print("  Connected to the model:", h.get("model", h.get("backend")))
        else:
            print("  NOT CONNECTED to the model yet. Open LM Studio, load MedGemma and start its server.")
            print("  Lia will connect by itself once it is on. No restart needed.")
        print("  Voice in:", "yes" if h.get("stt") else "no", "| Natural voice:", "yes" if h.get("tts") else "no",
              "| Face sign-in:", "yes" if h.get("face") else "no")
        print("\n  Close this window to stop Lia.\n")
        t0 = time.time()
        win = open_window(url, APP_DIR, args.browser)
        if win is not None:
            win.wait()
            if time.time() - t0 > 8:                    # the Lia window was really open and has now been closed
                srv.should_exit = True                  # (a quick exit means the browser handed off, so keep running)

    threading.Thread(target=after_start, daemon=True).start()
    srv.run()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    except Exception as e:                              # keep the window open so the message can be read
        import traceback
        traceback.print_exc()
        print("\nLia could not start:", e)
        if getattr(sys, "frozen", False):
            input("Press Enter to close.")
