# training/colab_auth.py
import os
import sys
import json
from importlib import resources
from google_auth_oauthlib.flow import InstalledAppFlow

PUBLIC_SCOPES = [
    "openid",
    "https://www.googleapis.com/auth/userinfo.profile",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/cloud-platform",
    "https://www.googleapis.com/auth/colaboratory",
    "https://www.googleapis.com/auth/drive.file",
]
REMOTE_REDIRECT_URI = "https://sdk.cloud.google.com/applicationdefaultauthcode.html"
CONFIG_DIR = os.path.expanduser("~/.config/colab-cli")
PENDING_FILE = os.path.join(CONFIG_DIR, "auth_pending.json")
TOKEN_FILE = os.path.join(CONFIG_DIR, "token.json")


def get_client_config():
    config_resource = resources.files("colab_cli").joinpath("oauth_config.json")
    return json.loads(config_resource.read_text())


def get_url():
    os.makedirs(CONFIG_DIR, exist_ok=True)
    cfg = get_client_config()
    flow = InstalledAppFlow.from_client_config(cfg, PUBLIC_SCOPES)
    flow.redirect_uri = REMOTE_REDIRECT_URI
    auth_url, state = flow.authorization_url(prompt="consent", token_usage="remote")

    state_data = {
        "state": state,
        "code_verifier": flow.code_verifier,
    }
    with open(PENDING_FILE, "w", encoding="utf-8") as f:
        json.dump(state_data, f)

    print("AUTH_URL_START")
    print(auth_url)
    print("AUTH_URL_END")


def exchange(code):
    if not os.path.exists(PENDING_FILE):
        print("ERROR: No pending auth request found. Run 'get-url' first.", file=sys.stderr)
        sys.exit(1)

    with open(PENDING_FILE, "r", encoding="utf-8") as f:
        state_data = json.load(f)

    cfg = get_client_config()
    flow = InstalledAppFlow.from_client_config(cfg, PUBLIC_SCOPES, state=state_data["state"])
    flow.redirect_uri = REMOTE_REDIRECT_URI
    flow.code_verifier = state_data["code_verifier"]

    flow.fetch_token(code=code.strip())
    creds = flow.credentials

    with open(TOKEN_FILE, "w", encoding="utf-8") as f:
        f.write(creds.to_json())

    os.remove(PENDING_FILE)
    print("SUCCESS: Token saved to", TOKEN_FILE)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python colab_auth.py [get-url | exchange <code>]")
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd == "get-url":
        get_url()
    elif cmd == "exchange":
        if len(sys.argv) < 3:
            print("Error: missing authorization code")
            sys.exit(1)
        exchange(sys.argv[2])
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)
