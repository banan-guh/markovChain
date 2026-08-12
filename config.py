import json, os, logging

CONFIG_FILE = "config.json"
ERMS = {"ermugo1", "ermugo2"}
SPECIAL_CHARS = set("!@$%^&*()_+-=[]{}|;':\",./<>?`~\\")

DEFAULT_CFG = {
    "client_id": "",
    "client_secret": "",
    "bot_id": "1468479097",
    "erm_list": ["1468479097", "974273622"],
    "admin_list": [],
    "banned_users": [],
    "dont_trainlist": [],
    "blocked_words": [], 
    "default_damping": 0.25,
    "default_entropy": 0.2,
    "training": False,
    "cooldown": 0,
    "start_time": [16, 30],
    "end_time": [10, 0],
    "sleep": False
}

LOGGER: logging.Logger = logging.getLogger("Bot")

def save_cfg():
    tmp = f"{CONFIG_FILE}.tmp"
    with open(tmp, "w") as f: json.dump(cfg, f, indent=2)
    os.replace(tmp, CONFIG_FILE)


try:
    with open(CONFIG_FILE, "r") as f:
        cfg = {**DEFAULT_CFG, **json.load(f)}
except FileNotFoundError:
    cfg = DEFAULT_CFG.copy()
    save_cfg()