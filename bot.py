import twitchio
from twitchio.ext import commands
import markov_lib
from datetime import datetime
import time, asyncio, re, os, json, signal, sys, shutil, textwrap, aiohttp

CONFIG_FILE = "config.json"
CHANNELS = ["ermugo2", "vedal987"]
ERMS = {"ermugo1", "ermugo2"}
SPECIAL_CHARS = set("!#$%^&*()_+-=[]{}|;':\",./<>?`~\\")

# Config Loader & API Keys Setup
DEFAULT_CFG = {
    "client_id": "", "client_secret": "", "token": "", "refresh_token": "", "bot_id": "1468479097",
    "admin_list": ["ermugo1", "ermugo2"], "user_blocklist": [], "train_list": [], "blocked_words": [], 
    "train_start": "15:00", "train_end": "10:00",
    "default_damping": 0.5,        # Added default damping control to json config
    "default_entropy": 0.65         # Added default context mixing entropy to json config
}

try:
    with open(CONFIG_FILE, "r") as f:
        cfg = {**DEFAULT_CFG, **json.load(f)}
except FileNotFoundError:
    cfg = DEFAULT_CFG.copy()

# Atomic save for config to prevent corruption
def save_cfg():
    tmp = f"{CONFIG_FILE}.tmp"
    with open(tmp, "w") as f: json.dump(cfg, f, indent=2)
    os.replace(tmp, CONFIG_FILE)

# Utilities
bot_instance = markov_lib.MarkovBot()

# Atomic save for brain files to prevent corruption on crash
def save_brain(bot_ref):
    os.makedirs("./brain", exist_ok=True)
    os.makedirs("./backups", exist_ok=True)
    
    cpp_engine = bot_ref.bot_instance
    cpp_engine.save("./brain")
    
    if os.path.exists("./brain/brain.dat.tmp"):
        if os.path.exists("./brain/brain.dat"):
            if os.path.exists("./brain/brain.dat.bak"):
                os.remove("./brain/brain.dat.bak")
            os.rename("./brain/brain.dat", "./brain/brain.dat.bak")
        os.rename("./brain/brain.dat.tmp", "./brain/brain.dat")
        
    if os.path.exists("./brain/vocab.txt.tmp"):
        if os.path.exists("./brain/vocab.txt"):
            os.remove("./brain/vocab.txt")
        os.rename("./brain/vocab.txt.tmp", "./brain/vocab.txt")

    now = datetime.now()
    date_str = now.strftime("%B").lower() + str(now.day)
    backup_filename = f"./backups/brain_backup_{date_str}.dat"
    
    if not os.path.exists(backup_filename) and os.path.exists("./brain/brain.dat"):
        shutil.copy2("./brain/brain.dat", backup_filename)

def clean_shutdown(bot_ref, *_):
    if hasattr(bot_ref, 'autosave_task') and bot_ref.autosave_task:
        bot_ref.autosave_task.cancel()

    if bot_ref.loop and bot_ref.loop.is_running():
        target_channel = bot_ref.get_channel("your_channel_name")
        if target_channel:
            asyncio.run_coroutine_threadsafe(
                target_channel.send("SadCat saving and shutting down..."), 
                bot_ref.loop
            )
            time.sleep(0.2)

    try:
        save_brain(bot_ref)
        if hasattr(bot_ref, 'save_cfg'):
            bot_ref.save_cfg()
    except Exception:
        pass

    sys.exit(0)

def clean_shutdown(bot_ref, *_):
    if hasattr(bot_ref, 'autosave_task') and bot_ref.autosave_task:
        bot_ref.autosave_task.cancel()

    if bot_ref.loop and bot_ref.loop.is_running():
        target_channel = bot_ref.get_channel("your_channel_name")
        if target_channel:
            asyncio.run_coroutine_threadsafe(
                target_channel.send("SadCat saving and shutting down..."), 
                bot_ref.loop
            )
            time.sleep(0.2)

    try:
        save_brain(bot_ref)
        if hasattr(bot_ref, 'save_cfg'):
            bot_ref.save_cfg()
    except Exception:
        pass

    sys.exit(0)

signal.signal(signal.SIGINT, clean_shutdown)
signal.signal(signal.SIGTERM, clean_shutdown)

def clean_spam(text):
    text = re.sub(r'(.)\1{10,}', r'\1'*10, text)
    words = [w for w in text.split() if len(w) > 1] if sum(1 for w in text.split() if len(w) == 1) > 3 else text.split()
    return " ".join(words) or "uuh"

class Bot(commands.Bot):
    def __init__(self, silent=False):
        super().__init__(
            token=cfg["token"],
            client_id=cfg["client_id"],
            client_secret=cfg["client_secret"],
            bot_id=cfg["bot_id"],
            prefix=list(SPECIAL_CHARS),
            initial_channels=CHANNELS
        )
        self.silent, self.sleep, self.train_until, self.erm_bypass = silent, False, 0, True
        self.cd, self.cd_warned, self.last_sent, self.cmd_cd, self.global_cd = {}, set(), 0, 1, 1

    # TwitchIO v2 built-in token expiration hook & manual auto-refresh logic
    async def event_token_expired(self):
        print("Token expired! Attempting automatic refresh...")
        url = "https://id.twitch.tv/oauth2/token"
        params = {"client_id": cfg["client_id"], "client_secret": cfg["client_secret"], "grant_type": "refresh_token", "refresh_token": cfg["refresh_token"]}
        async with aiohttp.ClientSession() as session:
            async with session.post(url, data=params) as resp:
                data = await resp.json()
                if "access_token" in data:
                    cfg["token"], cfg["refresh_token"] = data["access_token"], data.get("refresh_token", cfg["refresh_token"])
                    save_cfg()
                    print("Token successfully refreshed!")
                    return cfg["token"]
                print("Failed to auto-refresh token:", data)
                sys.exit(1)

    # TwitchIO v3 built-in auto-refresher catch hook
    async def event_token_refreshed(self, payload):
        cfg["token"], cfg["refresh_token"] = payload.token, payload.refresh_token
        save_cfg()
        print("Token auto-refreshed by TwitchIO built-in manager.")

    def is_admin(self, user): return user in set(cfg["admin_list"]) | ERMS

    async def safe_reply(self, ctx, text):
        if ctx.channel.name.lower() == "vedal987": return False
        if time.time() - self.last_sent < self.global_cd: return False
        try:
            await ctx.reply(text)
            self.last_sent = time.time()
            return True
        except: return False

    async def event_ready(self):
        print(f"bot ready | joined: {', '.join(CHANNELS)}")
        bot_instance.load("./brain")
        
        # Unlocks TwitchIO v3's automatic background token handling (fails silently on v2)
        if hasattr(self, 'add_token') and cfg["refresh_token"]:
            try: await self.add_token(cfg["token"], cfg["refresh_token"])
            except: pass

        asyncio.create_task(self.autosave())
        if not self.silent:
            for ch in filter(None, map(self.get_channel, CHANNELS)):
                await ch.send("Aloo bot is online 0")

    async def event_message(self, msg):
        if msg.echo or msg.author.name.lower() in {"streamelements", "nightbot", "moobot", "fossabot", self.nick.lower()}: return
        author, content = msg.author.name.lower(), msg.content
        is_cmd = content and content[0] in SPECIAL_CHARS

        if not is_cmd and time.time() < self.train_until and author in cfg["train_list"]: bot_instance.train(content, 2)
        words = content.split()
        if len(words) > 3 and sum(1 for w in words if len(w)==1) / len(words) > 0.8: return # spam check

        if is_cmd:
            if self.sleep and not content.lower()[1:].startswith("unsleep"): return
            if author in cfg["user_blocklist"] and not self.is_admin(author): return
            await self.handle_commands(msg)

    async def autosave(self):
        while True:
            await asyncio.sleep(10)
            now = time.localtime()
            curr = f"{now.tm_hour:02d}:{now.tm_min:02d}"
            s, e = cfg["train_start"], cfg["train_end"]
            in_window = (s <= curr < e) if s <= e else (curr >= s or curr < e)
            
            self.train_until = float('inf') if in_window else (0 if self.train_until == float('inf') else self.train_until)
            if int(time.time()) % 120 < 10:
                save_brain()
                self.cd = {u: t for u, t in self.cd.items() if time.time() - t <= 600}

    # Universal list modifier for admin commands
    async def mod_list(self, ctx, key, args, add=True):
        if not self.is_admin(ctx.author.name.lower()) or not args: return
        s = set(cfg[key])
        changed = [a.lower() for a in args if (a.lower() not in s) == add]
        cfg[key] = sorted(s | set(changed) if add else s - set(changed))
        if changed: save_cfg()
        await self.safe_reply(ctx, f"{'added' if add else 'removed'} {len(changed)} items {'1' if add else '0'}")

    @commands.command()
    async def addblock(self, ctx, *a):
        await self.mod_list(ctx, "blocked_words", a, True)

    @commands.command()
    async def removeblock(self, ctx, *a):
        await self.mod_list(ctx, "blocked_words", a, False)

    @commands.command()
    async def blockuser(self, ctx, *a):
        await self.mod_list(ctx, "user_blocklist", a, True)

    @commands.command()
    async def unblockuser(self, ctx, *a):
        await self.mod_list(ctx, "user_blocklist", a, False)

    @commands.command()
    async def addtrainer(self, ctx, *a):
        await self.mod_list(ctx, "train_list", a, True)

    @commands.command()
    async def removetrainer(self, ctx, *a):
        await self.mod_list(ctx, "train_list", a, False)

    @commands.command()
    async def addadmin(self, ctx, *a): 
        if ctx.author.name.lower() in ERMS:
            await self.mod_list(ctx, "admin_list", a, True)

    @commands.command()
    async def removeadmin(self, ctx, *a): 
        if ctx.author.name.lower() in ERMS:
            await self.mod_list(ctx, "admin_list", a, False) 

    @commands.command()
    async def uuh(self, ctx, *args):
        u, now = ctx.author.name.lower(), time.time()
        if (u not in ERMS or not self.erm_bypass) and now - self.cd.get(u, 0) < self.cmd_cd:
            if u not in self.cd_warned:
                self.cd_warned.add(u)
                await self.safe_reply(ctx, "on cooldown uuh")
            return
        self.cd_warned.discard(u); self.cd[u] = now

        # Initialize core variables and new flags
        w, f, rev, infix = False, False, False, False
        max_w = 30
        damping = cfg["default_damping"]
        context_entropy = cfg["default_entropy"]
        seeds = []

        # Parse through arguments
        for a in args:
            c = ''.join(ch for ch in a if ord(ch) < 128).strip()
            if c == "-w": w = True
            elif c == "-f": f = True
            elif c == "-r": rev = True
            elif c == "-i": infix = True   # Parse the new Infix Flag
            elif c.startswith("-c") and c[2:].isdigit(): max_w = max(1, min(int(c[2:]), 75))
            elif c.startswith("-d"):        # Dynamic execution-level Damping (e.g., -d0.01)
                try: damping = max(0.0, min(float(c[2:]), 1.0))
                except: pass
            elif c.startswith("-e"):        # Dynamic execution-level Entropy (e.g., -e0.40)
                try: context_entropy = max(0.0, min(float(c[2:]), 1.0))
                except: pass
            elif c: seeds.append(c)

        seed = seeds[-1] if seeds else ""

        # FIXED: Positional argument mapping corrected to match new C++ function definitions exactly
        if seed:
            # generate_seeded expects: string seed, int o, bool w, int c, bool r, bool infix, bool f, double damping, double context_entropy
            res = bot_instance.generate_seeded(seed, 2, w, max_w, rev, infix, f, damping, context_entropy) or "0"
        else:
            # generate expects: int o, bool w, int c, bool r, bool f, double damping, double context_entropy
            res = bot_instance.generate(2, w, max_w, rev, f, damping, context_entropy) or "0"

        if seed and res.strip() and not rev and not infix and res != seed: 
            res = f"{seed} {res}"
        
        for b in cfg["blocked_words"]: res = re.sub(re.escape(b), "", res, flags=re.IGNORECASE)
        res = clean_spam(" ".join((res[1:] if res and res[0] in SPECIAL_CHARS else res).split())) or (seed or "0")

        # Split at word boundaries (spaces) ensuring chunks do not exceed 150 chars
        msgs = textwrap.wrap(res, width=150, break_long_words=True) or ["0"]
        await self.safe_reply(ctx, msgs[0])
        for m in msgs[1:]: 
            await asyncio.sleep(1)
            await ctx.reply(m)

    @commands.command() 
    async def helpuuh(self, ctx): 
        await self.safe_reply(ctx, "uuh [seed, w, r, i, f, c1-75, d0-1, e0-1], guh, checkuuh, brainfiles. admin 0 : addblock, removeblock, blockuser, unblockuser, addadmin, removeadmin, addtrainer, removetrainer, sleep, unsleep, train, stoptrain, traintime, killuuh, stats, cooldown 1s global, 0min uuh")

    @commands.command() 
    async def guh(self, ctx): 
        await self.safe_reply(ctx, "SchizoUuh @ermugo1")

    @commands.command() 
    async def bih(self, ctx): 
        await self.safe_reply(ctx, "SchizoUuh @ermugo1")
    
    @commands.command() 
    async def sleep(self, ctx):
        if self.is_admin(ctx.author.name.lower()): 
            self.sleep = True
            await self.safe_reply(ctx, "bot sleeping evilEeping")

    @commands.command() 
    async def unsleep(self, ctx):
        if self.is_admin(ctx.author.name.lower()): 
            self.sleep = False
            await self.safe_reply(ctx, "bot awake Wokege")
    
    @commands.command() 
    async def train(self, ctx):
        if self.is_admin(ctx.author.name.lower()): 
            self.train_until = float('inf')
            await self.safe_reply(ctx, "training enabled 1")

    @commands.command() 
    async def stoptrain(self, ctx):
        if self.is_admin(ctx.author.name.lower()): 
            self.train_until = 0
            await self.safe_reply(ctx, "training disabled 0")
    
    @commands.command() 
    async def killuuh(self, ctx):
        if self.is_admin(ctx.author.name.lower()): 
            save_cfg()
            save_brain()
            await ctx.reply("SadCat saving and shutting down...")
            await self.close()

    @commands.command() 
    async def stats(self, ctx):
        if self.is_admin(ctx.author.name.lower()):
            rem = "inf" if self.train_until == float('inf') else max(0, int(self.train_until - time.time()))
            await self.safe_reply(ctx, f"sleep: {self.sleep}, train_rem: {rem}s, d_damping: {cfg['default_damping']}, d_entropy: {cfg['default_entropy']}, sched: {cfg['train_start']}-{cfg['train_end']}")

    @commands.command() 
    async def traintime(self, ctx, start=None, end=None):
        if not self.is_admin(ctx.author.name.lower()): 
            return
        if not start or not end: 
            return await self.safe_reply(ctx, f"current train time: {cfg['train_start']} to {cfg['train_end']} 0")
        cfg['train_start'], cfg['train_end'] = start, end
        save_cfg()
        await self.safe_reply(ctx, f"train time set {start} to {end} 1")

    @commands.command()
    async def checkuuh(self, ctx, user=None):
        u = (user or ctx.author.name).lower()
        if u in ERMS: return await self.safe_reply(ctx, f"@{u} SchizoUuh")
        sts = [s for c, s in [(u in cfg["user_blocklist"], "blocked uuh "), (u in cfg["admin_list"], "admin MONKA "), (u in cfg["train_list"], " 0")] if c]
        await self.safe_reply(ctx, f"{u} is {', '.join(sts) if sts else 'trusted ok'}")

    @commands.command()
    async def brainfiles(self, ctx):
        if not os.path.exists("./brain") or not os.listdir("./brain"): return await ctx.send("brain folder missing or empty")
        t_sz = t_ln = t_wd = 0
        sizes = []
        for f in os.listdir("./brain"):
            path = f"./brain/{f}"
            if os.path.isfile(path):
                sz, lines = os.path.getsize(path), open(path, "r", errors="ignore").readlines()
                t_sz += sz; t_ln += len(lines); t_wd += sum(len(l.split()) for l in lines)
                sizes.append(f"{f}: {sz/1024**2:.1f}MB" if sz > 1024**2 else f"{f}: {sz/1024:.1f}KB")
        await ctx.send(f"total: {t_sz/1024**2:.1f}MB [{t_ln} lines, {t_wd} words] | " + " | ".join(sizes))

    @commands.command()
    async def debug(self, ctx, cmd=None, val=None):
        if not self.is_admin(ctx.author.name.lower()): return
        if not cmd: return await ctx.reply(f"usercooldown: {self.cmd_cd}s, globalcooldown: {self.global_cd}s, damping: {cfg['default_damping']}, entropy: {cfg['default_entropy']}")
        if cmd == "usercooldown" and val: self.cmd_cd = int(val); await ctx.reply(f"usercooldown set to {val}s")
        elif cmd == "globalcooldown" and val: self.global_cd = int(val); await ctx.reply(f"globalcooldown set to {val}s")
        elif cmd == "damping" and val: 
            cfg['default_damping'] = max(0.0, min(float(val), 1.0)); save_cfg()
            await ctx.reply(f"global default damping set to {cfg['default_damping']}")
        elif cmd == "entropy" and val:
            cfg['default_entropy'] = max(0.0, min(float(val), 1.0)); save_cfg()
            await ctx.reply(f"global default entropy set to {cfg['default_entropy']}")
        elif cmd == "bypass": self.erm_bypass = not self.erm_bypass; await ctx.reply(f"bypass disabled: {self.erm_bypass}")
        else: await ctx.reply("unknown setting uuh")

    @commands.command()
    async def dailyreport(self, ctx):
        month_file = "./brain/brain_month.txt"
        top_10_list = []

        if os.path.exists(month_file):
            counts = {}
            with open(month_file, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if " : " in line:
                        w, c = line.strip().split(" : ", 1)
                        # Skip words 3 letters or shorter to keep spam out of the top 10
                        if len(w) > 3:
                            counts[w] = int(c)
            
            # Sort descending and isolate top 10
            sorted_words = sorted(counts.items(), key=lambda item: item[1], reverse=True)
            top_10_list = sorted_words[:10]

        # Format output inline: "1 [word] (300 er), 2 [word] (200 er), ..."
        if top_10_list:
            inline_report = ", ".join([f"{i+1} {w} ({c} er)" for i, (w, c) in enumerate(top_10_list)])
        else:
            inline_report = "0 words this month SchizoUuh"

        await self.safe_reply(ctx, inline_report)
    
    async def on_command_error(self, ctx, error):
        # Catch the "Command not found" error and silently drop it
        if isinstance(error, commands.CommandNotFound):
            return
        
        # Print other actual errors
        print(f"[Error] {error}")

if __name__ == "__main__":
    Bot(silent="-silent" in sys.argv).run()