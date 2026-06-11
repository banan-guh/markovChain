import twitchio
from twitchio.ext import commands
import markov_lib
from datetime import datetime
import time, asyncio, re, os, json, signal, sys, shutil, textwrap, aiohttp
from collections import Counter

CONFIG_FILE = "config.json"
CHANNELS = ["ermugo2", "vedal987"]
ERMS = {"ermugo1", "ermugo2"}
SPECIAL_CHARS = set("!#$%^&*()_+-=[]{}|;':\",./<>?`~\\")

DEFAULT_CFG = {
    "client_id": "", "client_secret": "", "token": "", "refresh_token": "", "bot_id": "1468479097",
    "admin_list": ["ermugo1", "ermugo2"], "user_blocklist": [], "train_list": [], "blocked_words": [], 
    "train_start": "15:00", "train_end": "10:00",
    "default_damping": 0.25,
    "default_entropy": 0.2
}

try:
    with open(CONFIG_FILE, "r") as f:
        cfg = {**DEFAULT_CFG, **json.load(f)}
except FileNotFoundError:
    cfg = DEFAULT_CFG.copy()

def save_cfg():
    tmp = f"{CONFIG_FILE}.tmp"
    with open(tmp, "w") as f: json.dump(cfg, f, indent=2)
    os.replace(tmp, CONFIG_FILE)

bot_instance = markov_lib.MarkovBot()

def parse_log_line(line):
    parts = line.strip().split(" | ", 3)
    if len(parts) != 4:
        return None
    timestamp_str, channel, author, content = parts
    timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
    return {"timestamp": timestamp, "channel": channel, "author": author.lower(), "content": content}

def format_training_data(log_path, output_path, target_user="ermugo1", window_seconds=45, min_context=2):
    with open(log_path, "r", encoding="utf-8") as f:
        lines = [parse_log_line(l) for l in f if l.strip()]
    lines = [l for l in lines if l]

    pairs = []
    for i, msg in enumerate(lines):
        if msg["author"] != target_user:
            continue
        # grab context window
        context = []
        for j in range(i - 1, -1, -1):
            prev = lines[j]
            if (msg["timestamp"] - prev["timestamp"]).total_seconds() > window_seconds:
                break
            if prev["author"] == target_user:
                continue  # skip your own prior messages in context
            context.insert(0, prev)
        
        if len(context) < min_context:
            continue

        context_str = "\n".join(f"{m['author']}: {m['content']}" for m in context)
        pairs.append({
            "messages": [
                {"role": "user", "content": context_str},
                {"role": "assistant", "content": msg["content"]}
            ]
        })

    with open(output_path, "w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair) + "\n")

    #print(f"wrote {len(pairs)} pairs to {output_path}")

def track_monthly_words(message_text):
    month_file = "./brain/brain_month.txt"
    now = datetime.now()
    words = re.findall(r'\b\w+\b', message_text)
    if not words:
        return

    if os.path.exists(month_file):
        mtime = datetime.fromtimestamp(os.path.getmtime(month_file))
        if mtime.month != now.month or mtime.year != now.year:
            try: os.remove(month_file)
            except OSError: pass

    counts = Counter()
    if os.path.exists(month_file):
        with open(month_file, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if " : " in line:
                    w, c = line.strip().split(" : ", 1)
                    counts[w] = int(c)

    counts.update(words)
    with open(month_file, "w", encoding="utf-8") as f:
        for w, c in counts.items():
            f.write(f"{w} : {c}\n")

def save_brain(bot_ref):
    os.makedirs("./brain", exist_ok=True)
    os.makedirs("./backups", exist_ok=True)
    
    # Legacy text-saving methods
    cpp_engine = bot_ref.bot_instance
    cpp_engine.save("./brain/memory.dat")
    cpp_engine.save("./brain/reverse_memory.dat")
    cpp_engine.save("./brain/vocab.txt")

    # Keep your custom backup feature running on the legacy files
    now = datetime.now()
    date_str = now.strftime("%B").lower() + str(now.day)
    backup_filename = f"./backups/memory_backup_{date_str}.dat"
    
    if not os.path.exists(backup_filename) and os.path.exists("./brain/memory.dat"):
        shutil.copy2("./brain/memory.dat", backup_filename)

def clean_shutdown(bot_ref, sig, frame):
    if hasattr(bot_ref, 'autosave_task') and bot_ref.autosave_task:
        bot_ref.autosave_task.cancel()

    if bot_ref.loop and bot_ref.loop.is_running():
        # Grabs the first live channel configured instead of using a placeholder string
        target_channel = bot_ref.get_channel(CHANNELS[0])
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
            initial_channels=CHANNELS,
            loop=asyncio.get_event_loop(),
        )
        self.silent, self.sleep, self.train_until, self.erm_bypass = silent, False, 0, True
        self.cd, self.cd_warned, self.last_sent, self.cmd_cd, self.global_cd = {}, set(), 0, 1, 1
        self.last_jsonl_update = 0
        self.autosave_task = None
        self.bot_instance = bot_instance

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
        
        # Legacy individual file loading
        self.bot_instance.load("./brain")
        
        if hasattr(self, 'add_token') and cfg["refresh_token"]:
            try: await self.add_token(cfg["token"], cfg["refresh_token"])
            except: pass

        self.autosave_task = asyncio.create_task(self.autosave())
        if not self.silent:
            for ch in filter(None, map(self.get_channel, CHANNELS)):
                await ch.send("Aloo bot is online 0")

    async def event_message(self, msg):
        if msg.echo or msg.author.name.lower() in {"streamelements", "nightbot", "moobot", "fossabot", self.nick.lower()}: return
        author, content = msg.author.name.lower(), msg.content
        is_cmd = content and content[0] in SPECIAL_CHARS

        # Track vocabulary changes per month asynchronously in background log
        if not msg.echo and msg.author.name.lower() != self.nick.lower():
            track_monthly_words(content)

        if not is_cmd and time.time() < self.train_until and author in cfg["train_list"]: self.bot_instance.train(content, 2)
        words = content.split()
        if len(words) > 3 and sum(1 for w in words if len(w)==1) / len(words) > 0.8: return

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
            if int(time.time()) % 3600 < 10:
                save_brain(self) # FIXED: Added positional self reference requirement
                self.cd = {u: t for u, t in self.cd.items() if time.time() - t <= 600}
            if time.time() - self.last_jsonl_update > 43200:  # 12 hours
                format_training_data("./logs/chat_log.txt", "./training_data.jsonl")
                self.last_jsonl_update = time.time()


    async def mod_list(self, ctx, key, args, add=True):
        if not self.is_admin(ctx.author.name.lower()) or not args: return
        s = set(cfg[key])
        changed = [a.lower() for a in args if (a.lower() not in s) == add]
        cfg[key] = sorted(s | set(changed) if add else s - set(changed))
        if changed: save_cfg()
        await self.safe_reply(ctx, f"{'added' if add else 'removed'} {len(changed)} items {'1' if add else '0'}")

    @commands.command()
    async def addblock(self, ctx, *a): await self.mod_list(ctx, "blocked_words", a, True)

    @commands.command()
    async def removeblock(self, ctx, *a): await self.mod_list(ctx, "blocked_words", a, False)

    @commands.command()
    async def blockuser(self, ctx, *a): await self.mod_list(ctx, "user_blocklist", a, True)

    @commands.command()
    async def unblockuser(self, ctx, *a): await self.mod_list(ctx, "user_blocklist", a, False)

    @commands.command()
    async def addtrainer(self, ctx, *a): await self.mod_list(ctx, "train_list", a, True)

    @commands.command()
    async def removetrainer(self, ctx, *a): await self.mod_list(ctx, "train_list", a, False)

    @commands.command()
    async def addadmin(self, ctx, *a): 
        if ctx.author.name.lower() in ERMS: await self.mod_list(ctx, "admin_list", a, True)

    @commands.command()
    async def removeadmin(self, ctx, *a): 
        if ctx.author.name.lower() in ERMS: await self.mod_list(ctx, "admin_list", a, False) 

    @commands.command()
    async def uuh(self, ctx, *args):
        u, now = ctx.author.name.lower(), time.time()
        if (u not in ERMS or not self.erm_bypass) and now - self.cd.get(u, 0) < self.cmd_cd:
            if u not in self.cd_warned:
                self.cd_warned.add(u)
                await self.safe_reply(ctx, "on cooldown uuh")
            return
        self.cd_warned.discard(u); self.cd[u] = now

        w, f, rev, infix = False, False, False, False
        max_w = 30
        damping = cfg["default_damping"]
        context_entropy = cfg["default_entropy"]
        seeds = []

        for a in args:
            c = ''.join(ch for ch in a if ord(ch) < 128).strip()
            if c == "-w": w = True
            elif c == "-f": f = True
            elif c == "-r": rev = True
            elif c == "-i": infix = True   
            elif c.startswith("-c") and c[2:].isdigit(): max_w = max(1, min(int(c[2:]), 75))
            elif c.startswith("-d"):        
                try: damping = max(0.0, min(float(c[2:]), 1.0))
                except: pass
            elif c.startswith("-e"):        
                try: context_entropy = max(0.0, min(float(c[2:]), 1.0))
                except: pass
            elif c: seeds.append(c)

        seed = seeds[-1] if seeds else ""

        if seed:
            # Order: seed, o, w, c, r, infix, f, damping, context_entropy
            res = self.bot_instance.generate_seeded(seed, 2, w, max_w, rev, infix, f, damping, context_entropy) or "0"
        else:
            # Order: o, w, c, r, f, damping, context_entropy
            res = self.bot_instance.generate(2, w, max_w, rev, f, damping, context_entropy) or "0"

        if seed and res.strip() and not rev and not infix and res != seed: 
            res = f"{seed} {res}"
        
        for b in cfg["blocked_words"]: res = re.sub(re.escape(b), "", res, flags=re.IGNORECASE)
        res = clean_spam(" ".join((res[1:] if res and res[0] in SPECIAL_CHARS else res).split())) or (seed or "0")

        msgs = textwrap.wrap(res, width=150, break_long_words=True) or ["0"]
        await self.safe_reply(ctx, msgs[0])
        for m in msgs[1:]: 
            await asyncio.sleep(1)
            await ctx.reply(m)

    @commands.command() 
    async def helpuuh(self, ctx): 
        await self.safe_reply(ctx, "uuh [seed, w, r, i, f, c1-75, d0-1, e0-1], guh, bih, dailyreport, checkuuh, brainfiles. admin 0 : addblock, removeblock, blockuser, unblockuser, addadmin, removeadmin, addtrainer, removetrainer, sleep, unsleep, train, stoptrain, traintime, killuuh, debug, stats, cooldown 1s global, 0min uuh")

    @commands.command() 
    async def guh(self, ctx): await self.safe_reply(ctx, "SchizoUuh @ermugo1")

    @commands.command() 
    async def bih(self, ctx): await self.safe_reply(ctx, "SchizoUuh @ermugo1")
    
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
            save_brain(self) # FIXED: Appended instance requirement parameters
            await ctx.reply("SadCat saving and shutting down...")
            await self.close()

    @commands.command() 
    async def stats(self, ctx):
        if self.is_admin(ctx.author.name.lower()):
            rem = "inf" if self.train_until == float('inf') else max(0, int(self.train_until - time.time()))
            await self.safe_reply(ctx, f"sleep: {self.sleep}, train_rem: {rem}s, d_damping: {cfg['default_damping']}, d_entropy: {cfg['default_entropy']}, sched: {cfg['train_start']}-{cfg['train_end']}")

    @commands.command() 
    async def traintime(self, ctx, start=None, end=None):
        if not self.is_admin(ctx.author.name.lower()): return
        if not start or not end: return await self.safe_reply(ctx, f"current train time: {cfg['train_start']} to {cfg['train_end']} 0")
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
                sz = os.path.getsize(path)
                lines = open(path, "r", errors="ignore").readlines()
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
                        if len(w) > 3:
                            counts[w] = int(c)
            
            sorted_words = sorted(counts.items(), key=lambda item: item[1], reverse=True)
            top_10_list = sorted_words[:10]

        if top_10_list:
            inline_report = ", ".join([f"{i+1} {w} ({c} er )" for i, (w, c) in enumerate(top_10_list)])
        else:
            inline_report = "0 words this month SchizoUuh"

        await self.safe_reply(ctx, inline_report)
    
    async def event_command_error(self, ctx, error):
        if isinstance(error, commands.CommandNotFound):
            return
        print(f"[Error] {error}")

async def pre_boot_refresh():
    print("Checking token validity before boot...")
    url = "https://id.twitch.tv/oauth2/validate"
    headers = {"Authorization": f"OAuth {cfg['token']}"}
    
    async with aiohttp.ClientSession() as session:
        async with session.get(url, headers=headers) as resp:
            # If the token is valid, we can safely exit this check
            if resp.status == 200:
                print("Token is valid!")
                return

        # If it failed validation, use the refresh token immediately
        print("Initial token invalid. Triggering pre-boot refresh...")
        refresh_url = "https://id.twitch.tv/oauth2/token"
        params = {
            "client_id": cfg["client_id"], 
            "client_secret": cfg["client_secret"], 
            "grant_type": "refresh_token", 
            "refresh_token": cfg["refresh_token"]
        }
        async with session.post(refresh_url, data=params) as resp:
            data = await resp.json()
            if "access_token" in data:
                cfg["token"] = data["access_token"]
                cfg["refresh_token"] = data.get("refresh_token", cfg["refresh_token"])
                save_cfg()
                print("Token successfully refreshed pre-boot!")
            else:
                print("CRITICAL: Failed to refresh token pre-boot. Check your client credentials.", data)
                sys.exit(1)

if __name__ == "__main__":
    # 1. Create a persistent event loop for the main thread
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # 2. Run the token verification step inside this persistent loop
    loop.run_until_complete(pre_boot_refresh())

    # 3. Reload the configuration with the updated token
    with open(CONFIG_FILE, "r") as f:
        cfg = {**DEFAULT_CFG, **json.load(f)}

    # 4. Instantiate the Bot (the class __init__ will automatically pick up our loop)
    bot = Bot(silent="-silent" in sys.argv)
    
    signal.signal(signal.SIGINT, lambda sig, frame: clean_shutdown(bot, sig, frame))
    signal.signal(signal.SIGTERM, lambda sig, frame: clean_shutdown(bot, sig, frame))
    
    # 5. Start the bot up normally
    bot.run()