from twitchio.ext import commands
from math import isclose
import config
from config import cfg, save_cfg, LOGGER
import re, textwrap, os, shutil
from datetime import datetime

import markov_lib


markov_bot = markov_lib.MarkovBot()
markov_bot.load("./brain")


message_buffer: list[str] = []
time_last_commit = datetime.now()


def save_brain():
    train_from_buffer(message_buffer)
    os.makedirs("./brain", exist_ok=True)
    os.makedirs("./backups", exist_ok=True)
    
    markov_bot.save("./brain")

    now = datetime.now() # prevent very rare edge case
    date_str = now.strftime("%B").lower() + str(now.day) + "-" + str(now.year)
    backup_folder = f"./backups/brain_backup_{date_str}"
    if not os.path.exists(backup_folder):
        shutil.copytree("./brain", backup_folder, dirs_exist_ok=False)


def moderate_spam(text):
    text = re.sub(r'(.)\1{10,}', r'\1' * 10, text)

    words = text.split()
    single_char_count = sum(1 for w in words if len(w) == 1)

    if single_char_count > 3:
        words = [w for w in words if len(w) > 1]
    
    for word in words:
        if word in cfg["blocked_words"]: return "1984 moderation"

    return " ".join(words) or "uuh . . . . . ."


def train_from_buffer(train_buffer) -> None:
    LOGGER.info(f"Committing {len(message_buffer)} chats to markov.")
    for msg in message_buffer: markov_bot.train(msg, 3)
    message_buffer.clear()


def train_guard(message, userid, is_live) -> None: # rename maybe?
    global time_last_commit
    if message[0] in config.SPECIAL_CHARS: return
    if len(message.strip().split()) < 2: return # if word is 1 emote spam, don't train
    if userid in cfg["dont_trainlist"]: return
    if not is_live: message_buffer.append(message)

    if (datetime.now() - time_last_commit).total_seconds() > 600.0: # 10 mins
        if is_live: message_buffer.clear()
        else:
            time_last_commit = datetime.now()
            train_from_buffer(message_buffer)


class Markov(commands.Component):
    def __init__(self, bot: Bot) -> None:
        self.bot = bot
        self.markov_bot = markov_bot
    
    @commands.Component.guard()
    def guard(self, ctx: commands.Context) -> bool:
        if ctx.chatter.id in cfg["banned_users"]:
            raise NotWhitelistedError # make this fleshed out, also fix notmoderatorerror
        return True


    async def event_ready(self):
        self.markov_bot.load("./brain")


    @commands.Component.listener()
    async def event_message(self, payload: twitchio.ChatMessage) -> None:
        train_guard(payload.text, payload.chatter.id, is_live=False) # this is less bloat (for now)
        # TODO: replace False with real live checking


    @commands.command()
    async def uh(self, ctx: commands.Context, *, args: str = "") -> None:
        import parser
        parsed = parser.parse_flags(args, ["-w", "-r", "-i", "-f"], ["-c", "-d", "-e"])
        seed, flags, flags_input = parsed[0], parsed[1], parsed[2]

        weighted = flags["-w"]
        max_words = flags_input["-c"]
        reverse = flags["-r"]
        infix = flags["-i"]
        force = flags["-f"]
        damping = flags_input["-d"]
        entropy = flags_input["-e"]

        # bad flag handling, if malformed  or not set, use default (-1 is default in parser.py)
        if isclose(max_words, -1, abs_tol=0.01): max_words = 45
        if isclose(damping, -1, abs_tol=0.01): damping = cfg["default_damping"]
        if isclose(entropy, -1, abs_tol=0.01): entropy = cfg["default_entropy"]

        if seed:
            # Order: seed, o, w, c, r, infix, f, damping, context_entropy
            result = self.markov_bot.generate_seeded(
                seed, 2, weighted, int(max_words), reverse, infix,
                force, damping, entropy
            ) or "0 gen seeded did not work"
        else:
            # Order: o, w, c, f, damping, context_entropy
            result = self.markov_bot.generate(
                2, weighted, int(max_words),
                force, damping, entropy
            ) or "0 gen did not work"

        for b in cfg["blocked_words"]:
            result = re.sub(re.escape(b), "", result, flags=re.IGNORECASE)

        result = moderate_spam(" ".join(result.split()))

        msgs = textwrap.wrap(result, width=300, break_long_words=True) or ["0 wrap failed"]
        await ctx.reply(msgs[0])
        for m in msgs[1:]:
            await asyncio.sleep(1)
            await ctx.reply(m)