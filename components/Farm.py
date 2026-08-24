import twitchio
from twitchio.ext import commands
from datetime import datetime, time, timedelta
import re, json, random, asyncio
from pathlib import Path
import config
from config import LOGGER



# shoehorn potat commands, clean up? move to seperate class probably

LOG_PATH = Path("potato_log.jsonl")

HARVEST_PATTERN = re.compile(
    r"\[(?P<sign>[+-])(?P<gained>[\d,]+) ⇒ (?P<total>-?[\d,]+)\]"
)

POTAT_COMMANDS = ["#p", "#potat", "#tater", "#potater", "#papa"]

def parse_harvest(text: str) -> dict | None:
    match = HARVEST_PATTERN.search(text)
    if not match:
        return None

    sign = -1 if match.group("sign") == "-" else 1
    gained = sign * int(match.group("gained").replace(",", ""))
    total = int(match.group("total").replace(",", ""))

    return {
        "gained": gained,
        "total": total,
    }

def log_harvest(harvest: dict, log_path: Path = LOG_PATH) -> None:
    entry = {
        "timestamp": datetime.now().isoformat(),
        **harvest,
    }
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def read_potat_stats(log_path: Path = LOG_PATH) -> dict:
    total_gained = 0
    harvest_count = 0

    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            date = datetime.fromisoformat(entry["timestamp"])
            if date < (datetime.now() - timedelta(hours=24)): continue
            total_gained += entry["gained"]
            harvest_count += 1

    average = total_gained / harvest_count if harvest_count else 0

    return {
        "total_gained": total_gained,
        "harvest_count": harvest_count,
        "average": average,
    }



class Farm(commands.Component):
    def __init__(self, bot: Bot) -> None:
        self.bot = bot
        self.last_potato: datetime = datetime.now()
        self.last_potato_msg_processed: datetime = datetime.min
        self.awaiting_potat_reply: bool = False
    
    TIME_PATTERN = re.compile(
        r"⏰\s*(?:(?P<hours>\d+)h\s*)?(?:(?P<minutes>\d+)m\s*)?(?:and\s*)?(?:(?P<seconds>\d+)s)?"
    )

    def parse_ready_time(self, text: str) -> datetime | None:
        """Returns the datetime when harvest will be ready, or None if no match."""
        match = self.TIME_PATTERN.search(text)
        if not match:
            return None

        hours = int(match.group("hours") or 0)
        minutes = int(match.group("minutes") or 0)
        seconds = int(match.group("seconds") or 0)

        if hours == 0 and minutes == 0 and seconds == 0:
            return None

        delta = timedelta(hours=hours, minutes=minutes, seconds=seconds)
        return datetime.now() + delta


    async def component_load(self) -> None:
        asyncio.create_task(self.potato_watcher())


    @commands.Component.listener()
    async def event_message(self, payload: twitchio.ChatMessage) -> None:
        if payload.chatter.id == "865895441":  # potatbotat
            if payload.reply is None or payload.reply.parent_user.id != self.bot.bot_id: return
            ready_time = self.parse_ready_time(payload.text)
            if ready_time is not None:
                self.last_potato = ready_time

            if self.awaiting_potat_reply: # and payload.reply is not None and payload.reply.parent_user.id == self.bot.bot_id:
                harvest = parse_harvest(payload.text)
                if harvest is not None:
                    log_harvest(harvest)
                    self.awaiting_potat_reply = False


    async def potato_watcher(self):
        LOGGER.info("dispatched potat watcher")
        while True:
            try:
                await asyncio.sleep(5)
                now = datetime.now()
                if not self.awaiting_potat_reply and now >= self.last_potato:
                    channel = self.bot.create_partialuser(user_id="974273622")
                    await channel.send_message(sender=self.bot.bot_id, message=random.choice(POTAT_COMMANDS))
                    self.awaiting_potat_reply = True
            except Exception as e:
                print(f"potato_watcher error: {e}")
