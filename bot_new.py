import asyncio
import logging
import random
from typing import TYPE_CHECKING
from irc_reader import AnonymousIRCReader

import asqlite

import twitchio
from twitchio import eventsub
from twitchio.ext import commands


if TYPE_CHECKING:
    import sqlite3


from datetime import datetime
import time, asyncio, re, os, json, signal, sys, shutil, textwrap, aiohttp
from collections import Counter


# --- INITIAL SETUP ---
CONFIG_FILE = "config.json"
CHANNELS = ["ermugo2"]
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
# --- END OF CFG ---


LOGGER: logging.Logger = logging.getLogger("Bot")




class Bot(commands.AutoBot):
    def __init__(self, *, token_database: asqlite.Pool, subs: list[eventsub.SubscriptionPayload]) -> None:
        self.token_database = token_database

        super().__init__(
            client_id=cfg["client_id"],
            client_secret=cfg["client_secret"],
            bot_id=cfg["bot_id"],
            owner_id=cfg["bot_id"], # boilerplate
            prefix="!",#list(SPECIAL_CHARS),
            subscriptions=subs,
            force_subscribe=True,
        )


    # async def setup_hook(self) -> None: # OOP hell (wtf is this boilerplate) edit: I take it back
    #     await self.add_component(Moderation(self))
    async def setup_hook(self) -> None:
        print("setup_hook is running!")
        try:
            comp = MainCmds(self)
            print(f"Moderation created: {comp}")
            await self.add_component(comp)
            print("Component added successfully!")
        except Exception as e:
            print(f"Error adding component: {e}")
            import traceback
            traceback.print_exc()


    # executes when successful init
    async def event_oauth_authorized(self, payload: twitchio.authentication.UserTokenPayload) -> None:
        await self.add_token(payload.access_token, payload.refresh_token)

        if not payload.user_id:
            return
        
        subs: list[eventsub.SubscriptionPayload] = [
            eventsub.ChatMessageSubscription(broadcaster_user_id=payload.user_id, user_id=self.bot_id),
        ]
        
        resp: twitchio.MultiSubscribePayload = await self.multi_subscribe(subs)
        if resp.errors:
            LOGGER.warning("Failed to subscribe to: %r, for user: %s", resp.errors, payload.user_id)


    # called in system (event_token_authorized, here recursively, main)
    async def add_token(self, token: str, refresh: str) -> twitchio.authentication.ValidateTokenPayload:
        # extend to twitchio library
        resp: twitchio.authentication.ValidateTokenPayload = await super().add_token(token, refresh)

        # sqlite query
        query = """
        INSERT INTO tokens (user_id, token, refresh)
        VALUES (?, ?, ?)
        ON CONFLICT(user_id)
        DO UPDATE SET
            token = excluded.token,
            refresh = excluded.refresh;
        """

        # execute sqlite query safely
        async with self.token_database.acquire() as connection:
            await connection.execute(query, (resp.user_id, token, refresh))
        
        LOGGER.info("Added token to database for user: %s", resp.user_id)
        return resp
    

    # stub
    async def event_ready(self) -> None:
        LOGGER.info("Successfully logged in as: %s", self.bot_id)



# component for admin, whitelist, blacklist, traintime, other boilerplate. boring crap
class MainCmds(commands.Component):
    def __init__(self, bot: Bot) -> None:
        self.bot = bot

    @commands.Component.listener()
    async def event_message(self, payload: twitchio.ChatMessage) -> None:
        print(f"Message received: {payload.text}")
    
    @commands.command(aliases=["guh", "bih"])
    async def schizoUuh(self, ctx: commands.Context, username: str = None) -> None:
        print("called function")
        if username is None:
            username = ctx.chatter.name
        else:
            username = username.lstrip("@")

        try:
            user = await self.bot.fetch_user(username)
            await ctx.send(f"SchizoUuh {user.mention}")
        except Exception:
            await ctx.send(f"uuh who is {username} ???")


# all this is witchcraft, just don't touch it, it just works...
async def setup_database(db: asqlite.Pool) -> tuple[list[tuple[str, str]], list[eventsub.SubscriptionPayload]]:
    query = """CREATE TABLE IF NOT EXISTS tokens(user_id TEXT PRIMARY KEY, token TEXT NOT NULL, refresh TEXT NOT NULL)"""
    async with db.acquire() as connection:
        await connection.execute(query)

        # fetch existing tokens
        rows: list[sqlite3.Row] = await connection.fetchall("""SELECT * from tokens""")

        tokens: list[tuple[str, str]] = []
        subs: list[eventsub.SubscriptionPayload] = []

        for row in rows:
            tokens.append((row["token"], row["refresh"]))

            if row["user_id"] == cfg["bot_id"]:
                continue
            
            subs.extend([eventsub.ChatMessageSubscription(broadcaster_user_id=row["user_id"], user_id=BOT_ID)])

    return tokens, subs


async def get_user_ids(bot: Bot, usernames: list[str]) -> list[str]:
    """Get a user's ID from their username"""
    users = await bot.fetch_users(logins=usernames)
    return [user.id for user in users]

async def handle_irc_message(channel: str, username: str, message: str) -> None:
    #print(f"[IRC #{channel}] {username}: {message}")
    print(message)


# main entry point for bot
# set up logging here if needed
def main() -> None:
    twitchio.utils.setup_logging(level=logging.INFO)

    async def runner() -> None:
        async with asqlite.create_pool("tokens.db") as tdb:
            tokens, subs = await setup_database(tdb)
            #print(f"Tokens from DB: {tokens}")
            #print(f"Subs from DB: {subs}")

            async with Bot(token_database=tdb, subs=subs) as bot:
                for pair in tokens:
                    await bot.add_token(*pair)
                
                irc_reader = AnonymousIRCReader("vedal987", on_message=handle_irc_message)
            
                #await bot.start(load_tokens=False)
                await asyncio.gather(
                    bot.start(load_tokens=False),
                    irc_reader.start(),
                )
    
    try:
        asyncio.run(runner())
    except KeyboardInterrupt:
        LOGGER.warning("Shutting down due to KeyboardInterrupt")
    


# call main
if __name__ == "__main__":
    main()