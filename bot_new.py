import asyncio
import logging
import random
from typing import TYPE_CHECKING
from irc_reader import AnonymousIRCReader

import asqlite

import twitchio
from twitchio import eventsub
from twitchio.ext import commands

import config
from config import LOGGER
from components.Moderation import Moderation
import components.Markov
from components.Markov import Markov

import parser


if TYPE_CHECKING:
    import sqlite3


from datetime import datetime, time
import asyncio, re, os, json, signal, sys, shutil, textwrap, aiohttp
from collections import Counter
from functools import partial


# CFG moved to its own file, logger moved there too


def is_time_between(begin_time, end_time, check_time=None):
    # If check time is not given, default to current UTC time
    check_time = check_time or datetime.now().time()
    if begin_time < end_time:
        return check_time >= begin_time and check_time <= end_time
    else: # crosses midnight
        return check_time >= begin_time or check_time <= end_time


class Bot(commands.AutoBot):
    def __init__(self, *, token_database: asqlite.Pool, subs: list[eventsub.SubscriptionPayload]) -> None:
        self.token_database = token_database
        self.is_live: dict[str, bool] = {}
        super().__init__(
            client_id=config.cfg["client_id"],
            client_secret=config.cfg["client_secret"],
            bot_id=config.cfg["bot_id"],
            owner_id=config.cfg["bot_id"], # boilerplate
            prefix=parser.build_prefixes(config.SPECIAL_CHARS),
            subscriptions=subs,
            force_subscribe=True,
        )


    @commands.Component.listener()
    async def event_message(self, payload: twitchio.ChatMessage) -> None:
        if payload.chatter.id.lower() == self.bot_id.lower():
            return
        payload.text = re.sub(r"\s+", " ", payload.text).strip()
        print(f"{payload.chatter.name}: {payload.text}")
        words = payload.text.split()
        # TODO: finish this
        #if "ass" in word.lower() for word in words: ctx
        await super().event_message(payload)


    # async def setup_hook(self) -> None: # OOP hell (wtf is this boilerplate) edit: I take it back
    #     await self.add_component(Moderation(self))
    async def setup_hook(self) -> None:
        component_list = [MainCmds(self), Moderation(self), Markov(self)]
        LOGGER.info("setup_hook is running!")
        for comp in component_list:
            try:
                await self.add_component(comp)
                LOGGER.info("Component added successfully!")
            except Exception as e:
                LOGGER.warning(f"Error adding component: {e}")
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


class MainCmds(commands.Component):
    def __init__(self, bot: Bot) -> None:
        self.bot = bot
    

    @commands.command()
    async def helpuuh(self, ctx: commands.Context) -> None:
        await ctx.reply("""
        uuh [seed, w, r, i, f, c1-75, d0-1, e0-1], bih, checkuuh, brainfiles.
        admin 0 : ban,, unban [-c], mod [-c, -a, -r], killuuh uuh""")
    

    # ping triggers =================================
    @commands.command(aliases=["charsy_nya", "charsy"])
    async def ping_charsy(self, ctx: commands.Context) -> None:
        await ctx.send("SchizoUuh charsyWTF")
    
    @commands.command(aliases=["RadiantLight", "radiantlight"])
    async def ping_radiant(self, ctx: commands.Context) -> None:
        await ctx.send("SchizoUuh radiantWTF")
    
    @commands.command(aliases=["NobleTrash38", "nobletrash38"])
    async def ping_noble(self, ctx: commands.Context) -> None:
        await ctx.send("SchizoUuh nobleWTF")
    
    @commands.command(aliases=["smuggaD", "smuggad"])
    async def ping_smugga(self, ctx: commands.Context) -> None:
        await ctx.send("SchizoUuh smuggaWTF")
    
    @commands.command(aliases=["ermugo1", "bih"])
    async def ping_ermugo(self, ctx: commands.Context) -> None:
        await ctx.send("SchizoUuh ermugoWTF")
    # =====================================


    @commands.command()
    async def time(self, ctx: commands.Context) -> None:
        st = config.cfg["start_time"]
        en = config.cfg["start_time"]
        now = datetime.now()
        time_str = str(now.hour) + ":" + str(now.minute)
        if is_time_between(time(st[0], st[1]), time(en[0], en[1])):
            await ctx.reply(f"kuh yea, time is {time_str}")
        else:
            await ctx.reply(f"uuh despair0 VoteNay , time is {time_str}")




# all this is witchcraft, just don't touch it, it just works...
async def setup_database(db: asqlite.Pool) -> tuple[list[tuple[str, str]], list[eventsub.SubscriptionPayload]]:
    query = """CREATE TABLE IF NOT EXISTS tokens(user_id TEXT PRIMARY KEY, token TEXT NOT NULL, refresh TEXT NOT NULL)"""
    async with db.acquire() as connection:
        await connection.execute(query)

        # fetch existing tokens
        rows: list[sqlite3.Row] = await connection.fetchall("""SELECT * FROM tokens""")

        tokens: list[tuple[str, str]] = []
        subs: list[eventsub.SubscriptionPayload] = []

        for row in rows:
            tokens.append((row["token"], row["refresh"]))

            if row["user_id"] == config.cfg["bot_id"]:
                continue
            
            subs.extend([eventsub.ChatMessageSubscription(broadcaster_user_id=row["user_id"], user_id=config.cfg["bot_id"])])

    return tokens, subs


async def get_user_ids(bot: Bot, usernames: list[str]) -> list[str]: # unused
    """Get a user's ID from their username"""
    users = await bot.fetch_users(logins=usernames)
    return [user.id for user in users]


async def poll_live_status(bot: Bot, channel_logins: list[str], interval: int = 60) -> None:
    while True:
        try:
            streams = await bot.fetch_streams(user_logins=channel_logins, token_for=bot.bot_id)
            currently_live = {s.user.name.lower() for s in streams}
            for name in channel_logins:
                bot.is_live[name.lower()] = name.lower() in currently_live
        except Exception as e:
            LOGGER.warning(f"Error polling stream status: {e}")
        await asyncio.sleep(interval)


async def handle_irc_message(bot: Bot, channel: str, username: str, message: str) -> None:
    print(f"[IRC #{channel}] {username}: {message}")
    user = await Bot.fetch_user(bot, login=username)
    if user is None: return
    channel_is_live = bot.is_live.get(channel.lower(), False)
    components.Markov.train_guard(message, user.id, channel_is_live) # little tunnel so the IRC can get into traindata without responding to commands and stuff


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
                
                irc_reader = AnonymousIRCReader("vedal987", on_message=partial(handle_irc_message, bot))
            
                #await bot.start(load_tokens=False)
                await asyncio.gather(
                    bot.start(load_tokens=False),
                    irc_reader.start(),
                    poll_live_status(bot, ["vedal987"]),
                )
    
    try:
        asyncio.run(runner())
    except KeyboardInterrupt:
        LOGGER.warning("Shutting down due to KeyboardInterrupt")
    finally:
        LOGGER.info("Run shutdown procedure")
        components.Markov.save_brain()

    


# call main
if __name__ == "__main__":
    main()