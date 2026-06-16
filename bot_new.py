import asyncio
import logging
import random
from typing import TYPE_CHECKING

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
# --- END OF CFG ---


LOGGER: logging.Logger = logging.getLogger("Bot")




class Bot(commands.Autobot):
    def __init__(self, *, token_database: asqlite.Pool, subs: list[eventsub.SubscriptionPayload]) -> None:
        self.token_database = token_database

        super().__init__(
            client_id=cfg["client_id"],
            client_secret=cfg["client_secret"],
            bot_id=cfg["bot_id"],
            owner_id=bot_id, # boilerplate
            prefix=list(SPECIAL_CHARS),
            subscriptions=subs,
            force_subscribe=True,
        )


    async def setup_hook(self) -> None: # OOP hell (wtf is this boilerplate) edit: I take it back
        await self.add_component(Moderation(self))


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
class Moderation(commands.Component):
    def __init__(self, bot: Bot) -> None:
        self.bot = bot
    
    