"""
Minimal anonymous (read-only) Twitch IRC chat reader.

No OAuth token, no authorization, no broadcaster action required.
Uses the "justinfan" anonymous login convention Twitch provides for
read-only IRC access.
"""

import asyncio
import logging
import random
from typing import Callable, Awaitable

LOGGER = logging.getLogger("IRCReader")

TWITCH_IRC_HOST = "irc.chat.twitch.tv"
TWITCH_IRC_PORT = 6667  # Plaintext. Use 6697 for SSL if you want to wrap with ssl.create_default_context()


class AnonymousIRCReader:
    """A minimal read-only Twitch IRC client.

    Usage:
        reader = AnonymousIRCReader("some_channel", on_message=my_callback)
        await reader.start()
    """

    def __init__(
        self,
        channel: str,
        on_message: Callable[[str, str, str], Awaitable[None]],
        *,
        reconnect_delay: float = 5.0,
    ) -> None:
        self.channel = channel.lower().lstrip("#")
        self.on_message = on_message
        self.reconnect_delay = reconnect_delay
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._running = False

    async def start(self) -> None:
        """Connect and read forever, reconnecting automatically on failure."""
        self._running = True
        while self._running:
            try:
                await self._connect_and_listen()
            except (ConnectionError, asyncio.IncompleteReadError, OSError) as e:
                LOGGER.warning("IRC connection lost for #%s: %s. Reconnecting in %.1fs...",
                                self.channel, e, self.reconnect_delay)
                await asyncio.sleep(self.reconnect_delay)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                LOGGER.error("Unexpected IRC error for #%s: %s", self.channel, e)
                await asyncio.sleep(self.reconnect_delay)

    def stop(self) -> None:
        self._running = False
        if self._writer:
            self._writer.close()

    async def _connect_and_listen(self) -> None:
        nick = f"justinfan{random.randint(10000, 99999)}"

        LOGGER.info("Connecting to Twitch IRC as %s for #%s...", nick, self.channel)
        self._reader, self._writer = await asyncio.open_connection(TWITCH_IRC_HOST, TWITCH_IRC_PORT)

        # Anonymous login: PASS can be anything (even blank-ish), NICK must be justinfanXXXXX
        self._send(f"PASS oauth:anonymous")
        self._send(f"NICK {nick}")
        self._send(f"JOIN #{self.channel}")

        LOGGER.info("Joined #%s anonymously (read-only).", self.channel)

        while True:
            line = await self._reader.readline()
            if not line:
                raise ConnectionError("Connection closed by server")

            decoded = line.decode("utf-8", errors="ignore").strip()
            if not decoded:
                continue

            # Respond to PING to stay alive
            if decoded.startswith("PING"):
                self._send(decoded.replace("PING", "PONG", 1))
                continue

            self._parse_line(decoded)

    def _send(self, message: str) -> None:
        if self._writer:
            self._writer.write(f"{message}\r\n".encode("utf-8"))

    def _parse_line(self, line: str) -> None:
        # We only care about PRIVMSG (actual chat messages)
        # Example raw line:
        # :nick!nick@nick.tmi.twitch.tv PRIVMSG #channel :the message text
        if "PRIVMSG" not in line:
            return

        try:
            prefix, _, rest = line.partition(" PRIVMSG ")
            username = prefix.split("!")[0].lstrip(":")

            channel_part, _, message_part = rest.partition(" :")
            channel = channel_part.lstrip("#")
            message = message_part

            asyncio.create_task(self.on_message(channel, username, message))
        except Exception as e:
            LOGGER.debug("Failed to parse IRC line: %r (%s)", line, e)


# --- Example standalone usage ---
async def example_handler(channel: str, username: str, message: str) -> None:
    print(f"[{channel}] {username}: {message}")


async def main() -> None:
    logging.basicConfig(level=logging.INFO)
    reader = AnonymousIRCReader("vedal987", on_message=example_handler)
    await reader.start()


if __name__ == "__main__":
    asyncio.run(main())