from twitchio.ext import commands
from config import cfg, save_cfg, LOGGER


# extracted edge case handling because it was ass to type it every time
async def get_user(bot, ctx, username):
    if username is None or username == "":
        await ctx.send("must specify a chatter. uuh")
        return None
    username = username.lstrip("@")
    try:
        user = await bot.fetch_user(login=username)
        if user is None: await ctx.send(f"uuh who is ' {username} ' ??")
        return user
    except Exception as e:
        await ctx.send(f"uuh . . {e}")
        return None


class Moderation(commands.Component):
    def __init__(self, bot: Bot) -> None:
        self.bot = bot


    # if not in master admin or admin list, nope
    @commands.Component.guard()
    def is_admin(self, ctx: commands.Context) -> bool:
        if ctx.chatter.id not in cfg["admin_list"] and ctx.chatter.id not in cfg["erm_list"]:
            raise NotModeratorError
        return True


    @commands.command()
    async def ban(self, ctx: commands.Context, username: str = None) -> None:
        user = await get_user(self.bot, ctx, username)
        if user is None: return
        if user.id not in cfg["banned_users"]:
            cfg["banned_users"].append(user.id)
            save_cfg()
            await ctx.send(f"uuh banned {user.mention} from bot")
        else:
            await ctx.send(f"uuh . . . {user.mention} is already banned WeirdDude")
    

    # uses flags because I want to clear all
    @commands.command()
    async def unban(self, ctx: commands.Context, *, args: str = "") -> None:
        import parser
        parsed = parser.parse_flags(args, ["--clear", "-c"], [])
        username, flags = parsed[0], parsed[1]
        
        if flags["--clear"] or flags["-c"]:
            cfg["banned_users"].clear()
            save_cfg()
            await ctx.send("unbanned all users uuh . . .")
            return

        # get user AFTER parsing flags because only flag without user will return early
        user = await get_user(self.bot, ctx, username)
        if user is None: return

        if user.id in cfg["banned_users"]:
            cfg["banned_users"].remove(user.id)
            save_cfg()
            await ctx.send(f"uuh unbanned {user.mention} from bot")
        else:
            await ctx.send(f"uuh {user.mention} is not banned")


    # not typing out 3 more cmds so it just compacts them into one
    @commands.command()
    async def mod(self, ctx: commands.Context, *, args: str = "") -> None:
        import parser
        parsed = parser.parse_flags(args, ["-a", "-r", "-c"], [])
        username, flags = parsed[0], parsed[1]

        if flags["-c"]:
            cfg["banned_users"].clear()
            save_cfg()
            await ctx.send("unmodded all users uuh . . .")
            return

        user = await get_user(self.bot, ctx, username)
        if user is None: return
        
        if flags["-a"]:
            if user.id in cfg["admin_list"]:
                await ctx.send(f"Awkward {user.mention} is already a mod . . .")
                return
            cfg["admin_list"].append(user.id)
            save_cfg()
            await ctx.send(f"added {user.mention} as mod Scared")
            return
        elif flags["-r"]:
            if not user.id in cfg["admin_list"]:
                await ctx.send(f"{user.mention} is not a mod eerm LurkingEyes")
                return
            cfg["admin_list"].remove(user.id)
            save_cfg()
            await ctx.send(f"removed {user.mention} as mod eerm")
            return
        else:
            await ctx.send("uuh usage: username -c[lear] -r[emove] -a[dd]")


    @commands.command()
    async def config(self, ctx: commands.Context, *, args: str = "") -> None:
        import parser
        parsed = parser.parse_flags(args, ["-h", "-train", "-cd", "-d", "-e"])
        value, flags = parsed[0], parsed[1]
        if sum(flags.values()) > 1:
            await ctx.send(f"uuh Tssk use one flag at a time")
        elif sum(flags.values()) < 1:
            if value == "": await ctx.send(f"0 training: {cfg['training']}, cooldown: {cfg['cooldown']}")
            else: await ctx.send(f"eerm . . use correct flag, -h for help")
        else:
            if flags["-h"]:
                await ctx.send(f"eerm flags: -h, -train, -cd [float], -d [float], -e [float]")
                return
            if flags["-train"]:
                cfg["training"] = not cfg["training"]
                await ctx.send(f"0 training is now {cfg['training']}")

            elif value is None or value == "":
                await ctx.send("0 provide a value")
                return

            elif flags["-cd"]:
                try:
                    cfg["cooldown"] = max(0.0, float(value))
                except ValueError:
                    await ctx.send("0 value is malformed, expected: float")
                    return

            elif flags["-d"]:
                try:
                    cfg["default_damping"] = max(0.0, min(1.0, float(value)))
                except ValueError:
                    await ctx.send("0 value is malformed, expected: float")
                    return

            elif flags["-e"]:
                try:
                    cfg["default_entropy"] = max(0.0, min(1.0, float(value)))
                except ValueError:
                    await ctx.send("0 value is malformed, expected: float")
                    return


    # nukes the bot
    @commands.command()
    async def killuuh(self, ctx: commands.Context) -> None:
        LOGGER.warning("killuuh called. Shutting down.")
        await ctx.send("kuh 👍")
        await self.bot.close()