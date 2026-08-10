from twitchio.ext import commands
from config import cfg, save_cfg


async def get_user(bot, ctx, username):
    if username is None or username == "":
        await ctx.send("must specify a chatter. uuh")
        return None
    username = username.lstrip("@")
    try:
        return await bot.fetch_user(login=username)
    except Exception as e:
        await ctx.send(f"uuh who is ' {e} ' ??")
        return None


class Moderation(commands.Component):
    def __init__(self, bot: Bot) -> None:
        self.bot = bot


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
    

    @commands.command()
    async def unban(self, ctx: commands.Context, *, args: str = "") -> None:
        import parser
        parsed = parser.parse_flags(args, ["--clear", "-c"])
        username, flags = parsed[0], parsed[1]
        
        if flags["--clear"] or flags["-c"]:
            cfg["banned_users"].clear()
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


    @commands.command()
    async def mod(self, ctx: commands.Context, *, args: str = "") -> None:
        import parser
        parsed = parser.parse_flags(args, ["-a", "-r", "-c"])
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
    async def killuuh1(self, ctx: commands.Context) -> None:
        return 1 / 0