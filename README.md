# markovChain

You must create a venv to compile.
These instructions are all for my own machine, and may differ for other machines. (Windows)
Create a venv:
[python flag] -m venv venv
(creates a venv called "venv")

Enter the venv:
venv\Scripts\activate.bat
(Only for cmd! for PS use .ps1 but then you need some extra config to allow PS to execute .ps1)
NOTE: use this for linux:
source venv/bin/activate

Compile (using setup.py):
[pip flag] install -e .
if not working, use:
[pip flag] install -e . --no-cache-dir --force-reinstall

then run bot.py:
[python flag] bot.py

bot_new.py is a migration from TwitchIO v2 to v3, it is still in progress. do not run that.

If the build still doesn't work, try deleting the .pyd files to diagnose (if there are any) or delete the build file, or check the venv python version. For most ease, make sure you only have 1 python installed on system so there aren't conflicts.

## NOTE: -e flag is REQUIRED!!!!

RUNNING bot_new.py (INDEV!!):
- get asqlite (pip install asqlite), get twitchio v3 (not v2, pip install twitchio works fine)
- run it (don't forget to build the C++ - doesn't matter rn because bot_new.py isn't connected to markov)

http://localhost:4343/oauth?scopes=user:read:chat%20user:write:chat%20user:bot&force_verify=true