My beginner nvim/setup script notes

## 0) Install & Verify

```powershell
# Install (pick one)
winget install Neovim.Neovim

# Verify
nvim --version     # expect v0.10+ 
```

## 1) Paths (Windows)

```vim
:echo stdpath('config')  " -> C:\Users\<you>\AppData\Local\nvim
:echo stdpath('data')    " -> C:\Users\<you>\AppData\Local\nvim-data
:echo $NVIM_APPNAME      " (empty = default 'nvim')
```

## 2) Create config

```powershell
mkdir $env:LOCALAPPDATA\nvim -Force
ni $env:LOCALAPPDATA\nvim\init.lua -Force
notepad $env:LOCALAPPDATA\nvim\init.lua
```

## 3) Minimal `init.lua` (paste into the file)

```lua
-- Leader & UI
vim.g.mapleader = " "
vim.opt.number = true
vim.opt.mouse = "a"
vim.o.guifont = "Consolas:h14"   -- change to a font you have

-- Clipboard: make y/p use system clipboard
vim.opt.clipboard = "unnamedplus"
```

## 4) Edit/Reload (without relying on $MYVIMRC)

```vim
:execute 'edit ' . stdpath('config') . '/init.lua'   " open config
:execute 'luafile ' . stdpath('config') . '/init.lua' " reload config
```

> After a restart (once config is loaded):

```vim
:echo $MYVIMRC        " shows full path to init.lua
:source $MYVIMRC      " reload via $MYVIMRC
```

## 5) Clipboard (system ↔ Neovim)

```vim
"+y          " yank to system clipboard
"+yy         " yank line to system clipboard
"+p / "+P    " paste from system clipboard (normal mode)
<C-r>+       " paste from system clipboard (insert mode)
:checkhealth " ensure clipboard: OK
```

## 6) Sanity / Diagnostics

```vim
:scriptnames                                   " list loaded scripts (should include init.lua)
:echo filereadable(stdpath('config').'/init.lua') " 1 = found
```

## 7) Neovide Tips

* Neovide uses your Neovim config (no separate `AppData\Local\neovide` needed).
* Launch & troubleshoot:

```powershell
neovide
neovide --log          # write a log next to the exe
neovide --opengl       # try OpenGL renderer
neovide --no-multigrid # bypass multigrid rendering
```

## 8) Common Issues (fast fixes)

* **E212: can’t open file for writing** → create the folder first, or check Windows “Controlled Folder Access”.
* `$MYVIMRC` empty in-session → file was created *after* launch; use the commands in §4 or restart Neovim.
* Clipboard not working → update Neovim (Win builds ship with clipboard), see §5.

---

