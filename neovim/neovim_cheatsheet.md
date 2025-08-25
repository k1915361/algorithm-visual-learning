# Neovim Keybinding Cheatsheet (Windows-friendly)

## Modes

* `Esc` — Normal mode
* `i` / `a` — Insert before / after cursor
* `I` / `A` — Insert at line start / end
* `o` / `O` — New line below / above and insert
* `v` / `V` / `Ctrl+v` — Visual (char/line/block)

---

## Save / Quit

* `:w` — Save file
* `:q` / `:q!` — Quit / force quit
* `:wq` / `:x` / `ZZ` — Save & quit
* `:wa` / `:qa` — Save all / quit all

---

## Cursor Basics

* `h j k l` — Left / down / up / right
* `w` / `b` / `e` — Next word / prev word / end of word (`W,B,E` = WORD)
* `0` / `^` / `$` — Start / first nonblank / end of line
* `gg` / `G` — Top / bottom of file
* `H` / `M` / `L` — Top / middle / bottom of screen
* `Ctrl+u` / `Ctrl+d` — Half-page up / down
* `Ctrl+b` / `Ctrl+f` — Page up / down

---

## Edit (characters, words, lines)

* `x` / `X` — Delete char under / before cursor
* `r{char}` — Replace one char
* `s` / `S` — Substitute char / whole line (enter Insert)
* `cw` / `c$` — Change to end of word / line
* `ciw` / `daw` — Change inner word / delete a word (text objects)
* `dd` / `D` / `C` — Delete line / delete to EOL / change to EOL
* `J` — Join with next line
* `u` / `Ctrl+r` — Undo / redo
* `.` — Repeat last change
* `>>` / `<<` — Indent / outdent line
* `==` — Reindent line
* `~` / `g~w` — Toggle case char / word
* `guw` / `gUw` — Lowercase / uppercase word

> Counts work with most commands: `3dd`, `5w`, `2>>`, etc.

---

## Yank / Paste (Vim registers)

* `yy` / `Y` — Yank line
* `p` / `P` — Paste after / before cursor
* In **Visual**: select, then `y` / `d` / `>` / `<` / `=`

### System clipboard (Windows)

* `"+y` / `"+yy` — Yank to OS clipboard
* `"+p` / `"+P` — Paste from OS clipboard
* In Insert: `Ctrl+r +` — Paste from clipboard
* **Optional**: always use system clipboard

  ```vim
  :set clipboard=unnamedplus
  ```

---

## Search / Replace

* `/text` / `?text` — Search forward / backward
* `n` / `N` — Next / previous match
* `*` / `#` — Search word under cursor forward / backward
* Replace (whole file example):

  ```
  :%s/find/replace/g
  :%s/find/replace/gc   " confirm each
  ```

---

## Files (open / write)

* `:e {path}` — Edit/open file
* `:w {path}` — Write to new file
* `:browse e` — Open file picker (basic)
* `:Ex` — Netrw file explorer (then use Enter to open, `-` up dir)

---

## Buffers (multiple files open)

* `:ls` / `:buffers` — List buffers
* `:bnext` / `:bprev` — Next / previous buffer (`:bn` / `:bp`)
* `:buffer {num}` — Jump to buffer by number
* `:bd` — Delete (close) current buffer
* `Ctrl+^` — Toggle last two files (alternate file)

---

## Tabs (tab pages)

* `:tabnew` / `:tabclose` — New / close tab
* `:tabnext` / `:tabprev` — Next / previous tab (`:tabn` / `:tabp`)
* `gt` / `gT` — Next / previous tab

---

## Windows / Splits

* `:split` / `:vsplit` — Horizontal / vertical split
* `Ctrl+w s` / `Ctrl+w v` — Same via shortcuts
* Move focus: `Ctrl+w h/j/k/l`
* Equalize sizes: `Ctrl+w =`
* Max height / width: `Ctrl+w _` / `Ctrl+w |`
* Close split: `:q` (in that window) or `Ctrl+w c`

---

## Marks (quick jumps)

* `ma` — Set mark `a`
* `` `a `` / `'a` — Jump to mark (exact / line)
* ` ` \`\` — Jump back to last position

---

## Help

* `:help {topic}` — e.g., `:help motion.txt`, `:help visual.txt`, `:help usr_02.txt`

---
