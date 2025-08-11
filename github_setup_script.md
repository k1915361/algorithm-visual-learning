## **2. Install via Commands (if you prefer terminal)**

**Windows (via Winget)**

```powershell
winget install --id Git.Git -e --source winget
winget install --id GitHub.GitHubDesktop -e --source winget
```

**macOS (via Homebrew)**

```bash
brew install git
brew install --cask github
```

**Ubuntu/Debian Linux**

```bash
sudo apt update
sudo apt install git -y
# GitHub Desktop is not officially supported on Linux, but you can use the fork:
wget https://github.com/shiftkey/desktop/releases/latest/download/GitHubDesktop-linux-amd64.deb
sudo apt install ./GitHubDesktop-linux-amd64.deb
```

---

## **3. First-Time Git Setup**

After installing Git, run these once in your terminal:

```bash
git config --global user.name "Your Name"
git config --global user.email "your@email.com"
```

Verify:

```bash
git --version
```

---
