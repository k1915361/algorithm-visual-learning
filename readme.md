## Algorithm Visual Learning

[![CI](https://github.com/k1915361/algorithm-visual-learning/actions/workflows/ci.yml/badge.svg)](https://github.com/k1915361/algorithm-visual-learning/actions/workflows/ci.yml)

Visual Algorithms Playground — learn by seeing & tweaking.

Tiny Generative Models, Big Insight — toy LLM/image/audio → real skills.

Gamified Benchmarks — compare · tweak · learn with live metrics.

Hands-On Mini-Lab — explore architectures; build intuition fast.

## Project Setup

For **stable & reliable Python environment** for long-term dev (especially for math-heavy + visualization projects), here’s a Windows CMD command sequence that will:

1. Install a well-supported Python version (LTS-like stability).
2. Create a clean virtual environment so projects don’t break when dependencies update.
3. Ensure packages compile correctly for math + viz.

---

## **Step 1 – Install Python (Recommended: 3.10.x LTS-like stability)**

Python **3.10** is very stable and widely supported in scientific libraries.
If not installed yet, download from:
[https://www.python.org/downloads/release/python-31012/](https://www.python.org/downloads/release/python-31012/)
(make sure to check “Add Python to PATH” during install)

Step 1.2. Microsoft Visual C++ Redistributable for Visual Studio 2015–2022 (x64)
X64	https://aka.ms/vs/17/release/vc_redist.x64.exe

Step 1.3. ffmpeg for audio/video processing

```cmd
winget install ffmpeg
```

---

## **Step 2 – Create a virtual environment (CMD)**

```cmd
:: Ensure it is python 3.10
python -V

:: Create project folder
cd C:\Users\eugen\Documents\Tiniest Language Model\algorithm_visual_learning

:: Create virtual environment named "venv10"
python -m venv venv310

:: Activate the environment
venv310\Scripts\activate

:: Run the app
python app.py
```

```sh
# activate the environment
source ./venv310/Scripts/activate

python app.py
```

---

## **Step 3 – Upgrade pip & install core packages**

```cmd
python -m pip install --upgrade pip
pip install numpy matplotlib manim pillow	
```

* **numpy** → math engine
* **matplotlib** → simple animations & plots
* **manim** → polished algorithm videos
* **pillow** → image handling

---

## **Step 4 – Verify installation**

```cmd
python -c "import sys, numpy, matplotlib, manim; print(sys.version, numpy.__version__, matplotlib.__version__)"
```

If this prints versions without errors, you’re good.

---

💡 **Tip:**
For rock-solid reproducibility, you can **freeze** package versions:

```cmd
pip freeze > requirements.txt
```

And later restore exactly:

```cmd
pip install -r requirements.txt
```

---

a **single `.bat` file** so you just double-click it and get this whole environment ready in one go.
Do you want me to prepare that?


```cmd
pip list --not-required
pip list --not-required --format=freeze > requirements.txt

```


## **Auto-run on file save (Windows)**

**watchdog** (simple):

```cmd
pip install watchdog
watchmedo shell-command --patterns="*.py" --recursive --command="cmd /c cls & venv\Scripts\python path\to\script.py" .
**PowerShell FileSystemWatcher** (no extra deps):
```

```powershell
$w = New-Object IO.FileSystemWatcher -Property @{ Path="."; Filter="*.py"; IncludeSubdirectories=$true; EnableRaisingEvents=$true }
Register-ObjectEvent $w Changed -Action { cls; & .\venv\Scripts\python.exe .\path\to\script.py } > $null
**VS Code**: add a “Run on Save” task/extension that executes your script. Good when you’re already editing there.
```