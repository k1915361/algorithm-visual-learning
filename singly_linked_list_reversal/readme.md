
```cmd
cd \algorithm_visual_learning

:: Activate the environment
venv310\Scripts\activate

:: Run the app
python app.py

:: Important, come to this path 
cd "singly_linked_list_reversal"
```

The -pql means Preview, Quick, Low quality (854×480) (15 fps).
You can change this to:

-pqh → Preview, Quick, High (1920×1080) 

-pqm → Preview, Quick, Medium (1280x720) (30 fps)

-pqh --fps 60 → 1080p at 60 fps

Custom resolution:

`-r 2560,1440`

`-r 360,270`

`-r 320,240`

```cmd
cd "singly linked list reversal" 

manim -pql _2_manim.py TextAnimationDemo

manim -pql _2_manim.py LinkedListReverseScene

manim -pqh --fps 60 -pql _2_manim.py LinkedListReverseScene

python profile_runner.py
```


