# Scroll The Scroll
Project for automatic scrolling based on real-time eye tracking.
### Scroll The Scroll Project
The Scroll The Scroll project aims to improve the experience of reading PDF documents on a laptop. By estimating when the reader has read a line by observing the movement of the pupil the program can scroll accordingly. Thus, the reader can maintain focus on the upper portion of the screen therefore reducing strain and making reading on a laptop an easier process.

The Scroll The Scroll project is open-source and completely free to use.

### About
The *scrollthescroll* package is purely python, and can be used to run the current prototype or can be used to analysis the performance of the prototype.

### Starting
Running the scrollthescroll prototype:
```python
from scrollthescroll import Prototype
p = prototype()
p.run("c:\\users\\user\\apdf.pdf",10)
```
"c:\\users\\user\\apdf.pdf": path to PDF doc for reading.
10: page number to begin reading from.

analysing past runs:
```python
from scrollthescroll import Prototype
p = prototype
p.display_run(0)
```

### Installation
```bash
pip3 install scrollthescroll
```

### Feedback
The Scroll The Scroll prject is still very much in its infancy, any feedback would be greatly appreciated. Email Theo.elliott.brown@gmail.com for any inquires.