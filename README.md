# Scroll The Scroll
Automatic scrolling based on real time eye tracking with python
### Scroll The Scroll Project
*Scrollthescroll* is a python package for automatic scrolling based on real time eye tracking using a webcam. By tracking the movement of the users pupils using a webcam it can estimate when the user has read a line and can scroll accordingly. *Scrollthescroll* allows a user to maintain focus on a fixed quadrant of the screen, making reading on devices easier.

### About
*Scrollthescroll* is currently in its prototype stages and works only with local PDF files.
*Scrollthescroll* is currently only in working condition for Windows OS, however new changes are coming soon.
### Usage
Install the PyPI package:
```bash
pip install scrollthescroll
```
or clone the repository (no installation required, source files are sufficient):
```bash
git clone https://github.com/Theo-eBrown/scrollthescroll
```
or [download and extract the zip](https://github.com/Theo-eBrown/scrollthescroll/archive/refs/heads/master.zip "download and extract the zip").
### Example
Running the scrollthescroll prototype:
```python
from scrollthescroll import Prototype

p = Prototype()

p.run()
```
### ChangeLog

#### Version 0.0.9:
- Replaced pdf. reader functions with ScreenFunctions.
- Improved scrolling function.
- Removed bugs.
- Added more bugs.