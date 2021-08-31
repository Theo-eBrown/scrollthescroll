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
   from scrollthescroll import prototype
p = prototype()
p.run(<PDF_path>,<page_number>)
```
PDF_path = path to PDF for reading,
page_number = page number to open PDF on - note: PDF reader will not auto-open to page.

### Feedback
The Scroll The Scroll prject is still very much in its infancy, any feedback would be greatly appreciated. Email Theo.elliott.brown@gmail.com for any inquires.