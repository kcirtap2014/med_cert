# Project: Text recognition
copyright P. Lee, J. Deraze

This project applies pytesseract to extract text from documents, and identifying necessary keywords from the text.
## Approach:
- Part 1: pytesseract 
- Part 2: segmentation + CRNN

## Files:
- main.py: process certificates using pytesseract
- evaluation.py: evaluate the performance after part 1
- segmentation.py: segment remaining certificates that cannot be correctly processed using pytesseract
- helper_functions.py: contain functions that support other .py files
