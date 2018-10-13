Project: Medical certificates validation
P. Lee, J. Deraze

This project applies pytesseract to extract text from a medical certificate image, and identifying necessary keywords from the text, such as first name, surname, validity date and discipline with the word 'compétition'.

Approach:
Part 1: pytesseract (>50% certificates can be correctly identified)
Part 2: segmentation + CRNN

Files:
------
- main.py: process certificates using pytesseract
- evaluation.py: evaluate the performance after part 1
- segmentation.py: segment remaining certificates that cannot be correctly processed using pytesseract
- helper_functions.py: contain functions that support other .py files
