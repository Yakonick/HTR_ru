# HTR_ru
### Handwritten Text Recognition model for russian 19th century text

## Installation

1. The zip-archive with the code is available with the [_link_](https://drive.google.com/file/d/1xiWQzlt4uU5T7F5HZrUt5eeJFwnkUdrI/view?usp=sharing).
Download it and unpack into your `PATH`.
2. Open terminal and go to your `PATH`:
```bash
cd /PATH/HTR_ru
```
3. Create python virtual environment and activate it:
```bash
python -m venv env
```
  * On Windows, run:
```bash
env\Scripts\activate.bat
```
  * On Unix or MacOS, run:
```bash
source env/bin/activate
```
4. Install requirements and go to `model` directory:
```bash
pip install -r requirements.txt
cd ./model
```
---
## How to use
Once you've done with all preparation, 
you can run the program typing in the command:
```bash
python htr.py --input INPUT_PATH --output OUTPUT_PATH
```
where `INPUT_PATH` - the path to the image which you want to translate
and `OUTPUT_PATH` - the path with the _*.txt_ file where you want to save the translation.

### For example:
```bash
python htr.py --input ../demo/demo1.jpg --output ../translate.txt
```
Also, you have 3 demo pictures in `PATH/HTR_ru/demo` directory to test the program.

## Finish
Once you've done with your work with the code, type
```bash
deactivate
```
to deactivate python virtual environment.
