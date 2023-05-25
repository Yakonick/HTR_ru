# HTR_ru
### Handwritten Text Recognition model for russian 19th century text

## Installation

1. The zip-archive with the code is available with the [_link_](https://drive.google.com/drive/folders/1vwqzXkQNhiHw1m-z6kxH4Q0JP1eZxVcD?usp=sharing).
Download it and unpack into your `PATH`.
2. Open terminal and go to your `PATH`:
```bash
cd /PATH/HTR_ru
```
3. Activate python virtual environment and go to `model` directory:
```bash
source ./env/bin/activate
cd ./model
```
---
## How to use
Once you've done with all preparation, 
you can run the program typing in the command:
```bash
python3 htr.py --input INPUT_PATH --output OUTPUT_PATH
```
where `INPUT_PATH` - the path to the image which you want to translate
and `OUTPUT_PATH` - the path with the _*.txt_ file where you want to save the translation.

### For example:
```bash
python3 htr.py --input ../demo/demo1.jpg --output ../translate.txt
```
Also, you have 3 demo pictures in `PATH/HTR_ru/demo` directory to test the program.

## Finish
Once you've done with your work with the code, type
```bash
deactivate
```
to deactivate python virtual environment.
