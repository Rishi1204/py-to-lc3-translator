# Python to LC-3 Translator

This project is a Python-to-LC3 translator that parses Python code and generates LC-3 assembly. It supports function definitions, arithmetic expressions, conditionals, loops, and recursive calls.

## Motivation

Python was chosen as the input language for this translator because it closely resembles pseudocode, making it highly accessible for students. By translating Python constructs like loops, conditionals, and functions into LC-3 assembly, this tool helps learners understand how high-level programming paradigms map to low-level machine instructions.

## Project Structure

```
├── py/                     # Folder containing Python (.py) files to be translated
├── asm/                    # Output folder for generated LC-3 (.asm) files
├── constants.py            # Contains reusable subroutine definitions (as strings)
├── function_translator.py  # Logic for translating individual Python functions to LC-3
├── lc3_translator.py       # Main translation engine that orchestrates translation
├── main.py                 # Entry point script to run translation from the command line
```

## How to Use

### 1. Translate **all** Python files in the `py/` directory:
```bash
python3 main.py
```

This will:
- Parse all `.py` files in the `py/` directory
- Translate them to `.asm` files
- Save the output to the `asm/` directory

---

### 2. Translate **specific** files in the `py/` directory:
```bash
python3 main.py file1.py file2.py
```

Replace `file1.py`, `file2.py`, etc. with the names of the files you want to translate.  
Each translated `.asm` file will be saved with the same name in the `asm/` directory.

---

## Requirements

- Python 3.10 or later (for match statements and modern type hint syntax such as `int | str`)
- No external packages required (uses the built-in `ast` module)

---

## Notes

- Only functions in the `py/` folder will be translated.
- The `main` function, if present, will automatically initialize the stack pointer and insert a `HALT` instruction.
- It is recommended to define a `main` function, as that is where execution begins. All other functions are translated as subroutines.
- Support for arrays, strings, multiple conditionals (i.e. `if cond_1 and cond_2`) has not been implemented yet.
- Custom LC-3 subroutines used across translations can be defined in `constants.py`.
- Several example Python files are included in the `py/` directory to showcase the capabilities of the translator.

---

## Example

```bash
python3 main.py fib.py math_utils.py
```

This command reads and translates `fib.py` and `math_utils.py`, then outputs `fib.asm` and `math_utils.asm` in the `asm/` folder.

---
