import sys
from lc3_translator import LC3Translator

def main():
    translator = LC3Translator()
    if len(sys.argv) == 1:
        # No arguments: translate all files in the py directory
        translator.translate_all()
    else:
        # Arguments provided: translate only the specified files
        print(sys.argv[1:])
        translator.translate_selected(sys.argv[1:])

if __name__ == "__main__":
    main()