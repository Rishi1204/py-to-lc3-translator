from ast import AST, FunctionDef, Assign, Name, Store, walk, parse, dump, unparse
from function_translator import FunctionTranslator
import os
from constants import ADDITIONAL_SUBROUTINES

class LC3Translator:
    def __init__(self) -> None:
        """
        Initializes an LC3Translator instance with:
        - `functions`: a list of FunctionTranslator objects
        - `num_functions`: (unused, possibly legacy)
        - `function_index`: a counter for assigning unique indices to functions
        """
        self.functions: list[FunctionTranslator] = []
        self.num_functions = 0
        self.function_index = 0

    def analyze_functions(self, tree: AST) -> list[FunctionTranslator]:
        """
        Parses an AST tree to extract all function definitions.

        For each function:
        - Extracts the function name, arguments, and local variables.
        - Constructs a FunctionTranslator object and appends it to `self.functions`.
        - Increments the `function_index` for unique function identification.

        Args:
            tree (ast.AST): The parsed Python AST.

        Returns:
            list[FunctionTranslator]: List of translated function representations.
        """

        for node in walk(tree):
            if isinstance(node, FunctionDef):
                func_name = node.name
                
                # Get argument names
                arg_names = [arg.arg for arg in node.args.args]
                
                # Find local variables (targets of Assign with Store ctx)
                local_vars = set()
                for subnode in walk(node):
                    if isinstance(subnode, Assign):
                        for target in subnode.targets:
                            if isinstance(target, Name) and isinstance(target.ctx, Store):
                                local_vars.add(target.id)
                
                self.functions.append(FunctionTranslator(
                    name=func_name.upper(),
                    args=arg_names,
                    local_vars=list(local_vars),
                    body_code=node.body,
                    function_index=self.function_index,
                    is_main=func_name == "main",
                ))

                self.function_index += 1
    
    def translate(self, code: str) -> None:
        """
        Translates a single Python code string into LC-3 assembly.

        Steps:
        - Parses the Python code into an AST.
        - Analyzes the functions in the AST.
        - Calls each function's `translate()` method to get its LC-3 subroutine.
        - Joins and prints the translated assembly code.

        Args:
            code (str): The Python source code to translate.
        """
        tree = parse(code)
        self.analyze_functions(tree)
        subroutines = []
        for function in self.functions:
            subroutines.append(function.translate())
        code = "\n\n".join(subroutines)
        print(code)

    
    def translate_all(self) -> None:
        """
        Translates all `.py` files in the `py/` directory into `.asm` files in the `asm/` directory.

        For each file:
        - Parses the source code into an AST.
        - Analyzes all functions and tracks subroutine dependencies.
        - Re-analyzes any additional required subroutines using the global `additional_subroutines` map.
        - Writes the combined translated LC-3 code into a matching `.asm` file.
        """
        python_dir = "py"
        asm_dir = "asm"
        os.makedirs(asm_dir, exist_ok=True)

        for filename in os.listdir(python_dir):
            if not filename.endswith(".py"):
                continue

            filepath = os.path.join(python_dir, filename)
            with open(filepath, "r") as f:
                code = f.read()

            # Reset state
            self.functions = []
            self.function_index = 0

            # Analyze base functions
            tree = parse(code)
            self.analyze_functions(tree)

            subroutines = []
            i = 0
            while i < self.function_index:
                function = self.functions[i]
                asm_code, subroutines_needed = function.translate()
                subroutines.append(asm_code)

                # Handle additional subroutines
                for subroutine_name in subroutines_needed:
                    if subroutine_name in ADDITIONAL_SUBROUTINES:
                        subroutine_code = ADDITIONAL_SUBROUTINES[subroutine_name]
                        sub_tree = parse(subroutine_code)
                        self.analyze_functions(sub_tree)
                i += 1

            # Save to corresponding .asm file
            output_filename = filename.replace(".py", ".asm")
            output_path = os.path.join(asm_dir, output_filename)
            with open(output_path, "w") as out_file:
                out_file.write("\n\n".join(subroutines))

    def translate_selected(self, filenames: list[str]) -> None:
        """
        Translates only the specified `.py` files from the `py/` directory into `.asm` files in the `asm/` directory.

        Args:
            filenames (list[str]): List of Python source file names (e.g., ["main.py", "math_utils.py"]).
        """
        python_dir = "py"
        asm_dir = "asm"
        os.makedirs(asm_dir, exist_ok=True)

        valid_filenames = []
        for filename in filenames:
            if not filename.endswith(".py"):
                continue
            filepath = os.path.join(python_dir, filename)
            if os.path.exists(filepath):
                valid_filenames.append(filename)
            else:
                print(f"File not found: {filepath}")

        if not valid_filenames:
            print("No valid files to translate.")
            return

        for filename in valid_filenames:

            filepath = os.path.join(python_dir, filename)

            with open(filepath, "r") as f:
                code = f.read()

            # Reset state
            self.functions = []
            self.function_index = 0

            # Analyze base functions
            tree = parse(code)
            self.analyze_functions(tree)

            subroutines = []
            i = 0
            while i < self.function_index:
                function = self.functions[i]
                asm_code, subroutines_needed = function.translate()
                subroutines.append(asm_code)

                # Handle additional subroutines
                for subroutine_name in subroutines_needed:
                    if subroutine_name in ADDITIONAL_SUBROUTINES:
                        subroutine_code = ADDITIONAL_SUBROUTINES[subroutine_name]
                        sub_tree = parse(subroutine_code)
                        self.analyze_functions(sub_tree)
                i += 1

            # Save to corresponding .asm file
            output_filename = filename.replace(".py", ".asm")
            output_path = os.path.join(asm_dir, output_filename)
            with open(output_path, "w") as out_file:
                out_file.write("\n\n".join(subroutines))
        
    def print_ast(self, code: str) -> None:
        """
        Prints the AST of the given Python code for debugging purposes.

        Args:
            code (str): The Python source code whose AST should be printed.
        """
        tree = parse(code)
        print(dump(tree, indent=4, include_attributes=False))
