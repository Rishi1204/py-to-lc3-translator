from typing import Any, Optional
from ast import BinOp, Call, Constant, Name, Assign, Return, If, While, Compare, Load, unparse, walk, Add, Sub, BitAnd, Mult, Eq, NotEq, Gt, GtE, Lt, LtE, Expr 

from constants import STACK_BUILDUP, STACK_TEARDOWN

class FunctionTranslator:
    def __init__(self, name, args, local_vars, body_code, function_index, is_main=False):
        self.name = name
        self.args = args
        self.local_vars = local_vars
        self.local_var_count = len(local_vars) + len(args)
        self.body = body_code
        self.reg_num = 0
        self.register_count = 5
        self.var_reg_map = {} # variable -> register mapping
        self.reg_var_map = {} # register -> variable mapping
        self.var_stack_map = {} # variable -> offset mapping
        self.temp_vars = set()
        self.curr_lv_count = 0
        self.subroutine = []
        self.fill_labels = {}
        if is_main:
            self.fill_labels = {"STACK_POINTER":"xFE00"}
        self.is_main = is_main
        self.function_index = function_index
        self.starting_address = 3000 + (function_index * 100) # starting address for the function
        self.while_count = 0
        self.if_count = 0
        self.tmp_var_cnt = 0
        self.subroutines_needed = set()
    
    def load_arguments(self) -> str:
        """
        Generate LC3 assembly instructions to load function arguments into registers R0 to R4.

        Assumes arguments are passed on the stack. Updates variable-to-register mappings.

        Raises:
            Exception: If more than 5 arguments are provided.

        Returns:
            str: LC3 assembly instructions for loading arguments.
        """
        if len(self.args) > 5:
            raise Exception("Too many arguments for LC3. Max 5 arguments allowed.")
        arg_instructions = []
        for arg in self.args:
            arg_instructions.append(f"LDR R{self.reg_num}, R5, {4 + self.reg_num}   ; load argument {arg} into R{self.reg_num}")
            self.update_mappings(self.reg_num, arg)
            self.reg_num += 1
        return "\n".join(arg_instructions)

    def stack_buildup(self) -> None:
        """
        Generate LC3 assembly instructions to set up the stack frame for the function.
        """
        self.subroutine.extend(STACK_BUILDUP.format(local_var_count=self.local_var_count, arg_loads=self.load_arguments()).splitlines())
    
    def stack_teardown(self) -> None:
        """
        Generate LC3 assembly instructions to clean up the stack frame after function execution.
        """
        self.subroutine.extend(STACK_TEARDOWN.splitlines())
    
    def push_local_var(self, reg: int, var: str) -> None:
        """
        Push the value of a local variable from a register to the stack.

        Args:
            reg (int): The register containing the value to be pushed.
            var (str): The variable to be pushed.
        """
        offset = self.var_stack_map[var]
        self.subroutine.append(f"STR R{reg}, R5, {-offset} ;; push {var} (R{reg}) to stack as local var")
        del self.reg_var_map[reg]
        del self.var_reg_map[var]
    
    def restore_local_var(self, reg: int, var: str) -> None:
        """
        Restore the value of a local variable from the stack to a register.

        Args:
            reg (int): The register to which the value will be restored.
            var (str): The variable to be restored.
        """
        offset = self.var_stack_map[var]
        self.subroutine.append(f"LDR R{reg}, R5, {-offset} ;; restore {var} to R{reg}")
        self.update_mappings(reg, var)
    
    def update_mappings(self, reg: int, var: str) -> None:
        """
        Update the register-to-variable and variable-to-register mappings.

        Args:
            reg (int): The register number.
            var (str): The variable name.
        """
        # print(f"Updating mappings: {reg} -> {var}")
        if reg in self.reg_var_map:
            old_var = self.reg_var_map[reg]
            if old_var in self.var_reg_map:
                del self.var_reg_map[old_var]
        if var in self.var_reg_map:
            old_reg = self.var_reg_map[var]
            if old_reg in self.reg_var_map:
                del self.reg_var_map[old_reg]
        self.reg_var_map[reg] = var
        self.var_reg_map[var] = reg

    def create_const_label(self, imm_val: int) -> str:
        """
        Create a unique label for a constant value and store it in the fill_labels dictionary.

        Args:
            imm_val (int): The constant value for which the label is created.

        Returns:
            str: The generated label for the constant value.
        """
        label = f"CONST{self.function_index}_{imm_val}".replace("-", "NEG")
        self.fill_labels[label] = imm_val
        return label

    def eval_operand(self, operand: Any, avoid: list[int]=[], is_subtract: bool=False) -> tuple[int | str, bool, Optional[str]]:
        """
        Evaluate an operand and return the register or immediate value.

        Args:
            operand: The operand to evaluate (Name, Constant, or Call).
            avoid (list[int]): Registers to avoid using.
            is_subtract (bool): If True, treat operand as negated for subtraction.

        Returns:
            tuple: (reg or immediate value, is_imm (bool), var_to_restore (str or None))
        """
        if isinstance(operand, Name):
            reg = self.find_register(operand.id, avoid)
            return reg, False, None

        elif isinstance(operand, Constant):
            imm_val = -operand.value if is_subtract else operand.value
            if -16 <= imm_val <= 15:
                return imm_val, True, None
            else:
                label = self.create_const_label(imm_val)
                reg, var_to_restore = self.find_temp_register(avoid)
                self.subroutine.append(f"LD R{reg}, {label} ;; R{reg} = {imm_val}")
                return reg, True, var_to_restore

        elif isinstance(operand, Call):
            var = unparse(operand)
            var_to_restore = self.handle_call(operand, var, dst_is_temp=True, dst_is_return=False)
            reg = self.var_reg_map[var]
            return reg, False, var_to_restore

        else:
            raise Exception(f"Unsupported operand type: {type(operand)}")
    
    def add(self, var: str, bin_op: BinOp, dst_is_temp: bool=False, dst_is_return: bool=False) -> Optional[str]:
        """
        Handle the addition operation for binary operations.

        Args:
            var: The variable to store the result.
            bin_op (BinOp): The binary operation object.
            dst_is_temp (bool): If True, the destination register is temporary.
            dst_is_return (bool): If True, the destination register is for return value.

        Returns:
            str or None: The variable to restore if the destination register is temporary.
        """

        left, left_is_imm, left_var_to_restore = self.eval_operand(bin_op.left)
        right, right_is_imm, right_var_to_restore = self.eval_operand(
            bin_op.right, avoid=[] if (not left_is_imm and not left_var_to_restore) else [left]
        )

        if dst_is_temp:
            dst_reg, var_to_restore = self.find_temp_register([left, right], var if dst_is_temp else None)
            self.update_mappings(dst_reg, var)
        
        # if the destination register is a return value, use R0
        elif dst_is_return:
            dst_reg = 0
        
        # if the destination register is not temporary, find a register for it
        else:
            dst_reg = self.find_register(var, avoid_registers=[left, right])

        # left: immediate, right: immediate
        if (left_is_imm and left_var_to_restore is None) and (right_is_imm and right_var_to_restore is None):
            imm_val = left + right
            label = self.create_const_label(imm_val)
            self.subroutine.append(f"LD R{dst_reg}, {label} ;; {var} (R{dst_reg}) = {left} + {right} = {imm_val}")

        # left: immediate, right: register
        elif left_is_imm and left_var_to_restore is None:
            right_var = self.reg_var_map[right]
            self.subroutine.append(f"ADD R{dst_reg}, R{right}, {left} ;; {var} = {left} + {right_var} (R{right})")

        # left: register, right: immediate
        elif right_is_imm and right_var_to_restore is None:
            left_var = self.reg_var_map[left]
            self.subroutine.append(f"ADD R{dst_reg}, R{left}, {right} ;; {var} = {left_var} (R{left}) + {right}")

        # left: register, right: register
        else:
            right_var = self.reg_var_map[right]
            left_var = self.reg_var_map[left]
            self.subroutine.append(f"ADD R{dst_reg}, R{left}, R{right} ;; {var} = {left_var} (R{left}) + {right_var} (R{right})")
        
        if not dst_is_return:
            if left_var_to_restore:
                self.restore_local_var(left, left_var_to_restore)
            if right_var_to_restore:
                self.restore_local_var(right, right_var_to_restore)
        
        return var_to_restore if dst_is_temp and not dst_is_return else None
        
    def subtract(self, var: str, bin_op: BinOp, dst_is_temp: bool=False, dst_is_return: bool=False) -> Optional[str]:
        """
        Handle the subtraction operation for binary operations.

        Args:
            var: The variable to store the result.
            bin_op (BinOp): The binary operation object.
            dst_is_temp (bool): If True, the destination register is temporary.
            dst_is_return (bool): If True, the destination register is for return value.

        Returns:
            str or None: The variable to restore if the destination register is temporary.
        """
        left, left_is_imm, left_var_to_restore = self.eval_operand(bin_op.left)
        right, right_is_imm, right_var_to_restore = self.eval_operand(
            bin_op.right, avoid=[] if (not left_is_imm and not left_var_to_restore) else [left],
            is_subtract=True
        )

        # if the destination register is temporary, find a register for it
        if dst_is_temp:
            dst_reg, var_to_restore = self.find_temp_register([left, right], var if dst_is_temp else None)
            self.update_mappings(dst_reg, var)
        
        # if the destination register is a return value, use R0
        elif dst_is_return:
            dst_reg = 0
        
        # if the destination register is not temporary, find a register for it
        else:
            dst_reg = self.find_register(var, avoid_registers=[left, right])

        # left: immediate, right: immediate
        if (left_is_imm and left_var_to_restore is None) and (right_is_imm and right_var_to_restore is None):
            imm_val = left - right
            label = self.create_const_label(imm_val)
            self.subroutine.append(f"LD R{dst_reg}, {label} ;; {var} (R{dst_reg}) = {left} - {right} = {imm_val}")

        # left: immediate, right: register
        elif left_is_imm and left_var_to_restore is None:
            # if a temp register was used, calculate 2's complement in that register
            right_var = self.reg_var_map[right]
            if right_var_to_restore:
                self.subroutine.append(f"NOT R{right}, R{right}")
                self.subroutine.append(f"ADD R{right}, R{right}, 1 ;; -{right_var} (R{right})")
                self.subroutine.append(f"ADD R{dst_reg}, R{right}, {left} ;; {var} = {left} - {right_var} (R{right})")
            else:
                neg_reg, var_to_restore = self.find_temp_register([dst_reg, right])
                self.subroutine.append(f"NOT R{neg_reg}, R{right}")
                self.subroutine.append(f"ADD R{neg_reg}, R{neg_reg}, 1 ;; -{right_var} (R{neg_reg})")
                self.subroutine.append(f"ADD R{dst_reg}, R{neg_reg}, {left} ;; {var} = {left} - {right_var} (R{neg_reg})")
                if var_to_restore:
                    self.restore_local_var(neg_reg, var_to_restore)

        # left: register, right: immediate
        elif right_is_imm and right_var_to_restore is None:
            # if a temp register was used, calculate 2's complement in that register
            left_var = self.reg_var_map[left]
            self.subroutine.append(f"ADD R{dst_reg}, R{left}, {right} ;; {var} = {left_var} (R{left}) - {-right}")


        # left: register, right: register
        else:
            right_var = self.reg_var_map[right]
            left_var = self.reg_var_map[left]
            not_reg, var_to_restore = self.find_temp_register([dst_reg, left, right])
            self.subroutine.append(f"NOT R{not_reg}, R{right}")
            self.subroutine.append(f"ADD R{not_reg}, R{not_reg}, 1 ;; -{right_var} (R{not_reg})")
            self.subroutine.append(f"ADD R{dst_reg}, R{left}, R{not_reg} ;; {var} = {left_var} (R{left}) - {right_var} (R{right})")
            if var_to_restore:
                self.restore_local_var(not_reg, var_to_restore)

        if not dst_is_return:
            if left_var_to_restore:
                self.restore_local_var(left, left_var_to_restore)
            if right_var_to_restore:
                self.restore_local_var(right, right_var_to_restore)
        
        return var_to_restore if dst_is_temp and not dst_is_return else None
        
    def bitwise_and(self, var: str, bin_op: BinOp, dst_reg: Optional[int]=None, dst_is_temp: bool=False, dst_is_return: bool=False) -> Optional[str]:
        """
        Handle the bitwise AND operation for binary operations.

        Args:
            var: The variable to store the result.
            bin_op (BinOp): The binary operation object.
            dst_reg (int, optional): The destination register.
            dst_is_temp (bool): If True, the destination register is temporary.
            dst_is_return (bool): If True, the destination register is for return value.

        Returns:
            str or None: The variable to restore if the destination register is temporary.
        """

        left, left_is_imm, left_var_to_restore = self.eval_operand(bin_op.left)
        right, right_is_imm, right_var_to_restore = self.eval_operand(
            bin_op.right, avoid=[] if (not left_is_imm and not left_var_to_restore) else [left]
        )

        # if the destination register is temporary, find a register for it
        if dst_is_temp:
            dst_reg, var_to_restore = self.find_temp_register([left, right], var if dst_is_temp else None)
            self.update_mappings(dst_reg, var)
        
        # if the destination register is a return value, use R0
        elif dst_is_return:
            dst_reg = 0
        
        # if the destination register is not temporary, find a register for it
        else:
            dst_reg = self.find_register(var, avoid_registers=[left, right])
        

        # left: immediate, right: immediate
        if (left_is_imm and left_var_to_restore is None) and (right_is_imm and right_var_to_restore is None):
            imm_val = left + right
            label = self.create_const_label(imm_val)
            self.subroutine.append(f"LD R{dst_reg}, {label} ;; {var} (R{dst_reg}) = {left} & {right} = {imm_val}")

        # left: immediate, right: register
        elif left_is_imm and left_var_to_restore is None:
            right_var = self.reg_var_map[right]
            self.subroutine.append(f"AND R{dst_reg}, R{right}, {left} ;; {var} = {left} & {right_var} (R{right})")

        # left: register, right: immediate
        elif right_is_imm and right_var_to_restore is None:
            left_var = self.reg_var_map[left]
            self.subroutine.append(f"AND R{dst_reg}, R{left}, {right} ;; {var} = {left_var} (R{left}) & {right}")

        # left: register, right: register
        else:
            right_var = self.reg_var_map[right]
            left_var = self.reg_var_map[left]
            self.subroutine.append(f"AND R{dst_reg}, R{left}, R{right} ;; {var} = {left_var} (R{left}) & {right_var} (R{right})")
        
        if not dst_is_return:
            if left_var_to_restore:
                self.restore_local_var(left, left_var_to_restore)
            if right_var_to_restore:
                self.restore_local_var(right, right_var_to_restore)
        
        return var_to_restore if dst_is_temp and not dst_is_return else None

    def multiply(self, var: str, bin_op: BinOp, dst_is_temp: bool=False, dst_is_return: bool=False) -> Optional[str]:
        """
        Handle the multiplication operation for binary operations. Creates a subroutine for multiplication.

        Args:
            var: The variable to store the result.
            bin_op (BinOp): The binary operation object.
            dst_is_temp (bool): If True, the destination register is temporary.
            dst_is_return (bool): If True, the destination register is for return value.

        Returns:
            str or None: The variable to restore if the destination register is temporary.
        """

        left, right = bin_op.left, bin_op.right
        call_obj = Call(func=Name(id="mult", ctx=Load()), args=[left, right], keywords=[], starargs=None, kwargs=None)
        self.subroutine.append(f";; Calculating {unparse(bin_op)}")
        var_to_restore = self.handle_call(call_obj, var, dst_is_temp=dst_is_temp, dst_is_return=dst_is_return)
        self.subroutines_needed.add("mult")

        return var_to_restore if dst_is_temp and not dst_is_return else None

    def eval_binop(self, var: str, bin_op: BinOp, avoid_vars: list[int]=[], dst_is_temp: bool=False, dst_is_return: bool=False) -> Optional[str]:
        """
        Evaluate a binary operation and generate LC3 code for it.

        Args:
            var: The variable to store the result.
            bin_op (BinOp): The binary operation object.
            avoid_vars (list[int]): Registers to avoid using.
            dst_is_temp (bool): If True, the destination register is temporary.
            dst_is_return (bool): If True, the destination register is for return value.

        Returns:
            str or None: The variable to restore if the destination register is temporary.
        """

        left_var_to_restore, right_var_to_restore = None, None

        if isinstance(bin_op.left, BinOp):
            # create a temp var so that operation functions know where the result is going
            left_tmp = f"tmp_var_{self.tmp_var_cnt}"
            self.tmp_var_cnt += 1
            left_var_to_restore = self.eval_binop(left_tmp, bin_op.left, dst_is_temp=True)
            # edit the bin_op to use the temp var
            bin_op.left = Name(id=left_tmp, ctx=Load())
        else:
            left_tmp = None

        if isinstance(bin_op.right, BinOp):
            right_tmp = f"tmp_var_{self.tmp_var_cnt}"
            self.tmp_var_cnt += 1
            right_var_to_restore = self.eval_binop(
                right_tmp,
                bin_op.right,
                avoid_vars=[avoid_vars] if not left_tmp else avoid_vars + [left_tmp],
                dst_is_temp=True
            )
            # edit the bin_op to use the temp var
            bin_op.right = Name(id=right_tmp, ctx=Load())
        else:
            right_tmp = None
        
        var_to_restore = None
        
        if isinstance(bin_op.op, Add):
            var_to_restore = self.add(var, bin_op, dst_is_temp=dst_is_temp, dst_is_return=dst_is_return)
        elif isinstance(bin_op.op, Sub):
            var_to_restore = self.subtract(var, bin_op, dst_is_temp=dst_is_temp, dst_is_return=dst_is_return)
        elif isinstance(bin_op.op, BitAnd):
            var_to_restore = self.bitwise_and(var, bin_op, dst_is_temp=dst_is_temp, dst_is_return=dst_is_return)
        elif isinstance(bin_op.op, Mult):
            var_to_restore = self.multiply(var, bin_op, dst_is_temp=dst_is_temp, dst_is_return=dst_is_return)
        else:
            raise Exception("Unsupported binary operation type")
        
        # no point in restoring the variable if we are returning after this
        if not dst_is_return:
            if left_var_to_restore and left_tmp:
                reg = self.var_reg_map[left_tmp]
                self.restore_local_var(reg, left_var_to_restore)
            if right_var_to_restore and right_tmp:
                reg = self.var_reg_map[right_tmp]
                self.restore_local_var(reg, right_var_to_restore)

        return var_to_restore if dst_is_temp and not dst_is_return else None

    
    def evict_register(self, avoid_registers: list[int]=[]) -> tuple[int, Optional[str]]:
        """
        Evict a register that is not in the avoid_registers list.

        Args:
            avoid_registers (list[int]): Registers to avoid evicting.

        Returns:
            tuple: (reg (int), var_to_restore (str or None))
        """
        for _ in range(self.register_count):
            reg = self.reg_num
            self.reg_num = (self.reg_num + 1) % self.register_count
            if reg not in avoid_registers:
                break
        else:
            raise Exception("No available register to evict")
        
        # If the register is holding a temporary variable, there is no need to store its value on the stack
        if reg in self.reg_var_map:
            var = self.reg_var_map[reg]
            if var not in self.temp_vars:
                # If the variable is not in the stack, push it to the stack
                if var not in self.var_stack_map:
                    self.var_stack_map[var] = self.curr_lv_count
                    self.curr_lv_count += 1
                self.push_local_var(reg, var)
                return reg, var
            else:
                # If the variable is temporary, just remove it from the mappings
                # if it was needed for the current operation, avoid_registers would've stopped this
                del self.reg_var_map[reg]
                del self.var_reg_map[var]
                self.temp_vars.remove(var)
                return reg, None
        # If the register is not anything, just return it
        return reg, None

    
    def find_register(self, var: str, avoid_registers: list[int]=[]) -> int:
        """
        Get the register where the given variable is currently stored.

        Args:
            var (str): The variable name.
            avoid_registers (list[int]): Registers to avoid evicting.

        Returns:
            int: The register number containing the variable.
        """
        # If variable is currently in a register
        if var in self.var_reg_map:
            return self.var_reg_map[var]

        # If variable is on the stack
        elif var in self.var_stack_map:
            # find register that does not conflict with avoid_registers
            reg, _ = self.evict_register(avoid_registers)
            self.restore_local_var(reg, var)
            return reg

        # Variable never seen before
        else:
            # find register that does not conflict with avoid_registers
            reg, _ = self.evict_register(avoid_registers)
            self.update_mappings(reg, var)
            return reg
    
    # returns a register for temporary use which should be restored soon after (for NOT, for loop iterators, etc.)
    # also returns the variable so that it can be restored
    def find_temp_register(self, avoid_registers: list[int]=[], var: Optional[str]=None) -> tuple[int, Optional[str]]:
        """
        Find a register for temporary use, which should be restored soon after.

        Args:
            avoid_registers (list[int]): Registers to avoid evicting.
            var (str, optional): If provided, mark this variable as temporary.

        Returns:
            tuple: (reg (int), var_to_restore (str or None))
        """
        reg, var_to_restore = self.evict_register(avoid_registers)
        if var:
            self.temp_vars.add(var)
        return reg, var_to_restore
    
    def caller_buildup(self, call_obj: Call) -> list[int]:
        """
        Prepare the stack for a function call by pushing arguments onto the stack.

        Args:
            call_obj (Call): The Call object representing the function call.

        Returns:
            list[int]: List of registers to avoid using after the call.
        """
        avoid_registers = []
        args = call_obj.args
        function_name = call_obj.func.id.upper()
        # assumes function has at most 5 args 
        # push them in reverse order
        self.subroutine.append(f"ADD R6, R6, -{len(args)} ;; make space to push args in reverse order")
        i = 0
        for arg in args:
            # if the argument is a constant, load it into a register and push it to the stack
            if isinstance(arg, Constant):
                reg, var_to_restore = self.find_temp_register()
                imm_val = arg.value
                label = f"CONST{self.function_index}_{imm_val}".replace("-", "NEG")
                self.fill_labels[label] = imm_val
                self.subroutine.append(f"LD R{reg}, {label} ;; R{reg} = {imm_val}")
                self.subroutine.append(f"STR R{reg}, R6, {i} ;; push {imm_val} (R{reg}) onto the stack for {function_name}")
                if var_to_restore:
                    self.restore_local_var(reg, var_to_restore)
            # if the argument is a variable, find its register and push it to the stack
            elif isinstance(arg, Name):
                reg = self.find_register(arg.id)
                self.subroutine.append(f"STR R{reg}, R6, {i} ;; push {arg.id} (R{reg}) onto the stack for {function_name}")
            elif isinstance(arg, BinOp):
                # if the argument is a binary operation, evaluate it and push the result to the stack
                var_to_restore = self.eval_binop(unparse(arg), arg, dst_is_temp=True, dst_is_return=False)
                reg = self.var_reg_map[unparse(arg)]
                self.subroutine.append(f"STR R{reg}, R6, {i} ;; push {unparse(arg)} (R{reg}) onto the stack for {function_name}")
                if var_to_restore:
                    self.restore_local_var(reg, var_to_restore)
            elif isinstance(arg, Call):
                # if the argument is a function call, build up the caller and push the result to the stack
                var_to_restore = self.handle_call(arg, unparse(arg), dst_is_temp=True, dst_is_return=False)
                reg = self.var_reg_map[unparse(arg)]
                self.subroutine.append(f"STR R{reg}, R6, {i} ;; push {unparse(arg)} (R{reg}) onto the stack for {function_name}")
                if var_to_restore:
                    self.restore_local_var(reg, var_to_restore)
            else:
                raise Exception("Unsupported argument type")

            avoid_registers.append(reg)
            i += 1
        self.subroutine.append(f"JSR {function_name} ;; call {function_name}")
        return avoid_registers
    
    def caller_teardown(self, call_obj: Call, var: Optional[str]=None, avoid_registers: Optional[list[int]]=None, dst_is_temp: bool=False, dst_is_return: bool=False) -> Optional[str]:
        """
        Clean up the stack after a function call by popping arguments off the stack.

        Args:
            call_obj (Call): The Call object representing the function call.
            var (str, optional): The variable to store the return value.
            avoid_registers (list[int], optional): Registers to avoid using.
            dst_is_temp (bool): If True, the destination register is temporary.
            dst_is_return (bool): If True, the destination register is for return value.

        Returns:
            str or None: The variable to restore if the destination register is temporary.
        """
    
        args = call_obj.args
        function_name = call_obj.func.id.upper()

        dst_reg = None

        # if var is None, the return value is not stored anywhere
        if var is None:
            dst_reg = None

        # if the destination register is temporary, find a register for it
        if dst_is_temp:
            dst_reg, var_to_restore = self.find_temp_register(avoid_registers, var if dst_is_temp else None)
            self.update_mappings(dst_reg, var)
        
        # if the destination register is a return value, use R0
        elif dst_is_return:
            dst_reg = 0
        
        # if the destination register is not temporary, find a register for it
        else:
            dst_reg = self.find_register(var, avoid_registers=avoid_registers)
        
        if dst_reg is not None:
            # print(self.reg_var_map, self.var_reg_map)
            if dst_reg in self.reg_var_map:
                self.subroutine.append(f"LDR R{dst_reg}, R6, 0 ;; R{dst_reg} = {unparse(call_obj)}")
            else:
                self.subroutine.append(f"LDR R{dst_reg}, R6, 0 ;; return value (R{dst_reg}) = {unparse(call_obj)}")
        self.subroutine.append(f"ADD R6, R6, {len(args) + 1} ;; pop return value + args off stack for {function_name}")

        return var_to_restore if dst_is_temp and not dst_is_return else None
    
    def handle_assign(self, assign_obj: Assign) -> None:
        """
        Handle the assignment operation for variables.

        Args:
            assign_obj (Assign): The Assign object representing the assignment operation.
        """
        var = assign_obj.targets[0].id
        val = assign_obj.value
        if isinstance(val, Constant):
            imm_val = val.value
            self.fill_labels[f"{var.upper()}_{self.function_index}"] = imm_val
            assignee_reg = self.find_register(var)
            self.subroutine.append(f"LD R{assignee_reg}, {var.upper()}_{self.function_index} ;; {var} (R{assignee_reg}) = {imm_val}")
        elif isinstance(val, BinOp):
            var_to_restore = self.eval_binop(var, val)
            if var_to_restore:
                reg = self.var_reg_map[var]
                self.restore_local_var(reg, var_to_restore)
        elif isinstance(val, Name):
            val_reg = self.find_register(val.id)
            assignee_reg = self.find_register(var, [val_reg])
            self.subroutine.append(f"ADD R{assignee_reg}, R{val_reg}, 0 ;; {var} (R{assignee_reg}) = {val.id} (R{val_reg})")
        elif isinstance(val, Call):
            var_to_restore = self.handle_call(val, var, dst_is_temp=False, dst_is_return=False)
            if var_to_restore:
                reg = self.var_reg_map[var]
                self.restore_local_var(reg, var_to_restore)
        else:
            raise Exception("Unsupported assignment value type")
        

    def handle_return(self, return_obj: Return) -> None:
        """
        Handle the return operation for functions.

        Args:
            return_obj (Return): The Return object representing the return operation.
        """

        if return_obj.value is not None:
            ret_val = return_obj.value
            # if the return value is a variable
            if isinstance(ret_val, Name):
                ret_reg = self.find_register(ret_val.id)
                self.subroutine.append(f"ADD R0, R{ret_reg}, 0 ;; return value (R0) = {ret_val.id} (R{ret_reg})")
            # if the return value is a constant
            elif isinstance(ret_val, Constant):
                imm_val = ret_val.value
                label = self.create_const_label(imm_val)
                self.subroutine.append(f"LD R0, {label};; return value (R0) = {imm_val}")
            # if the return value is a binary operation
            elif isinstance(ret_val, BinOp):
                self.eval_binop(unparse(ret_val), ret_val, dst_is_temp=False, dst_is_return=True)
            # if the return value is a function call
            elif isinstance(ret_val, Call):
                self.handle_call(ret_val, "return value", dst_is_temp=False, dst_is_return=True)
            else:
                raise Exception("Unsupported return value type")
        else:
            self.subroutine.append("ADD R0, R0, 0 ;; return value (R0) = 0")
    
    def get_vars(self, body: list[Any]) -> set[str]:
        """
        Get all variable names in the body of a function.

        Args:
            body: The body of the function.

        Returns:
            set: A set of variable names used in the body.
        """
        # Get all variables in the body
        vars = set()

        for stmt in body:
            # Collect all Name nodes from this statement
            for node in walk(stmt):
                if isinstance(node, Name):
                    vars.add(node.id)

            # Recurse deeper into If and While bodies
            if isinstance(stmt, If):
                vars.update(self.get_vars(stmt.body))
                vars.update(self.get_vars(stmt.orelse))
            elif isinstance(stmt, While):
                vars.update(self.get_vars(stmt.body))

        return vars

    def handle_if(self, if_obj: If) -> None:
        """
        Handle the if statement and its body.

        Args:
            if_obj (If): The If object representing the if statement.
        """

        # Generate unique labels for else and end
        else_label = f"ELSE{self.function_index}_{self.if_count}"
        end_label = f"ENDIF{self.function_index}_{self.if_count}"
        self.if_count += 1
        # Evaluate the condition
        condition = if_obj.test
        # Evaluate condition
        if isinstance(condition, Compare):
            # Only supporting a simple "a > b", "a == b" etc. now.
            left = condition.left
            right = condition.comparators[0]
            op = condition.ops[0]
            self.subroutine.append(f";; Evaluating condition: {unparse(condition)}")
            # Evaluate left
            if isinstance(left, Name):
                left_reg = self.find_register(left.id)
            elif isinstance(left, Constant):
                imm_val = left.value
                left_reg, restore_var = self.find_temp_register()
                label = self.create_const_label(imm_val)
                self.subroutine.append(f"LD R{left_reg}, {label} ;; R{left_reg} = {imm_val}")
                if restore_var:
                    self.restore_local_var(left_reg, restore_var)
            else:
                raise Exception("Unsupported left condition")
            # Evaluate right
            # keep track of variables that have been pushed to the stack so that they can be restored later
            vars_to_restore = {}
            if isinstance(right, Name):
                right_reg = self.find_register(right.id, [left_reg])
                subtract_reg, var_to_restore = self.find_temp_register([left_reg, right_reg])
                if var_to_restore:
                    vars_to_restore[var_to_restore] = subtract_reg
                # perform two's complement on right
                self.subroutine.append(f"NOT R{subtract_reg}, R{right_reg}")
                self.subroutine.append(f"ADD R{subtract_reg}, R{subtract_reg}, 1 ;; {right.id} (R{subtract_reg}) = -{right.id}")
                # Compare → subtract right from left (can reuse the subtract register for comparison)
                self.subroutine.append(f"ADD R{subtract_reg}, R{left_reg}, R{subtract_reg} ;; Comparison: R{subtract_reg} = {left.id} (R{left_reg}) - {right.id} (R{subtract_reg})")
            elif isinstance(right, Constant):
                imm_val = -right.value
                right_reg, var_to_restore = self.find_temp_register([left_reg])
                if var_to_restore:
                    vars_to_restore[var_to_restore] = right_reg
                label = self.create_const_label(imm_val)
                self.subroutine.append(f"LD R{right_reg}, {label} ;; R{right_reg} = {imm_val}")
                # Compare → subtract right from left
                cond_reg, var_to_restore = self.find_temp_register([left_reg, right_reg])
                if var_to_restore:
                    vars_to_restore[var_to_restore] = cond_reg
                self.subroutine.append(f"ADD R{cond_reg}, R{left_reg}, R{right_reg} ;; Comparison: R{cond_reg} = {left.id} (R{left_reg}) - {-imm_val} (R{right_reg})")
            else:
                raise Exception("Unsupported right condition")
            # Branch based on comparison type
            if isinstance(op, Eq):
                self.subroutine.append(f"BRnp {else_label} ;; if !=, go to else")
            elif isinstance(op, NotEq):
                self.subroutine.append(f"BRz {else_label} ;; if ==, go to else")
            elif isinstance(op, Lt):
                self.subroutine.append(f"BRzp {else_label} ;; if >=, go to else")
            elif isinstance(op, Gt):
                self.subroutine.append(f"BRnz {else_label} ;; if <=, go to else")
            elif isinstance(op, LtE):
                self.subroutine.append(f"BRp {else_label} ;; if >, go to else")
            elif isinstance(op, GtE):
                self.subroutine.append(f"BRn {else_label} ;; if <, go to else")
            else:
                raise Exception("Unsupported comparison operator")
        else:
            raise Exception("Unsupported if condition type (only Compare supported now)")
        
        # handle if body
        for stmt in if_obj.body:
            self.handle_statement(stmt)

        # After if block, jump to end
        self.subroutine.append(f"BR {end_label}")
        # Else block (if present)
        # Else or Elif block
        self.subroutine.append(f"{else_label}")
        if len(if_obj.orelse) == 1 and isinstance(if_obj.orelse[0], If):
            # Nested If → this is an elif chain
            self.handle_if(if_obj.orelse[0])
        else:
            for stmt in if_obj.orelse:
                self.handle_statement(stmt)
        # End of if
        self.subroutine.append(f"{end_label}")
    
    def handle_while(self, while_obj: While) -> None:
        """
        Handle the while statement and its body.

        Args:
            while_obj (While): The While object representing the while statement.
        """

        # Generate unique labels
        start_label = f"WHILE{self.function_index}_{self.while_count}"
        end_label = f"ENDWH{self.function_index}_{self.while_count}"

        body = while_obj.body
        vars_in_while_loop = self.get_vars(body)
        var_reg_loop = dict()

        # push any register values that are going to be used in the while loop
        self.subroutine.append(f";; Pushing any variables that are going to be used in the while loop")
        for var in vars_in_while_loop:
            if var in self.var_reg_map:
                reg = self.var_reg_map[var]
                if var not in self.var_stack_map:
                    self.var_stack_map[var] = self.curr_lv_count
                    self.curr_lv_count += 1
                self.push_local_var(reg, var)
                var_reg_loop[var] = reg
        
        self.subroutine.append(f"{start_label}")

        # Evaluate condition
        condition = while_obj.test
        if isinstance(condition, Compare):
            # Only supporting a simple "a > b", "a == b" etc. now.
            left = condition.left
            right = condition.comparators[0]
            op = condition.ops[0]
            # Evaluate left
            if isinstance(left, Name):
                left_reg = self.find_register(left.id)
            elif isinstance(left, Constant):
                imm_val = left.value
                left_reg, restore_var = self.find_temp_register()
                label = self.create_const_label(imm_val)
                self.subroutine.append(f"LD R{left_reg}, {label} ;; R{left_reg} = {imm_val}")
                if restore_var:
                    self.restore_local_var(left_reg, restore_var)
            else:
                raise Exception("Unsupported left condition")
            # Evaluate right
            # keep track of variables that have been pushed to the stack so that they can be restored later
            vars_to_restore = {}
            if isinstance(right, Name):
                right_reg = self.find_register(right.id, [left_reg])
                subtract_reg, var_to_restore = self.find_temp_register([left_reg, right_reg])
                if var_to_restore:
                    vars_to_restore[var_to_restore] = subtract_reg
                # perform two's complement on right
                self.subroutine.append(f"NOT R{subtract_reg}, R{right_reg}")
                self.subroutine.append(f"ADD R{subtract_reg}, R{subtract_reg}, 1 ;; {right.id} (R{subtract_reg}) = -{right.id}")
                # Compare → subtract right from left (can reuse the subtract register for comparison)
                self.subroutine.append(f"ADD R{subtract_reg}, R{left_reg}, R{subtract_reg} ;; Comparison: R{subtract_reg} = {left.id} (R{left_reg}) - {right.id} (R{subtract_reg})")
            elif isinstance(right, Constant):
                imm_val = -right.value
                right_reg, var_to_restore = self.find_temp_register([left_reg])
                if var_to_restore:
                    vars_to_restore[var_to_restore] = right_reg
                label = self.create_const_label(imm_val)
                self.subroutine.append(f"LD R{right_reg}, {label} ;; R{right_reg} = {imm_val}")
                # Compare → subtract right from left
                cond_reg, var_to_restore = self.find_temp_register([left_reg, right_reg])
                if var_to_restore:
                    vars_to_restore[var_to_restore] = cond_reg
                self.subroutine.append(f"ADD R{cond_reg}, R{left_reg}, R{right_reg} ;; Comparison: R{cond_reg} = {left.id} (R{left_reg}) - {-imm_val} (R{right_reg})")
            else:
                raise Exception("Unsupported right condition")

            # Branch based on comparison type
            if isinstance(op, Eq):
                self.subroutine.append(f"BRnp {end_label} ;; if !=, exit")
            elif isinstance(op, NotEq):
                self.subroutine.append(f"BRz {end_label} ;; if ==, exit")
            elif isinstance(op, Lt):
                self.subroutine.append(f"BRzp {end_label} ;; if >=, exit")
            elif isinstance(op, Gt):
                self.subroutine.append(f"BRnz {end_label} ;; if <=, exit")
            elif isinstance(op, LtE):
                self.subroutine.append(f"BRp {end_label} ;; if >, exit")
            elif isinstance(op, GtE):
                self.subroutine.append(f"BRn {end_label} ;; if <, exit")
            else:
                raise Exception("Unsupported comparison operator")
        else:
            raise Exception("Unsupported while condition type (only Compare supported now)")
        
        # before starting the body of the while loop, restore the variables and their register values
        for var, reg in vars_to_restore.items():
            self.restore_local_var(reg, var)
        for var, reg in var_reg_loop.items():
            self.restore_local_var(reg, var)

        # Body of the while loop
        for stmt in while_obj.body:
            self.handle_statement(stmt)

        for var, reg in var_reg_loop.items():
            if var in self.var_reg_map:
                self.push_local_var(self.var_reg_map[var], var)

        # Jump back to start
        self.subroutine.append(f"BR {start_label}")

        # End of while
        self.subroutine.append(f"{end_label}")

        # to restore in case the loop never ran
        for var, reg in vars_to_restore.items():
            self.restore_local_var(reg, var)

        self.while_count += 1

    def handle_call(self, call_obj: Call, var: Optional[str]=None, dst_is_temp: bool=False, dst_is_return: bool=False) -> Optional[str]:
        """
        Handle function calls and their arguments.

        Args:
            call_obj (Call): The Call object representing the function call.
            var (str, optional): The variable to store the return value.
            dst_is_temp (bool): If True, the destination register is temporary.
            dst_is_return (bool): If True, the destination register is for return value.

        Returns:
            str or None: The variable to restore if the destination register is temporary.
        """

        # Handle function calls
        avoid_registers = self.caller_buildup(call_obj)
        var_to_restore = self.caller_teardown(call_obj, var, avoid_registers=avoid_registers, dst_is_temp=dst_is_temp, dst_is_return=dst_is_return)
        return var_to_restore if dst_is_temp and not dst_is_return else None
    
    def handle_statement(self, statement: Any) -> None:
        """
        Handle different types of statements in the function body.

        Args:
            statement: The statement object to handle.
        """

        if isinstance(statement, Assign):
            self.handle_assign(statement)
        elif isinstance(statement, Return):
            self.handle_return(statement)
        elif isinstance(statement, If):
            self.handle_if(statement)
        elif isinstance(statement, While):
            self.handle_while(statement)
        elif isinstance(statement, Call):
            self.handle_call(statement)
        elif isinstance(statement, Expr):
            self.handle_statement(statement.value)
        else:
            raise Exception(f"Unsupported statement type: {type(statement)}")
            

    def translate(self) -> tuple[str, list[str]]:
        """
        Translate the function to LC3 assembly code.

        Returns:
            tuple: (assembly (str), subroutines_needed (list))
        """

        self.subroutine.append(f"{self.name}")
        # if this is the main function, set the stack pointer
        if self.is_main:
            self.subroutine.append("LD R6, STACK_POINTER")
        # if this is the main function, do not do stack buildup
        if not self.is_main:
            self.stack_buildup()
        for statement in self.body:
            self.handle_statement(statement)
        if not self.is_main:
            self.stack_teardown()
        if self.is_main:
            self.subroutine.append("HALT")
        for label, value in self.fill_labels.items():
            self.subroutine.append(f"{label} .fill {value}")
        assembly = f".orig x{self.starting_address}\n     " + "\n     ".join(self.subroutine) + "\n.end"
        return assembly, list(self.subroutines_needed)

        
