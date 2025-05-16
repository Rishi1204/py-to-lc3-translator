STACK_BUILDUP = """; ====== BUILDUP ======
ADD R6, R6, -1  ; Allocate space for the return value
ADD R6, R6, -1  ; Save return address (R7)
STR R7, R6, 0
ADD R6, R6, -1  ; Save old frame pointer (R5)
STR R5, R6, 0 
ADD R6, R6, -1  ; Allocate space for one local variable (can have more if necessary)
ADD R5, R6, 0   ; Set current frame pointer to first local variable
ADD R6, R6, -{local_var_count}  ; set R5 to first local var and allocate space for {local_var_count} local vars
ADD R6, R6, -1
STR R0, R6, 0   ; save R0
ADD R6, R6, -1
STR R1, R6, 0   ; save R1
ADD R6, R6, -1
STR R2, R6, 0   ; save R2
ADD R6, R6, -1
STR R3, R6, 0   ; save R3
ADD R6, R6, -1
STR R4, R6, 0   ; save R4
{arg_loads}

; ====== SUBROUTINE ======
"""

STACK_TEARDOWN = """
; ====== TEARDOWN ======
STR R0, R5, 3   ; save return value on stack (if return value was in R0)
LDR R4, R6, 0   ; restore R4
ADD R6, R6, 1   ; pop R4 off stack
LDR R3, R6, 0   ; restore R3
ADD R6, R6, 1   ; pop R3 off stack
LDR R2, R6, 0   ; restore R2
ADD R6, R6, 1   ; pop R2 off stack
LDR R1, R6, 0   ; restore R1
ADD R6, R6, 1   ; pop R1 off stack
LDR R0, R6, 0   ; restore R0
ADD R6, R6, 1   ; pop R0 off stack
ADD R6, R5, 1   ; pop all local variables
LDR R5, R6, 0   ; restore R5 (old FP)
ADD R6, R6, 1   ; pop R5 off stack
LDR R7, R6, 0   ; restore R7 (return address)
ADD R6, R6, 1   ; pop R7 off stack
RET
"""

ADDITIONAL_SUBROUTINES = {"mult": """def mult(a, b):
                                    c = 0
                                    while a > 0:
                                        c = c + b
                                        a = a - 1
                                    return c
                                """}