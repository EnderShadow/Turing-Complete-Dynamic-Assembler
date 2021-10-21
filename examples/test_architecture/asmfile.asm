# calculate fibonacci number
    xor r0 r0 r0
    add r0 1 r1
    add r0 0 io
label loop:
    add r0 r1 r2
    add r1 0 r0
    add r2 0 r1
    add r0 0 io
    cmp r0 233
<   jmp loop

# stop executing code at this point
    jmp 0 pc

# compilation tests below. Never executed
=   call loop
    ret

# 233