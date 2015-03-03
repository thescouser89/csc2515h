function B = subsasgn(A, S, B)
fprintf(1, 'in subsasgn');
B = builtin('subsasgn', A, S, B);
