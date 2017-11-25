import ast
from functools import partial
import inspect
import math
from operator import mul, add


class Variable(object):
    def __init__(self, name, parents, op, grad_ops):
        self.name = name
        self.parents = parents
        self.op = op
        self.grad_ops = grad_ops
        self.val = None

    def set(self, val):
        self.val = val

    def eval(self):
        if self.val is None:
            self.val = self.op(*[p.eval() for p in self.parents])
        return self.val

    def __repr__(self):
        return "Variable(%r, %r, %r, %r, %r)" % (
            self.name, self.parents, self.op, self.grad_ops, self.val)


class Graph(object):
    def __init__(self, vars):
        self.vars = vars

    def get_operation(self, v):
        return v.op

    def get_inputs(self, v):
        return v.parents

    def get_consumers(self, v):
        children = []
        for var in self.vars:
            if var.parents is not None and v in var.parents:
                children.append(var)
        return children


def _reverse_autodiff(nodes, inputs, output, *input_values):
    g = Graph(nodes)

    # Forward pass
    for input, input_value in zip(inputs, input_values):
        input.set(input_value)
    output.eval()

    # Reverse pass

    grad_table = {output: 1}  # final output has gradient of 1

    def build_grad(v, g, grad_table):
        if v in grad_table:
            return grad_table[v]
        # print("build_grad for %s" % v.name)
        grad = 0
        for c in g.get_consumers(v):
            # print("consumer %s with inputs %s" % (c.name, [i.name for i in g.get_inputs(c)]))
            d = build_grad(c, g, grad_table)
            # print("\tgrad_ops %s" % (c.grad_ops,))
            c_inputs = g.get_inputs(c)
            index = c_inputs.index(v)  # the index of v in the inputs to c
            grad += c.grad_ops[index](*[input.eval() for input in c_inputs]) * d
        grad_table[v] = grad
        # print("grad_table for %s is %s" % (v.name, grad))
        return grad_table[v]

    for input in inputs:
        build_grad(input, g, grad_table)

    if len(inputs) == 1:
        return grad_table[inputs[0]]
    else:
        return tuple(grad_table[input] for input in inputs)


def _autodiff(nodes, input, output):
    return partial(_reverse_autodiff, nodes, input, output)


def identity(x):
    return x


def const(c):
    return lambda x: c


# tanh(x)
n0 = Variable("n0", None, None, None)
n1 = Variable("n1", (n0,), identity, (const(1),))
n2 = Variable("n2", (n1,), math.tanh,
              (lambda x: 1 / (math.cosh(x) * math.cosh(x)),))

fdash = _autodiff((n1, n2), n0, n2)
#print(fdash(1))

# x * tanh(x)
n3 = Variable("n3", (n0,), identity, (const(1),))
n4 = Variable("n4", (n2, n3), mul, (lambda x, y: y, lambda x, y: x))

fdash = _autodiff((n0, n1, n2, n3, n4), n0, n4)
#print(fdash(1))

# x * x
n1 = Variable("n1", (n0,), identity, (const(1),))
n2 = Variable("n2", (n0,), identity, (const(1),))
n3 = Variable("n3", (n1, n2), mul, (lambda x, y: y, lambda x, y: x))

fdash = _autodiff((n1, n2, n3), n0, n3)
#print(fdash(1))

####


def find_arg_names(f):
    '''Returns a list of argument names for a function.'''
    src = inspect.getsource(f)
    tree = ast.parse(src)
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            for arg in node.args.args:
                names.append(arg.arg)
    return names


def extract_return_node_value(f):
    src = inspect.getsource(f)
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Return):
            return node.value
    return None


derivatives = {}


def register_reverse_autodiff(function, derivative):
    derivatives[function.__name__] = (function, derivative)


def reverse_autodiff(f):
    def gensym(vars):
        return "n" + str(len(vars))

    def consume(node, vars):
        if isinstance(node, ast.Name):
            if node.id in args:
                var = Variable(gensym(vars), (args[node.id],), identity, (const(1),))
                vars.append(var)
                return var
        elif isinstance(node, ast.BinOp):
            left = consume(node.left, vars)
            right = consume(node.right, vars)
            if isinstance(node.op, ast.Mult):
                var = Variable(gensym(vars), (left, right), mul,
                               (lambda x, y: y, lambda x, y: x))
                vars.append(var)
                return var
            elif isinstance(node.op, ast.Add):
                var = Variable(gensym(vars), (left, right), add,
                               (lambda x, y: 1, lambda x, y: 1))
                vars.append(var)
                return var
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id in derivatives:
                    arg = consume(node.args[0], vars)  # TODO: what if not single-valued?
                    function, derivative = derivatives[node.func.id]
                    var = Variable(gensym(vars), (arg,), function, (derivative,))
                    vars.append(var)
                    return var
        else:
            print("No match for", node)

    args = {}
    for arg_name in find_arg_names(f):
        arg = Variable(gensym(args), None, None, None)
        args[arg_name] = arg
    vars = [arg for arg in args.values()]
    v = consume(extract_return_node_value(f), vars)
    return _autodiff(vars, [arg for arg in args.values()], v)


# Register functions and their derivatives here
register_reverse_autodiff(math.sin, math.cos)
register_reverse_autodiff(math.tanh, lambda x: 1 / (math.cosh(x) * math.cosh(x)))