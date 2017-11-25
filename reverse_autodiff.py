import ast
import inspect
import math
from operator import mul


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


def _reverse_autodiff(nodes, input, output, input_value):
    g = Graph(nodes)

    # Forward pass
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

    build_grad(input, g, grad_table)

    return grad_table[input]


def _autodiff(nodes, input, output):
    return lambda input_value: _reverse_autodiff(nodes, input, output, input_value)


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


def extract_return_node_value(f):
    src = inspect.getsource(f)
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Return):
            return node.value
    return None


def reverse_autodiff(f):
    n0 = Variable("n0", None, None, None)

    def gensym(vars):
        return "n" + str(len(vars))

    def consume(node, vars):
        if isinstance(node, ast.Name):
            if node.id == 'x':  # TODO other variables
                var = Variable(gensym(vars), (n0,), identity, (const(1),))
                vars.append(var)
                return var
        elif isinstance(node, ast.BinOp):
            if isinstance(node.op, ast.Mult):
                left = consume(node.left, vars)
                right = consume(node.right, vars)
                var = Variable(gensym(vars), (left, right), mul,
                               (lambda x, y: y, lambda x, y: x))
                vars.append(var)
                return var
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id == 'tanh':
                    arg = consume(node.args[0], vars)  # TODO: what if not single-valued?
                    var = Variable(gensym(vars), (arg,), math.tanh,
                                   (lambda x: 1 / (math.cosh(x) * math.cosh(x)),))
                    vars.append(var)
                    return var
        else:
            print("No match for", node)

    vars = [n0]
    v = consume(extract_return_node_value(f), vars)
    return _autodiff(vars, n0, v)

