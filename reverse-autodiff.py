import ast
import inspect		
def walk(f):
	src = inspect.getsource(f)
	tree = ast.parse(src)
	for node in ast.walk(tree):
		print(node)
	print(ast.dump(tree))

def f(x):
	return x * x
	
walk(f)

def fdash(x):
	ret = x * x
	
	# Backwards
	df_dn3 = 1.0
	
	df_dn1 = df_dn3 * x
	df_dn2 = df_dn3 * x
	
	df_dx = df_dn1 + df_dn2
	
	return df_dx
	

print(f(1))
print(fdash(1))

import math
def h(x):
  return math.tanh(x)

walk(h)