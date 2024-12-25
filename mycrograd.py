import math
import numpy as np

class Value:
    def __init__(self, value, _children=(), ):
        self.v = (value)
        self._prev = set(_children)
        self.grad = 0.0 if np.isscalar(value) else np.zeros_like(value)
        self._back = lambda: None

    def __add__(self,other):
        other = other if isinstance(other,Value) else Value(other)
        out = Value(self.v+other.v,(self,other))

        def _back():
            if np.isscalar(out.grad):
                self.grad += out.grad
                other.grad += out.grad

        out._back = _back
        return out

    def __mul__(self,other):
        other = other if isinstance(other,Value) else Value(other)
        out = Value(self.v*other.v,(self,other))

        def _back():
            self.grad += other.v*out.grad
            other.grad += self.v*out.grad

        out._back = _back
        return out

    def __pow__(self, other):
        other = other if isinstance(other,Value) else Value(other)
        out = Value(self.v**other.v,(self,other))

        def _back():
            self.grad += other.v*(self.v**(other.v-1))*out.grad
            other.grad += math.copysign(1,out.v)*out.v*math.log(abs(self.v))*out.grad

        out._back = _back
        return out

    def backward(self):
        self.grad = 1
        seen = set()
        def walk_back(e):
            if e not in seen:
                seen.add(e)
                e._back()
                for i in e._prev:
                    walk_back(i)
                    
        walk_back(self)
        
        

    def __sub__(self,other):
        other = other if isinstance(other,Value) else Value(other)
        return self + (Value(-1)*other)

    def __truediv__(self,other):
        other = other if isinstance(other,Value) else Value(other)
        return self*(other**-1)

    def __repr__(self):
        return f"Value(v={self.v})"

    
