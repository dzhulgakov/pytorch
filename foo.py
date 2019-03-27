















import torch

msgs = []

class Observer:
    def __init__(self, id):
        self.id = id

    def __call__(self, x):
        print("Called {} with {}".format(self.id,x))
        self.id += "BLA"

    def get_stats(self):
        return len(self.id)

ob1 = Observer("!"*15)
ob2 = Observer("!"*20)

def qqq(x):
    print('hi')

@torch.jit.script
def foo(x):
    ob1(x)
    x=x+1
    ob2(x)
    return x

#import pdb; pdb.set_trace()
print(foo.graph)
torch._C.qqq(foo.graph, Observer)
torch._C.qqq(foo.graph, Observer)
print(foo.graph)

foo(torch.tensor([5]))

torch._C.qqq_stats(foo.graph)

#print(list(foo.graph.nodes())[1].pyobj())
