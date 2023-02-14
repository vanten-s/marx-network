
import main
import math

def f(x: float) -> float:
    return math.sin(x)

network = main.Network(1, 4, [1, 4, 4, 1], 1, [main.ReLU, main.ReLU, main.ReLU, main.ReLU])

network = main.iterate( network, 10, 1, f, [.1] )




