
import main
import math

def f(x: float) -> float:
    return math.sin(x)

network = main.Network(1, 2, [4, 4, 1], 1, [main.ReLU, main.ReLU, main.ReLU, main.ReLU])




