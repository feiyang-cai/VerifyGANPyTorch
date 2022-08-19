import math

def next_state(s, a):
    v, L, dt = 5, 5, 0.05
    
    p, theta = s
    p_ = v * dt * math.sin(math.radians(theta)) + p
    theta_ = v / L * math.tan(math.radians(a)) + theta

    s_ = [p_, theta_]
    return s_
