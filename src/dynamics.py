import math

def next_state(s, a):
    v, L, dt = 5, 5, 0.05
    p, theta = s

    for step in range(20):
        p += v * dt * math.sin(math.radians(theta))
        theta += math.degrees(v / L * dt*math.tan(math.radians(a)))

    s_ = [p, theta]
    return s_
