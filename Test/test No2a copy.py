import numpy as np

def objective_function(x, y):
    return (x + y**2 - 13)**2 + (x**2 + y - 9)**2

def init_particles(x, y, vx):
    particles = []
    for i in range(len(x)):
        particle = {'x': x[i], 'y': y[i], 'vx': vx[i], 'vy': 0, 'pbest': None}
        particles.append(particle)
    return particles

def find_pbest(particles):
    for particle in particles:
        particle['pbest'] = objective_function(particle['x'], particle['y'])
    return particles

def find_gbest(particles):
    gbest_particle = min(particles, key=lambda x: x['pbest'])
    return gbest_particle['x'], gbest_particle['y']

def update_velocity(particle, gbest_x, gbest_y, c, r, w):
    new_vx = w * particle['vx'] + c[0] * r[0] * (particle['pbest_x'] - particle['x']) + c[1] * r[1] * (gbest_x - particle['x'])
    new_vy = w * particle['vy'] + c[0] * r[0] * (particle['pbest_y'] - particle['y']) + c[1] * r[1] * (gbest_y - particle['y'])
    return new_vx, new_vy

def update_position(particle, new_vx, new_vy):
    particle['x'] += new_vx
    particle['y'] += new_vy
    return particle

def iterate(particles, c, r, w, iterations):
    for _ in range(iterations):
        for particle in particles:
            particle['pbest_x'], particle['pbest_y'] = find_pbest([particle])[-1]['x'], find_pbest([particle])[-1]['y']
        gbest_x, gbest_y = find_gbest(particles)
        for particle in particles:
            new_vx, new_vy = update_velocity(particle, gbest_x, gbest_y, c, r, w)
            particle = update_position(particle, new_vx, new_vy)
    return particles

def print_status(iteration, particles):
    print(f'Iteration {iteration + 1}:')
    for i, particle in enumerate(particles):
        fxy = objective_function(particle['x'], particle['y'])
        print(f'  Particle {i+1} - x: {particle["x"]}, y: {particle["y"]}, f(x, y): {fxy}, v: ({particle["vx"]}, {particle["vy"]}), pbest: {particle["pbest"]}')

# Initial parameters
x = [1.0, -1.0, 2.0]
y = [1.0, -1.0, 1.0]
vx = [0, 0, 0]
c = [1.0, 0.5]
r = [1.0, 1.0]
w = 1.0
iterations = 3

particles = init_particles(x, y, vx)

for iter_num in range(iterations):
    particles = iterate(particles, c, r, w, 1)
    print_status(iter_num, particles)
