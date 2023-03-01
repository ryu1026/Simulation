from simulation import Simulate
import matplotlib.pyplot as plt

sim = Simulate(num_beads=5, grid_step=0.1)
help(sim.make_grid_and_gaussian_beads)
# gaussian_beads, gaussian_clip_beads = sim.make_grid_and_gaussian_beads()
# sim.draw_fluo_beads(gaussian_clip_beads)
#
# not_gaussian_beads, not_gaussian_clip_beads = sim.make_grid_and_not_gaussian_beads()
#
# sim.draw_not_gaussian_beads(not_gaussian_clip_beads)


# x,y = sim.triangle_spot(10,10)
# print("x_list_len:", len(x))
# fig, ax = plt.subplots()
# for i in range(4):
#     print(x[i], y[i])
#     ax.scatter(x[i], y[i])
#     ax.set_aspect('equal')
#
# plt.show()
