import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D, art3d
from matplotlib.patches import Circle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


df = pd.read_csv('trajectories.csv')


df_sol1 = df[df['monopole#'] == 1]
df_sol2 = df[df['monopole#'] == 2]
combined_df = pd.concat([df_sol1, df_sol2])


x_min, x_max = combined_df['x'].min(), combined_df['x'].max()
y_min, y_max = combined_df['y'].min(), combined_df['y'].max()
z_min, z_max = combined_df['z'].min(), combined_df['z'].max()

x_range = max(abs(x_min), abs(x_max))
y_range = max(abs(y_min), abs(y_max))
z_range = max(abs(z_min), abs(z_max))

max_range = max(x_range, y_range, z_range)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


ax.plot(df_sol1['x'], df_sol1['y'], df_sol1['z'], label='monopole 1', color='blue')
ax.plot(df_sol2['x'], df_sol2['y'], df_sol2['z'], label='monopole 2', color='red')

radii = [3.10, 5.05, 8.85, 12.25, 29.9, 37.1, 44.3, 51.4]
for radius in radii:
    for z in range(int(z_min), int(z_max), 5):  # Adjust the step size as needed
        circle = Circle((0, 0), radius, color='green', fill=False, alpha=0.2)
        ax.add_patch(circle)
        art3d.pathpatch_2d_to_3d(circle, z=z, zdir="z")


ax.set_xlim(-max_range, max_range)
ax.set_ylim(-max_range, max_range)
ax.set_zlim(-max_range, max_range)

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title('monopole trajectories')
ax.legend()


plt.show()
