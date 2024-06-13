import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


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


ax.set_xlim(-max_range, max_range)
ax.set_ylim(-max_range, max_range)
ax.set_zlim(-max_range, max_range)


ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title('monopole trajectories')
ax.legend()


plt.show()
