import jax.numpy as jnp
import numpy as np
import yaml
from box import Box
from matplotlib import pyplot as plt
from jax import random as random
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from src.dynamics import KS
from src.utils import plot_with_horizontal_colorbar

def main():
    
  def plot_multivalue(
    input,
    output,
    DIM,
    mesh,
    index_array,
    visualize_type: str = "PCA",
    plot_mesh: bool = False,
    fig_name: str = "",
  ):
    """
    visualization of the multivalue mapping
    input: [batch, input_dim]
    output: [batch, output_dim]

    """

    n_array = [512, 256, 128, 64, 32]
    
    # add neighborhood information to the input
    # new_input = jnp.zeros((input.shape[0], input.shape[1], DIM))
    # for j in range(DIM):
    #   new_input = new_input.at[...,j].set(jnp.roll(input, j, axis=1))

    output = output.reshape(-1, 1)
    dx = 10 * jnp.pi / 256
    # u_x
    u_x = (jnp.roll(input, 1, axis=1) - jnp.roll(input, -1, axis=1)) /dx/2
    u_x = u_x.reshape(-1, DIM)
    plt.scatter(u_x, output, s=.2)
    plt.savefig('results/fig/ux_tau_n10.pdf')
    plt.clf()
    # u_xx
    u_xx = ((jnp.roll(input, 1, axis=1) + jnp.roll(input, -1, axis=1)) -
     2 * input) / dx**2
    u_xx = u_xx.reshape(-1, DIM)
    plt.scatter(u_xx, output, s=.2)
    plt.savefig('results/fig/uxx_tau_n10.pdf')
    plt.clf()
    # u_xxxx
    u_xxxx = ((jnp.roll(input, 2, axis=1) + jnp.roll(input, -2, axis=1)) -\
      4*(jnp.roll(input, 1, axis=1) + jnp.roll(input, -1, axis=1)) + 6 * input) /\
      dx**4
    u_xxxx = u_xxxx.reshape(-1, DIM)
    plt.scatter(u_xxxx, output, s=.2)
    plt.savefig('results/fig/uxxxx_tau_n10.pdf')
    plt.clf()
    
    return

    # standardization of the input and output
    input = input.reshape(-1, input.shape[-1])
    output = output.reshape(-1, output.shape[-1])
    input_means = np.mean(input, axis=0)
    input_stds = np.std(input, axis=0)
    input = (input - input_means) / input_stds
    output_means = np.mean(output, axis=0)
    output_stds = np.std(output, axis=0)
    output = (output - output_means) / output_stds

    fig, axs = plt.subplots(4, len(n_array
                                   ), figsize=(8, 6))
    input = input.reshape(-1, input.shape[-1])
    output = output.reshape(-1, output.shape[-1])
    output_ = output.copy()
    if plot_mesh:
      mesh = mesh.reshape(-1, mesh.shape[-1])

    # TODO: we may need to move this to the outside of the function
    # section = 0
    # one_section = 138600
    # num = 100000
    # input = input[one_section * section:one_section * section + num]
    # output = output[one_section * section:one_section * section + num]

    if input.shape[-1] == 1:
      next
    elif visualize_type == "PCA":
      # PCA visualization of stress
      pca = PCA(n_components=2)
      if output.shape[-1] > 1:
        output = pca.fit_transform(output)
      input = pca.fit_transform(input)
    elif visualize_type == "tSNE":
      # tSNE visualization
      print("start doing tSNE...")
      output = TSNE(
        n_components=2, learning_rate="auto", init="random", perplexity=3
      ).fit_transform(output)

    for j in range(len(n_array)):
      n = n_array[j]
      print(n)
      nbrs = NearestNeighbors(n_neighbors=n, algorithm="ball_tree").fit(input)
      distances, indices = nbrs.kneighbors(input)

      for i in range(4):
        index = index_array[i]

        if input.shape[-1] == 1 and output.shape[-1] == 1:
          # for 1D input we need to use different visualization
          _ = axs[i][j].scatter(
            input[indices[index]],
            output[indices[index]],
            s=0.1,
            # coloring by the value of the stress components
            # c=output_[indices[index], 1]
            # coloring by the distance to the centered point
            c=np.abs(input[indices[index]] - input[index])
          )
          xmin = np.min(input[indices[index]])
          xmax = np.max(input[indices[index]])
          axs[i][j].set_xticks(
            ticks=[xmin, xmax],
            labels=["{:.2e}".format(xmin), "{:.2e}".format(xmax)]
          )
          ymin = np.min(output[indices[index]])
          ymax = np.max(output[indices[index]])
          axs[i
              ][j].set_yticks(
            ticks=[ymin, ymax],
            labels=["{:.2e}".format(ymin), "{:.2e}".format(ymax)]
          )
          axs[i][j].tick_params(axis="x", labelsize=5)
          axs[i][j].tick_params(axis="y", labelsize=5)
        
        elif output.shape[-1] == 1:
          _ = axs[i][j].scatter(
            np.linalg.norm(input[indices[index]] - input[index], axis=1),
            output[indices[index]],
            s=0.1,
            # coloring by the value of the stress components
            # c=output_[indices[index], 1]
            # coloring by the distance to the centered point
            c=np.linalg.norm(input[indices[index]] - input[index], axis=1)
          )
          xmin = np.min(
            np.linalg.norm(input[indices[index]] - input[index], axis=1)
          )
          xmax = np.max(
            np.linalg.norm(input[indices[index]] - input[index], axis=1)
          )
          axs[2][j].set_xticks(
            ticks=[xmin, xmax],
            labels=["{:.2e}".format(xmin), "{:.2e}".format(xmax)]
          )
          ymin = np.min(output[indices[index]])
          ymax = np.max(output[indices[index]])
          axs[2][j].set_yticks(
            ticks=[ymin, ymax],
            labels=["{:.2e}".format(ymin), "{:.2e}".format(ymax)]
          )
          axs[2][j].tick_params(axis="x", labelsize=5)
          axs[2][j].tick_params(axis="y", labelsize=5)

        else:
          # visualize the 2d density
          # density, x_edges, y_edges = np.histogram2d(
          #   stress[indices[index],0], stress[indices[index],1], bins=50)
          # axs[i][j].hexbin(
          #   stress[indices[index],0], stress[indices[index],1],
          #   gridsize=50, cmap="inferno")
          # axs[i//2][i%2].imshow(density.T, extent=[x_edges[0], x_edges[-1],
          # y_edges[0], y_edges[-1]])#, origin="lower", cmap="inferno",
          # extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]])
          _ = axs[2 * i][j].scatter(
            output[indices[index], 0],
            output[indices[index], 1],
            s=0.1,
            # coloring by the value of the stress components
            # c=output_[indices[index], 1]
            # coloring by the distance to the centered point
            c=np.linalg.norm(input[indices[index]] - input[index], axis=1)
          )
          xmin = np.min(output[indices[index], 0])
          xmax = np.max(output[indices[index], 0])
          axs[i * 2][j].set_xticks(
            ticks=[xmin, xmax],
            labels=["{:.2e}".format(xmin), "{:.2e}".format(xmax)]
          )
          ymin = np.min(output[indices[index], 1])
          ymax = np.max(output[indices[index], 1])
          axs[i * 2][j].set_yticks(
            ticks=[ymin, ymax],
            labels=["{:.2e}".format(ymin), "{:.2e}".format(ymax)]
          )
          axs[i * 2][j].tick_params(axis="x", labelsize=5)
          axs[i * 2][j].tick_params(axis="y", labelsize=5)

        if plot_mesh:
          # plot the position of the points
          coord1 = 1
          coord2 = 2
          mesh_indices = indices[index] % one_section
          _ = axs[2 * i + 1][j].scatter(
            mesh[mesh_indices, coord1],
            mesh[mesh_indices, coord2],
            s=0.1,
            # coloring by the value of the stress components
            c=output_[indices[index], 1]
            # coloring by the distance to the centered point
            # c=np.linalg.norm(input[indices[index]]-input[index], axis=1)
          )
          xmin = np.min(mesh[mesh_indices, coord1])
          xmax = np.max(mesh[mesh_indices, coord1])
          axs[i * 2 + 1][j].set_xticks(
            ticks=[xmin, xmax],
            labels=["{:.2e}".format(xmin), "{:.2e}".format(xmax)]
          )
          ymin = np.min(mesh[mesh_indices, coord2])
          ymax = np.max(mesh[mesh_indices, coord2])
          axs[i * 2 + 1][j].set_xticks(
            ticks=[ymin, ymax],
            labels=["{:.2e}".format(ymin), "{:.2e}".format(ymax)]
          )
          axs[i * 2 + 1][j].tick_params(axis="x", labelsize=5)
          axs[i * 2 + 1][j].tick_params(axis="y", labelsize=5)

        elif input.shape[-1] > 1 and output.shape[-1] > 1:
          # if not plot the mesh, then plot the visualization of the input
          _ = axs[2 * i + 1][j].scatter(
            input[indices[index], 0],
            input[indices[index], 1],
            s=0.1,
            # coloring by the value of the stress components
            # c=output_[indices[index], 1]
            # coloring by the distance to the centered point
            c=np.linalg.norm(input[indices[index]] - input[index], axis=1)
          )
          xmin = np.min(input[indices[index], 0])
          xmax = np.max(input[indices[index], 0])
          axs[i * 2 + 1][j].set_xticks(
            ticks=[xmin, xmax],
            labels=["{:.2e}".format(xmin), "{:.2e}".format(xmax)]
          )
          ymin = np.min(input[indices[index], 1])
          ymax = np.max(input[indices[index], 1])
          axs[i * 2 + 1][j].set_yticks(
            ticks=[ymin, ymax],
            labels=["{:.2e}".format(ymin), "{:.2e}".format(ymax)]
          )
          axs[i * 2 + 1][j].tick_params(axis="x", labelsize=5)
          axs[i * 2 + 1][j].tick_params(axis="y", labelsize=5)

        Lip = np.max(
          np.linalg.norm(
            output_[np.delete(indices[index], np.where(indices[index] == index))]
            - output_[index],
            axis=1
          ) / np.linalg.norm(
            input[np.delete(indices[index], np.where(indices[index] == index))] -
            input[index],
            axis=1
          )
        )
        print("n = {}, index = {}, Lip = {:.2e}".format(n, index, Lip))
        # if i == 0 and j == 0:
        #   fig.colorbar(im, orientation="vertical")

    fig.suptitle(
      ",".join(str(index) for index in index_array) + ";" +
      ",".join(str(n) for n in n_array)
    )
    if fig_name is None:
      plt.savefig("results/fig/multivalue.pdf")
    else:
      plt.savefig("results/fig/" + fig_name + ".pdf")
    plt.clf()

    return
  
  DIM = 1
  plot_multivalue(
    input,
    output,
    DIM,
    mesh=None,
    index_array=[2000, 4000, 6000, 8000],
    fig_name=f'{DIM}_neigh_multivalue'
  )

if __name__ == "__main__":
  nu = .1
  c = .1

  data = np.load('data/ks/nu{:.1f}_c{:.1f}_n{}.npz'.format(nu, c, 10))
  input = data["input"]
  output = data["output"]
  main()