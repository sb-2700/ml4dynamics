import matplotlib.pyplot as plt
import numpy as np
import re
import seaborn as sns

from ml4dynamics.utils.viz_utils import plot_bias_variance_comparison


if __name__ == "__main__":
  # plot_bias_variance_comparison("dnbc_nu1.0_c1.7_n10")
  filename = "results/data/dnbc_nu1.0_c1.7_n10.txt"
  data = []

  with open(filename, 'r') as file:
    for line in file:
      # Remove whitespace and split by comma
      line = line.strip()
      if line:  # Skip empty lines
        # Use regex to find all floating point numbers (including negative)
        numbers = re.findall(r'-?\d+\.?\d*', line)
        # Convert to float and add to data
        row = [float(num) for num in numbers]
        if len(row) == 4:  # Ensure we have exactly 4 numbers
          data.append(row)

  data = list(np.array(data).T) 
  data = {
    "baseline": data[0],
    "global correction": data[1],
    "global filter": data[2],
    "local correction": data[3],
  }
  sns.catplot(data=data, x="baseline", y="global correction", hue="global filter", kind="violin")
  plt.savefig("error_bar.png", dpi=300)
  # c = 1.6
  # l2_biases = [21.21507791, 18.67754587, 21.15538667, 18.73159481, 20.29150618]
  # l2_variances = [0.28219568, 0.30444609, 0.26415674, 0.31522692, 0.19112105]
  # first_moment_biases = [0.11764719, 0.02047228, 0.1409446, 0.02661518, 0.02826032]
  # first_moment_variances = [0.0223624, 0.01840503, 0.02321143, 0.02661518, 0.02826032]
  # second_moment_biases = [-0.41530047, -0.05377717, -0.3642728, 0.10012896, -0.21791786]
  # second_moment_variances = [0.08009741, 0.07427221, 0.06695318, 0.07001851, 0.10591935]
  # c = 1.8
  # l2_biases = [21.21507791, 18.67754587, 21.15538667, 18.73159481, 20.29150618]
  # l2_variances = [0.28219568, 0.30444609, 0.26415674, 0.31522692, 0.19112105]
  # first_moment_biases = [0.16624797, 0.10501413, 0.17826416, -0.04200691, 0.08885495]
  # first_moment_variances = [0.00825225, 0.00651592, 0.01092361, 0.00121675, 0.00626909]
  # second_moment_biases = [-0.59119097, -0.36376558, -0.5717222, 0.13099228, -0.26906323]
  # second_moment_variances = [0.02884514, 0.01962418, 0.03823023, 0.00418607, 0.01929106]
  # c = 1.8
  # l2_biases = [21.21507791, 18.67754587, 21.15538667, 18.73159481, 20.29150618]
  # l2_variances = [0.28219568, 0.30444609, 0.26415674, 0.31522692, 0.19112105]
  # first_moment_biases = [0.16624797, 0.10501413, 0.1409446, 0.02661518, 0.02826032]
  # first_moment_vari

  # nu = 1.0
  # l2_biases = [21.21507791, 18.67754587, 21.15538667, 18.73159481, 20.29150618]
  # l2_variances = [0.28219568, 0.30444609, 0.26415674, 0.31522692, 0.19112105]
  # first_moment_biases = [0.16624797, 0.10501413, 0.17826416, -0.04200691, 0.08885495]
  # first_moment_variances = [0.00825225, 0.00651592, 0.01092361, 0.00121675, 0.00626909]
  # second_moment_biases = [-0.59119097, -0.36376558, -0.5717222, 0.13099228, -0.26906323]
  # second_moment_variances = [0.02884514, 0.01962418, 0.03823023, 0.00418607, 0.01929106]
