import matplotlib.pyplot as plt


def graphplt(x_value, fig_main, labelX, labelY):
      plt.figure(figsize=[8, 6],num=fig_main)
      plt.plot(x_value)
      plt.xlabel(labelX, fontsize=14)
      plt.ylabel(labelY, fontsize=14)
      plt.savefig(fig_main+".png")
      plt.show()
