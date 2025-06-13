from matplotlib import pyplot as plt
import imageio
import numpy as np

class ImageClicker:
    def __init__(self, image, num_points=2):
        self.image = image
        self.num_points = num_points
        self.count = 0
        self.points = []
        self.cid = self.image.figure.canvas.mpl_connect('button_press_event', self)
        
    def __call__(self, event):
        if (event.xdata is None) or (event.ydata is None):
            return
        print(event.xdata, event.ydata)
        plt.scatter(event.xdata, event.ydata, s=10, c='red')
        self.points.append(np.array([event.xdata, event.ydata]))
        self.count += 1
        self.image.figure.canvas.draw_idle()
        if self.count >= self.num_points:
            plt.pause(0.2)
            self.image.figure.canvas.mpl_disconnect(self.cid)
            plt.close()
