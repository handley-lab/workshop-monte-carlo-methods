fig, axes = plt.subplots(4, gridspec_kw={'height_ratios':[20,1,1,1]})

def update(ax):
    dist = vmf_dist(kappa_slider.val, phi0_slider.val, theta0_slider.val)
    pdf = dist.pdf(x)
    axes[0].cla()
    axes[0].contourf(phi, theta, pdf)

from matplotlib.widgets import Slider
kappa_slider = Slider(axes[1], r'$\kappa$', 0.1, 10, valinit=1)
theta0_slider = Slider(axes[2], r'$\theta$', 0, np.pi, valinit=np.pi/2)
phi0_slider = Slider(axes[3], r'$\phi$', 0, 2*np.pi, valinit=np.pi)

update(None)

kappa_slider.on_changed(update)
theta0_slider.on_changed(update)
phi0_slider.on_changed(update)
fig.tight_layout()
