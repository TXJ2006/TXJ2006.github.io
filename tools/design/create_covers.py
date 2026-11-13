from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
out=Path('source/images/editorial')
plt.rcParams.update({'font.family':'DejaVu Sans'})
def save(fig,name):
 p=out/(name+'.png');fig.savefig(p,dpi=160,facecolor=fig.get_facecolor(),bbox_inches=None,pad_inches=0);plt.close(fig)
 im=Image.open(p).convert('RGB');im.save(out/(name+'.webp'),quality=88);p.unlink()
# Mathematical illustrations are original plots, not reproductions of reference-site assets.
fig=plt.figure(figsize=(9,6),facecolor='#c6d9dc');ax=fig.add_subplot(projection='3d',computed_zorder=False)
u=np.linspace(0,2*np.pi,85);v=np.linspace(0,2*np.pi,45);u,v=np.meshgrid(u,v)
x=(2+.78*np.cos(v))*np.cos(u);y=(2+.78*np.cos(v))*np.sin(u);z=.78*np.sin(v)
ax.plot_surface(x,y,z,color='#47757b',edgecolor='#d9e6e0',linewidth=.32,alpha=1,shade=True)
ax.view_init(46,25);ax.set_box_aspect((1,1,.7));ax.set_axis_off();ax.set_facecolor('#c6d9dc');ax.dist=8.5
fig.subplots_adjust(0,0,1,1);save(fig,'geometry')
fig,ax=plt.subplots(figsize=(9,6),facecolor='#e4d7ca');ax.set_facecolor('#e4d7ca')
for k in np.linspace(-3,3,13):
 ax.plot([-4,4],[k-2,k+2],color='#a28770',lw=.65,alpha=.7)
 ax.plot([k-2,k+2],[-4,4],color='#a28770',lw=.65,alpha=.7)
for xx in np.arange(-4,5):
 for yy in np.arange(-4,5):
  X=xx+.5*yy;Y=.5*xx+yy
  ax.scatter(X,Y,s=18,c='#665b50',zorder=2)
ax.quiver([0,0],[0,0],[2,1],[1,2],angles='xy',scale_units='xy',scale=1,color=['#3f5e60','#776350'],width=.008)
ax.set(xlim=(-3.4,3.4),ylim=(-2.5,2.5),aspect='equal');ax.axis('off');fig.subplots_adjust(0,0,1,1);save(fig,'algebra')
fig,ax=plt.subplots(figsize=(9,6),facecolor='#dcded1');ax.set_facecolor('#dcded1')
x=np.linspace(-3.4,3.4,700)
for k in range(15):
 phase=k*.18; y=np.sin(x*1.3+phase)*np.exp(-x*x/20)+k*.13
 ax.plot(x,y,color=plt.cm.Greys(.36+k*.032),lw=1.2)
ax.set(xlim=(-3.4,3.4),ylim=(-1.5,3.5));ax.axis('off');fig.subplots_adjust(0,0,1,1);save(fig,'analysis')
fig,ax=plt.subplots(figsize=(9,6),facecolor='#d4dce6');ax.set_facecolor('#d4dce6')
t=np.linspace(0,2*np.pi,500)
for k in range(1,23):
 a=k/23;ax.plot((1+a*.7)*np.cos(t)+.3*np.cos(3*t),(.55+a*.7)*np.sin(t),color='#5c718e',lw=.7,alpha=.6)
ax.set(xlim=(-2.3,2.3),ylim=(-1.5,1.5),aspect='equal');ax.axis('off');fig.subplots_adjust(0,0,1,1);save(fig,'topology')
fig=plt.figure(figsize=(9,6),facecolor='#ccd9d5');ax=fig.add_subplot(projection='3d')
u=np.linspace(-2.1,2.1,13);v=np.linspace(-2.1,2.1,13);u,v=np.meshgrid(u,v)
ax.plot_surface(u,v,u,alpha=.68,color='#829d91',edgecolor='#edf0e7',linewidth=.7,shade=False)
ax.quiver(0,0,0,-1.7,0,1.7,color='#665d4f',linewidth=2.6,arrow_length_ratio=.1)
ax.quiver(0,0,0,1.7,0,1.7,color='#304f4a',linewidth=2.4,arrow_length_ratio=.1)
ax.quiver(0,0,0,0,2.3,0,color='#304f4a',linewidth=2.4,arrow_length_ratio=.1)
ax.view_init(27,135);ax.set_box_aspect((1,1,.85));ax.set_axis_off();ax.set_facecolor('#ccd9d5');fig.subplots_adjust(0,0,1,1);save(fig,'linear')
