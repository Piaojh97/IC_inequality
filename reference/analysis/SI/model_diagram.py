"""Mechanistic overview: initialization, behaviour, recommendation and updating."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.patches as patches

BLUE, ORANGE = "#6384B3", "#F08743"
INK, GREY = "#222222", "#7F8993"
HERE = Path(__file__).resolve().parent


def draw_workflow(ax, scale=1):
    ax.set(xlim=(0,132), ylim=(0,37.9), aspect="equal")
    ax.axis("off")

    def label(x,y,s,**kw):
        kw.setdefault("fontsize",9.5)
        kw.setdefault("zorder",5)
        ax.text(x,y,s,ha="center",va="center",color=INK,**kw)

    def arrow(a,b,color=INK,lw=.9,style="-",rad=0):
        ax.add_patch(patches.FancyArrowPatch(a,b,arrowstyle="-|>",
            mutation_scale=8,linewidth=lw,color=color,linestyle=style,
            connectionstyle=f"arc3,rad={rad}",shrinkA=0,shrinkB=0,zorder=3))

    def region(x,y,w,h,fill,edge="#C7CDD3",lw=.55,radius=1):
        ax.add_patch(patches.FancyBboxPatch((x,y),w,h,
            boxstyle=f"round,pad=0,rounding_size={radius}",
            facecolor=fill,edgecolor=edge,linewidth=lw,zorder=-10))

    # Empirical initialization connects to both actors. Distributional and
    # numerical details are explained in the note and algorithm, not this panel.
    region(18,30.5,96,7.4,"#F1F2F3")
    label(31,34.2,"Initialization",weight="bold")
    p0=pd.read_csv(HERE/"source_data/model_initialization.csv").mean_first_hour_share.to_numpy()
    xs=45+np.arange(35)*.58
    ax.bar(xs,p0/p0.max()*4.6,bottom=31.7,width=.42,color=BLUE,linewidth=0)
    ax.plot([44.5,65.3],[31.7,31.7],color=INK,lw=.45)
    arrow((68.5,34.2),(74.5,34.2),GREY,lw=.8)
    label(95.5,34.2,"Initial preferences")
    dash=(0,(3,2))
    ax.plot([95.5,95.5],[30.5,28],color=GREY,lw=.8,ls=dash,zorder=2)
    ax.plot([8.25,121.5],[28,28],color=GREY,lw=.8,ls=dash,zorder=2)
    arrow((8.25,28),(8.25,17.8),GREY,lw=.8,style=dash)
    arrow((121.5,28),(121.5,20.1),GREY,lw=.8,style=dash)
    white=dict(facecolor="white",edgecolor="none",pad=.5)
    label(33,28,r"$\mathbf{x}$",fontsize=8.8,bbox=white)
    label(111,28,r"$\mathbf{u}(0)$",fontsize=8.8,bbox=white)

    # One user supplies both behavioural routes. Colour identifies the two
    # mechanisms; only the parameters that distinguish them remain on the figure.
    region(0,0,56,23.5,"#FFFFFF")
    label(28,21.2,"User behaviour",weight="bold")
    region(21,10.8,33,6.5,"#FDEDDD",edge="none",lw=0)
    region(21,2.2,33,6.5,"#E7EFF8",edge="none",lw=0)
    icons=[("user",3.5,7.3,9.5,9.5),
           ("item",59,15.5,10,7.3),
           ("recommender",116.5,9.6,10,9.5)]
    label(37.5,14.05,"Exploitation  α")
    label(37.5,5.45,"Exploration  σ")
    arrow((14.3,13.8),(20,14.05),ORANGE,lw=.9)
    arrow((14.3,9.1),(20,5.45),BLUE,lw=.9)

    # The update feeds the next recommendation round.
    region(72,0,60,23.5,"#F1F3F5")
    label(102,21.2,"Recommender",weight="bold")
    label(91,14.6,"Matching  β")
    label(121.5,8.1,r"$\mathbf{u}(t)$",fontsize=8.8)
    arrow((115,14.6),(102.7,14.6),INK,lw=.95)
    ax.plot([75,129],[6.5,6.5],color="#CFD4DA",lw=.55)
    label(97,3.2,"Preference update")
    ax.plot([113.5,121.5],[3.2,3.2],color=INK,lw=.95)
    arrow((121.5,3.2),(121.5,6.7),INK,lw=.95)

    # Solid arrows carry recommendations and the two behavioural contributions.
    ax.plot([91,91],[17.2,18.5],color=INK,lw=.95)
    arrow((91,18.5),(70.2,18.5),INK,lw=.95)
    label(64,25.3,"Recommendations",fontsize=8.5)
    arrow((57.9,18.5),(15.3,18.5),INK,lw=.95)
    arrow((55.1,12.1),(71,5.4),ORANGE,lw=1)
    label(64,13.4,"Feedback",fontsize=8.5)
    arrow((55.1,5.45),(71,3.3),BLUE,lw=1)

    # Keep the compact layout proportional when the whole panel is enlarged.
    for text in ax.texts:
        text.set_fontsize(text.get_fontsize()*scale)
    for line in ax.lines:
        line.set_linewidth(line.get_linewidth()*scale)
    for patch in ax.patches:
        patch.set_linewidth(patch.get_linewidth()*scale)
        if isinstance(patch,patches.FancyArrowPatch):
            patch.set_mutation_scale(patch.get_mutation_scale()*scale)
    return icons


def save_icon_positions(ax,icons,path):
    """Store PDF point coordinates for the separate, lossless composition step."""
    fig=ax.figure
    fig.canvas.draw()
    entries=[]
    for name,x,y,w,h in icons:
        lo=ax.transData.transform((x,y))/fig.dpi*72
        hi=ax.transData.transform((x+w,y+h))/fig.dpi*72
        entries.append({"name":name,"rect":[*lo,*(hi-lo)]})
    Path(path).write_text(json.dumps(entries,indent=2)+"\n")
