"""Build supplementary figures and their aggregate source values."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
from scipy.stats import norm
import seaborn as sns
import statsmodels.api as sm

HERE = Path(__file__).resolve().parent
DATA = HERE.parent
SIM = DATA / "processed_data"
OUT = HERE / "overleaf"
RESULTS = HERE / "results"
BLUE, ORANGE = "#6384B3", "#F08743"
AGE_KEYS = ["18-23", "24-30", "31-40", "41-50", "51-60", "60+"]
AGES = ["19–23", "24–30", "31–40", "41–50", "51–60", "61–79"]
plt.rcParams.update({"font.family":"Arial", "font.size":9, "axes.titlesize":10,
                     "axes.labelsize":9, "xtick.labelsize":8, "ytick.labelsize":8,
                     "legend.fontsize":8, "axes.linewidth":.7, "pdf.fonttype":42,
                     "mathtext.fontset":"stixsans", "savefig.facecolor":"white"})
OUT.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)


def panel(ax, letter, fontsize=12):
    ax.text(-.14, 1.055, letter, transform=ax.transAxes, weight="bold", fontsize=fontsize,
            verticalalignment="bottom", horizontalalignment="left")


def finish(fig, name):
    fig.savefig(OUT / f"ic-{name}.pdf", bbox_inches="tight", pad_inches=.06)
    plt.close(fig)


def csv(frame, name):
    frame.to_csv(RESULTS / f"{name}.csv", index=False)


def read_cohort():
    df = pd.read_csv(DATA / "cleaned_data_v2.csv", dtype={"user_id":str})
    final = pd.read_csv(DATA / "cleaned_data_1216.csv", usecols=["user_id","m_ratio"], dtype={"user_id":str})
    df = df.merge(final, on="user_id", how="inner", validate="one_to_one", suffixes=("","_check"))
    assert len(df) == 653486 and np.allclose(df.m_ratio, df.m_ratio_check)
    cutoff = 1 + norm.ppf(.1)*df.m_ratio.std(ddof=0)
    df["deep"] = (df.m_ratio < cutoff).astype(int)
    assert abs(df.deep.mean() - .3735076191) < 1e-8
    return df


def plot_early():
    series = pd.read_csv(RESULTS / "fig_s1_hourly_digitized.csv")
    labels = [("viewing_entropy", "Viewing entropy"), ("watch_time_per_video", "Mean watch time per video"),
              ("exposure_coverage", "Exposure coverage")]
    fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.42), layout="constrained", sharex=True)
    for letter, ax, (metric, ylabel) in zip("abc", axes, labels):
        frame = series.loc[series.metric == metric]
        assert frame.viewing_hour.tolist() == list(range(1,31))
        ax.plot(frame.viewing_hour, frame["mean"], "o-", color=BLUE, ms=2.5, lw=1.15)
        ax.set(xticks=[1,10,20,30], xlabel="Viewing hour", ylabel=ylabel)
        ax.yaxis.set_major_locator(MaxNLocator(4))
        panel(ax, letter)
    finish(fig, "s1")


def group_disparities(df):
    price = ["1-1000","1000-2000","2000-4000","4000-5000","5000+"]
    df = df.copy()
    qlabels = ["Q1", "Q2", "Q3", "Q4", "Q5"]
    boundaries = []
    for source, target in [("entertainment_ratio", "ent_quintile"),
                           ("knowledge_ratio", "knowledge_quintile"),
                           ("news_ratio", "news_quintile"),
                           ("watch_entropy_1st", "entropy_quintile")]:
        df[target], bins = pd.qcut(df[source], 5, labels=qlabels, retbins=True)
        boundaries.extend((source, q, bins[i], bins[i+1]) for i, q in enumerate(qlabels))
    np.testing.assert_allclose(df[["entertainment_ratio", "knowledge_ratio", "news_ratio"]].sum(axis=1), 1)
    fig, axes = plt.subplots(3, 3, figsize=(180/25.4, 165/25.4), layout="constrained")
    # Panel d retains the demographic contrasts cited by the main manuscript.
    sets = [
        (0, "gender", ["F", "M"], ["Female", "Male"], "Gender", ""),
        (1, "mod_price_range", price, ["≤1", "1–2", "2–4", "4–5", ">5"],
         "Handset price", "Price band (thousand CNY)"),
        (2, "fre_city_level_eng", ["first", "second", "third", "fourth", "fifth"],
         ["1", "2", "3", "4", "5"], "City tier", "Tier"),
        (4, "fre_community_level_eng", ["City", "County", "Countryside"],
         ["Urban", "County", "Rural"], "Community type", ""),
        (5, "entropy_quintile", qlabels, qlabels, "Initial viewing diversity", "First-hour entropy quintile"),
        (6, "ent_quintile", qlabels, qlabels, "Entertainment", "Viewing-share quintile"),
        (7, "knowledge_quintile", qlabels, qlabels, "Knowledge", "Viewing-share quintile"),
        (8, "news_quintile", qlabels, qlabels, "News", "Viewing-share quintile"),
    ]
    summary = []
    for k, column, keys, labels, title, xlabel in sets:
        g=df.groupby(column,observed=True).deep.agg(["size","mean"]).loc[keys]
        assert g["size"].sum() == len(df)
        lo,hi=sm.stats.proportion_confint(g["mean"]*g["size"],g["size"],method="wilson")
        x=np.arange(len(keys)); ax=axes.flat[k]
        ax.bar(x, g["mean"]*100, width=.64, color=BLUE)
        ax.errorbar(x,g["mean"]*100,yerr=[(g["mean"]-lo)*100,(hi-g["mean"])*100],
                    fmt="none", ecolor="#29394D", capsize=2.3, lw=.8, zorder=4)
        ax.axhline(df.deep.mean()*100,ls=(0,(3,2)),color=".45",lw=.75)
        ax.set_xticks(x,labels)
        ax.set(xlim=(-.55,len(x)-.45), ylim=(0,70), yticks=[0,20,40,60],
               xlabel=xlabel, ylabel="Deep-IC risk (%)")
        ax.set_title(title, fontsize=7, pad=8)
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, color="#E9E9E9", lw=.5)
        ax.spines[["top","right"]].set_visible(False)
        ax.tick_params(length=3, width=.6, labelsize=7)
        ax.xaxis.label.set_size(7)
        ax.yaxis.label.set_size(7)
        panel(ax,"abcdefghi"[k],fontsize=8)
        for key,n,m,l,h in zip(keys,g["size"],g["mean"],lo,hi):summary.append((column,key,n,m,l,h))
    csv(pd.DataFrame(summary,columns=["dimension","group","n","risk","lo","hi"]),"fig_s2_groups")
    csv(pd.DataFrame(boundaries,columns=["measure","quintile","lower","upper"]),"fig_s2_quintiles")
    ax=axes.flat[3]
    thresholds=np.array([.5,.35,.25,.2,.15,.1,.05,.025,.005])
    pairs=[("age_range","60+","18-23","Age",BLUE),("gender","F","M","Gender","#497E6D"),
           ("mod_price_range",price[0],price[-1],"Handset price",ORANGE)]
    curves=[]
    for name,g1,g0,label,color in pairs:
        values=[]
        for q in thresholds:
            y=df.m_ratio < 1+norm.ppf(q)*df.m_ratio.std(ddof=0)
            gap=(y[df[name]==g1].mean()-y[df[name]==g0].mean())/y.mean()
            values.append(gap)
            curves.append((label,q,gap))
        ax.plot(1-thresholds,values,marker="o",ms=2.8,color=color,lw=1,label=label)
    ax.axhline(0,color=".7",lw=.7)
    ax.legend(frameon=False,fontsize=7,loc="upper left",handlelength=1.3,handletextpad=.4)
    ax.set(xlim=(.49,1.005), xticks=[.5,.75,1], ylim=(-.15,.9), yticks=[0,.4,.8],
           xlabel="Threshold stringency, 1 − α", ylabel="Relative disparity")
    ax.set_title("Demographic contrasts", fontsize=7, pad=8)
    ax.spines[["top","right"]].set_visible(False)
    ax.tick_params(length=3, width=.6, labelsize=7)
    ax.xaxis.label.set_size(7)
    ax.yaxis.label.set_size(7)
    panel(ax,"d",fontsize=8)
    csv(pd.DataFrame(curves,columns=["dimension","tail_probability","normalised_gap"]),"fig_s2_contrasts")
    fig.savefig(OUT / "ic-s2.pdf")
    plt.close(fig)


def regression(df):
    x=pd.get_dummies(df,columns=["gender","fre_city_level_eng","fre_community_level_eng"],dtype=float)
    x["mod_price_range_conti"]=x["price_conti"]
    social=["mod_price_range_conti","fre_city_level_eng_first","fre_city_level_eng_fourth",
            "fre_city_level_eng_second","fre_city_level_eng_third","fre_community_level_eng_Countryside",
            "fre_community_level_eng_County"]
    interest=["entertainment_ratio","news_ratio","watch_entropy_1st"]
    models={"Age only":[],"Gender":["gender_F"],"Socioeconomic":social,"Interests":interest,
            "All covariates":["gender_F"]+social+interest}
    rows=[]
    full_coefficients = []
    for label,extra in models.items():
        X=sm.add_constant(x[["age_conti"]+extra])
        for metric in ["risk","depth"]:
            if metric=="risk":result=sm.Logit(x.deep,X).fit(disp=False,cov_type="HC0")
            else:result=sm.OLS(x.m_ratio,X).fit(cov_type="HC0")
            if label == "All covariates":
                intervals = result.conf_int()
                for term, coefficient in result.params.items():
                    full_coefficients.append((metric, term, coefficient, *intervals.loc[term], len(x)))
            v=np.r_[result.params["age_conti"],result.conf_int().loc["age_conti"].to_numpy()]
            if metric=="risk":v=np.exp(v)
            rows.append((label,metric,len(x),*v))
    csv(pd.DataFrame(full_coefficients, columns=["model","term","estimate","lo","hi","n"]), "regression_full")
    csv(pd.DataFrame(rows,columns=["adjustment","metric","n","estimate","lo","hi"]),"table_s2")


def behaviour(df):
    p=pd.read_pickle(SIM/"age_range_all_dist.pkl")
    fig,axes=plt.subplots(2,3,figsize=(6.8,4.35),sharex=True,sharey=True,layout="constrained")
    rows=[]
    mats=[p[key]*100 for key in AGE_KEYS]
    vmax=np.ceil(max(m.max() for m in mats))
    for i,(ax,age,m) in enumerate(zip(axes.flat,AGES,mats)):
        assert np.isclose(m.sum(),100)
        assert p["age_dist"][i]==(df.age_range==AGE_KEYS[i]).sum()
        im=ax.imshow(m,origin="lower",cmap="Blues",vmin=0,vmax=vmax,interpolation="nearest")
        ax.set_title(f"{age} years",fontsize=9)
        ax.set_xticks([0,3,6,9]);ax.set_yticks([0,3,6,9])
        if i>=3:ax.set_xlabel("Exploitation score")
        if i%3==0:ax.set_ylabel("Exploration score")
        panel(ax,"abcdef"[i])
        rows.extend((age,a,b,m[a,b]) for a,b in np.ndindex(m.shape))
    fig.colorbar(im,ax=axes.ravel().tolist(),shrink=.78,label="Users within age group (%)")
    csv(pd.DataFrame(rows,columns=["age","exploration","exploitation","percent"]),"fig_s3")
    finish(fig,"s3")


def calibration():
    from model_diagram import draw_workflow, save_icon_positions
    obs = pd.read_pickle(SIM / "heatmap_em_new.pkl")
    sim = pd.read_pickle(SIM / "heatmap_model.pkl")["mat"]
    corners = [(0,0),(0,9),(9,0),(9,9)]
    workflow_left, workflow_right = .075, .970
    workflow_width = (workflow_right-workflow_left)*180
    workflow_scale = workflow_width/132
    workflow_height = 37.9*workflow_scale
    figure_height = workflow_height+3.2
    fig = plt.figure(figsize=(180/25.4,figure_height/25.4))
    workflow = fig.add_axes([workflow_left,1.6/figure_height,
                            workflow_width/180,workflow_height/figure_height])
    icons = draw_workflow(workflow,scale=workflow_scale)
    fig.savefig(OUT / "ic-model.pdf")
    save_icon_positions(workflow,icons,HERE / "results/workflow_icon_positions.json")
    plt.close(fig)

    figure_height = 65
    fig = plt.figure(figsize=(180/25.4,figure_height/25.4))
    axes = [fig.add_axes([left,16.2/figure_height,.235,42.3/figure_height]) for left in [.075,.405,.735]]
    with plt.rc_context({"font.size":7,"axes.labelsize":7,"xtick.labelsize":7,
                         "ytick.labelsize":7,"legend.fontsize":7}):
        heat = sns.color_palette("vlag_r",as_cmap=True)
        im = axes[0].imshow(obs*100,origin="lower",cmap=heat,
                            norm=TwoSlopeNorm(vmin=23.2,vcenter=37.4,vmax=56.9))
        cax = fig.add_axes([.094,6.912/figure_height,.197,1.944/figure_height])
        bar = fig.colorbar(im,cax=cax,orientation="horizontal",ticks=[25,37.4,55])
        bar.set_label("Deep-IC risk (%)",fontsize=7,labelpad=2)
        bar.ax.tick_params(labelsize=7,length=2,width=.6,pad=1)
        for a,b in corners:
            axes[0].plot(b,a,marker="s",ms=7,mfc="none",mec="black",mew=.8)
        x = np.arange(4)
        axes[1].plot(x,[obs[c]*100 for c in corners],"o",c=BLUE,label="Empirical",ms=4)
        axes[1].plot(x,[sim[c]*100 for c in corners],"s",c=ORANGE,label="Simulated",ms=3.5)
        axes[1].set_xticks(x,["0,0","0,9","9,0","9,9"])
        axes[1].set(xlim=(-.35,3.35),ylim=(17,62),yticks=[20,40,60])
        axes[1].set_xlabel("Exploration, exploitation",fontsize=7)
        axes[1].set_ylabel("Deep-IC risk (%)",fontsize=7,labelpad=2)
        axes[1].legend(frameon=False,fontsize=6.5,ncol=2,loc="upper center",
                       bbox_to_anchor=(.5,-.25),handlelength=1,handletextpad=.3,columnspacing=.7)
        resid = (sim-obs)*100
        for c in corners:
            resid[c] = np.nan
        limit = int(np.ceil(np.nanmax(np.abs(resid))))
        cmap = LinearSegmentedColormap.from_list("signed_error",
            ["#684485","#AC8AB8","#FFFFFF","#7CB7AC","#16766D"],N=257).with_extremes(bad="#E8E8E8")
        im = axes[2].imshow(resid,origin="lower",cmap=cmap,
                            norm=TwoSlopeNorm(vcenter=0,vmin=-limit,vmax=limit))
        for a,b in corners:
            axes[2].plot(b,a,marker="x",c="#666666",ms=3.5,mew=.7)
        cax = fig.add_axes([.754,6.912/figure_height,.197,1.944/figure_height])
        bar = fig.colorbar(im,cax=cax,orientation="horizontal",ticks=[-limit,0,limit])
        bar.set_ticklabels([f"−{limit}","0",f"+{limit}"])
        bar.set_label("Error (percentage points)",fontsize=7,labelpad=2)
        bar.ax.tick_params(labelsize=7,length=2,width=.6,pad=1)
        for ax in [axes[0],axes[2]]:
            ax.set_xticks([0,3,6,9]); ax.set_yticks([0,3,6,9])
            ax.set_xlabel("Exploitation score",fontsize=7,labelpad=2)
            ax.set_ylabel("Exploration score",fontsize=7,labelpad=2)
        for i,(ax,title) in enumerate(zip(axes,["Empirical risk","Four calibration corners","Model − empirical"])):
            ax.set_title(title,fontsize=7,pad=6)
            ax.tick_params(length=2.5,width=.6,labelsize=7,pad=2)
            fig.text([.013,.343,.673][i],61.344/figure_height,"abc"[i],fontsize=8,weight="bold")
    csv(pd.DataFrame([(a,b,obs[a,b],sim[a,b],(a,b) in corners) for a,b in np.ndindex(obs.shape)],
        columns=["exploration","exploitation","empirical","model","corner"]),"fig_s4")
    fig.savefig(OUT / "ic-calibration.pdf")
    plt.close(fig)


def interventions():
    cells=sorted(pd.read_pickle(SIM/"count_explore_exploit.pkl"),key=lambda r:-r[2])[:5]
    grid=pd.read_pickle(SIM/"intervention_beta.pkl").groupby(["sigma","alpha","beta"]).indicator.agg(["mean","size"])
    shades=["#2F608D","#477CA8","#678FBA","#E49146","#A45D15"]
    fig,axes=plt.subplots(1,3,figsize=(7.0,2.65),sharey=True,layout="constrained")
    rows=[]
    for mode,ax in enumerate(axes):
        for (s,a,_),color in zip(cells,shades):
            x=range(4 if mode==0 else 5); vals=[]
            for step in x:
                key=(s,a,7-step) if mode==0 else ((s,max(a-step,0),7) if mode==1 else (min(s+step,9),a,7))
                r=grid.loc[key];vals.append(r["mean"]*100)
                rows.append((mode+1,s,a,step,*key,r["size"],r["mean"]))
            ax.plot(list(x),vals,"o-",lw=1.2,ms=3.2,label=f"({s}, {a})",color=color)
        ax.set_xticks(list(x));ax.set_xlabel(["Reduction in β (units)","Exploitation steps down","Exploration steps up"][mode])
        panel(ax,"abc"[mode])
    axes[0].set_ylabel("Simulated users in deep ICs (%)")
    fig.legend(*axes[0].get_legend_handles_labels(),loc="outside lower center",ncol=5,
               title="Baseline (exploration, exploitation)",frameon=False,columnspacing=1.4)
    csv(pd.DataFrame(rows,columns=["panel","origin_s","origin_a","step","s","a","beta","n","risk"]),"fig_s5")
    finish(fig,"s5")
    age_curves = []
    for parameter in ("beta", "alpha", "sigma"):
        records = pd.read_pickle(SIM / f"varying_{parameter}.pkl")
        dose = "increased_sigma" if parameter == "sigma" else f"reduced_{parameter}"
        settings = records.groupby(dose).indicator.agg(["size", "mean"]).reset_index()
        settings.columns = ["archived_setting", "n", "risk"]
        settings["parameter"] = parameter
        age_curves.append(settings)
    csv(pd.concat(age_curves, ignore_index=True), "fig4_oldest_interventions")


if __name__=="__main__":
    d=read_cohort()
    age_table=d.groupby("age_range").size().loc[AGE_KEYS]
    csv(pd.DataFrame({"age":AGES,"n":age_table.values,"percent":age_table.values/len(d)*100}),"table_s1")
    behaviour(d)
    calibration()
    interventions()
    group_disparities(d)
    plot_early()
    regression(d)
