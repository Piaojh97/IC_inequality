"""Describe the final cohort and empirical content inputs for Supplementary Note 1."""
from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from build_figures import read_cohort, BLUE, ORANGE, AGE_KEYS, AGES, HERE, OUT, RESULTS

MM = 1 / 25.4
PREVIEW = HERE / 'previews' / 'note1'
PREVIEW.mkdir(parents=True, exist_ok=True)
CONTENT_COLS = ['entertainment_ratio', 'knowledge_ratio', 'news_ratio']
CONTENT_NAMES = ['Entertainment', 'Knowledge', 'News']
CONTENT_COLORS = [BLUE, ORANGE, '#8B969E']


def finish(fig, name):
    fig.savefig(OUT / f'ic-{name}.pdf')
    fig.savefig(PREVIEW / f'{name}.png', dpi=200)
    plt.close(fig)


def style(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(length=3, width=.65)


def cohort_profile(df):
    fig = plt.figure(figsize=(180 * MM, 112 * MM))
    grid = fig.add_gridspec(2, 3, left=.09, right=.985, bottom=.12,
                           top=.93, wspace=.44, hspace=.64)
    axes = [fig.add_subplot(grid[0, :2]), fig.add_subplot(grid[0, 2])]
    axes += [fig.add_subplot(grid[1, i]) for i in range(3)]
    specs = [
        ('age_range', AGE_KEYS, AGES, 'Age', 'Age (years)'),
        ('gender', ['F', 'M'], ['Female', 'Male'], 'Gender', ''),
        ('mod_price_range', ['1-1000', '1000-2000', '2000-4000', '4000-5000', '5000+'],
         ['≤1', '1–2', '2–4', '4–5', '>5'], 'Handset price', 'Price band (thousand CNY)'),
        ('fre_city_level_eng', ['first', 'second', 'third', 'fourth', 'fifth'],
         ['1', '2', '3', '4', '5'], 'City tier', 'Tier'),
        ('fre_community_level_eng', ['City', 'County', 'Countryside'],
         ['Urban', 'County', 'Rural'], 'Community type', ''),
    ]
    rows = []
    for i, (ax, (column, keys, labels, title, xlabel)) in enumerate(zip(axes, specs)):
        g = df[column].value_counts().reindex(keys)
        assert g.sum() == len(df)
        values = g.to_numpy() / len(df) * 100
        ax.bar(np.arange(len(keys)), values, width=.62, color=BLUE)
        ax.set(xticks=np.arange(len(keys)), xticklabels=labels,
               xlabel=xlabel, ylabel='Users (%)', title=title,
               ylim=(0, max(values) * 1.25))
        for j, value in enumerate(values):
            ax.text(j, value + max(values) * .035, f'{value:.1f}',
                    ha='center', va='bottom', fontsize=8)
        for key, n in g.items():
            rows.append([column, key, int(n), 100 * n / len(df)])
        style(ax)
        ax.text(-.07 if i == 0 else -.16, 1.07, 'abcde'[i],
                transform=ax.transAxes, weight='bold', fontsize=11,
                ha='left', va='bottom')
    pd.DataFrame(rows, columns=['dimension', 'category', 'n', 'percent']).to_csv(
        RESULTS / 'note1_cohort_profile.csv', index=False)
    finish(fig, 'cohort-profile')


def load_topics():
    topics = pd.read_csv(HERE / 'source_data' / 'model_initialization.csv')
    labels_path = HERE / 'source_data' / 'topic_labels.csv'
    if labels_path.exists():
        labels = pd.read_csv(labels_path)
        topics = topics.merge(labels[['topic_id', 'topic_name']], on='topic_id',
                              how='left', validate='one_to_one')
        assert topics.topic_name.notna().all()
        assert topics.topic_name.is_unique
        assert set(topics.topic_id) == set(labels.topic_id)
        labels_verified = True
    else:
        topics['topic_name'] = topics.topic_id.map(lambda x: f'Topic {x}')
        labels_verified = False
    items = pd.read_pickle(HERE.parent / 'server_backup/fine_grained/Data/items_num.pkl')
    topics = topics.merge(items.rename(columns={'first_level_category_id': 'topic_id'}),
                          on='topic_id', validate='one_to_one')
    np.testing.assert_allclose(topics.candidate_topic_probability,
                               topics.items_num / topics.items_num.sum())
    np.testing.assert_allclose(topics[['candidate_topic_probability', 'mean_first_hour_share']].sum(), 1)
    topics = topics.sort_values('items_num', ascending=False, kind='stable').reset_index(drop=True)
    return topics, labels_verified


def content_profile(df, topics):
    means = df.groupby('age_range')[CONTENT_COLS].mean().reindex(AGE_KEYS)
    np.testing.assert_allclose(df[CONTENT_COLS].sum(axis=1), 1)
    means.to_csv(RESULTS / 'note1_content_age_means.csv')
    topics[['topic_id', 'topic_name', 'items_num', 'candidate_topic_probability',
            'mean_first_hour_share']].to_csv(RESULTS / 'note1_topic_profile.csv', index=False)
    fig = plt.figure(figsize=(180 * MM, 190 * MM))
    a = fig.add_axes([.24, .42, .31, .53])
    b = fig.add_axes([.65, .42, .32, .53])
    c = fig.add_axes([.10, .095, .35, .24])
    d = fig.add_axes([.64, .095, .33, .24])
    positions = np.arange(len(topics))
    for ax, key, color, title, xlabel in [
        (a, 'candidate_topic_probability', BLUE, 'Video supply', 'Videos (%)'),
        (b, 'mean_first_hour_share', ORANGE, 'Initial consumption', 'First-hour viewing (%)'),
    ]:
        ax.barh(positions, topics[key].to_numpy() * 100, height=.68, color=color)
        ax.set(ylim=(len(topics)-.3, -.7), title=title, xlabel=xlabel)
        ax.tick_params(axis='y', length=0)
        style(ax)
    a.set(yticks=positions, yticklabels=topics.topic_name)
    a.tick_params(axis='y', labelsize=7.5, length=0, pad=3)
    b.set_yticks([])
    for col, name, color in zip(CONTENT_COLS, CONTENT_NAMES, CONTENT_COLORS):
        values, counts = np.unique(df[col].to_numpy() * 100, return_counts=True)
        c.step(values, 100 * np.cumsum(counts) / len(df), where='post',
               color=color, lw=1.2, label=name)
    c.set(xlim=(0, 100), ylim=(0, 100), xlabel='Individual viewing share (%)',
          ylabel='Cumulative users (%)', title='Content consumption')
    left = np.zeros(6)
    for col, name, color in zip(CONTENT_COLS, CONTENT_NAMES, CONTENT_COLORS):
        values = means[col].to_numpy() * 100
        d.barh(np.arange(6), values, left=left, height=.62, color=color, label=name)
        left += values
    d.set(yticks=np.arange(6), yticklabels=AGES, xlim=(0, 100),
          xlabel='Mean viewing share (%)', ylabel='Age (years)', title='Consumption by age')
    d.invert_yaxis()
    for letter, ax, x in [('a', a, -.72), ('b', b, -.12), ('c', c, -.19), ('d', d, -.20)]:
        style(ax)
        ax.text(x, 1.035 if letter in 'ab' else 1.075, letter,
                transform=ax.transAxes, weight='bold', fontsize=11, va='bottom')
    fig.legend(*c.get_legend_handles_labels(), loc='lower center', ncol=3,
               frameon=False, bbox_to_anchor=(.54, .005), fontsize=8)
    finish(fig, 'content-profile')


def correlation_matrix(topics):
    matrix = pd.read_csv(HERE / 'source_data/model_topic_correlations.csv', index_col=0)
    assert matrix.shape == (35, 35)
    order = topics.topic_id.map(lambda value: f'{value}_num').tolist()
    assert set(order) == set(matrix.index) == set(matrix.columns)
    matrix = matrix.loc[order, order]
    np.testing.assert_allclose(np.diag(matrix), 1)
    fig = plt.figure(figsize=(180 * MM, 153 * MM))
    ax = fig.add_axes([.23, .33, .69, .64])
    cmap = LinearSegmentedColormap.from_list('topiccorr', [BLUE, 'white', ORANGE])
    im = ax.imshow(matrix, cmap=cmap, vmin=-1, vmax=1, interpolation='nearest')
    ax.set(xticks=np.arange(35), yticks=np.arange(35),
           xticklabels=topics.topic_name, yticklabels=topics.topic_name)
    ax.tick_params(axis='x', labelrotation=90, labelsize=7.2, length=0, pad=3)
    ax.tick_params(axis='y', labelsize=7.2, length=0, pad=3)
    cb = fig.colorbar(im, cax=fig.add_axes([.32, .09, .50, .02]), orientation='horizontal')
    cb.set_ticks([-1, 0, 1]); cb.set_label('Topic correlation', fontsize=9)
    cb.outline.set_linewidth(.6)
    finish(fig, 'topic-correlations')


if __name__ == '__main__':
    cohort = read_cohort()
    topics, labels_verified = load_topics()
    cohort_profile(cohort)
    content_profile(cohort, topics)
    correlation_matrix(topics)
    audit = {'n': len(cohort), 'topics': len(topics),
             'topic_labels_verified': labels_verified,
             'video_count_total': int(topics.items_num.sum()),
             'top_three_video_share': float(topics.candidate_topic_probability.nlargest(3).sum()),
             'display_order_topic_ids': topics.topic_id.tolist(),
             'display_order': 'Descending video count; shared by both distributions and correlation axes',
             'plot_inputs_unchanged': True}
    (RESULTS / 'note1_profiles_audit.json').write_text(json.dumps(audit, indent=2)+'\n')
    print(audit)
