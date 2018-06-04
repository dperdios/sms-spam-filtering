#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
from collections import Counter
from utils import load_spam_dataframe, savefig_kwargs
import matplotlib.pyplot as plt
from wordcloud import WordCloud

###############################################################################
# Load test set and main settings
###############################################################################
# Load dataset
filepath = os.path.join(os.pardir, 'datasets', 'spam.csv')
df = load_spam_dataframe(filepath)

# Save settings
flag_save_fig = False
save_folder = os.path.join(os.pardir, 'results', 'data-exploration')
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

###############################################################################
# Wordclouds
###############################################################################
wc_height, wc_width = (512, 1024)
wc_bckp_color = 'white'
wc_max_words = 400
wc_max_font_size = 60
random_state = 42
#   Creating wordcloud for spam
spam_df = df.loc[df['label'] == 'spam']
spam_wc = WordCloud(
    height=wc_height, width=wc_width, background_color=wc_bckp_color,
    max_words=wc_max_words, max_font_size=wc_max_font_size,
    random_state=random_state
).generate(str(spam_df['message']))
#   Creating wordcloud for ham
ham_df = df.loc[df['label'] == 'ham']
ham_wc = WordCloud(
    height=wc_height, width=wc_width, background_color=wc_bckp_color,
    max_words=wc_max_words, max_font_size=wc_max_font_size,
    random_state=random_state
).generate(str(ham_df['message']))

###############################################################################
# Word frequency
###############################################################################
# Visualization of the most frequent words
count1 = Counter(" ".join(df["message"]).split()).most_common(30)
df1 = pd.DataFrame.from_dict(count1)
df1 = df1.rename(columns={0: "words", 1 : "count"})

###############################################################################
# Plots
###############################################################################
wc_figsize = (12, 10)
fig_spam = plt.figure(figsize=wc_figsize)
ax = fig_spam.add_subplot(111)
ax.imshow(spam_wc)
ax.set_axis_off()

fig_ham = plt.figure(figsize=wc_figsize)
ax = fig_ham.add_subplot(111)
ax.imshow(ham_wc)
ax.set_axis_off()

fig_freq = plt.figure()
# fig = plt.figure(1, figsize=(10,12))
ax = fig_freq.add_subplot(111)
df1.plot.bar(ax=ax, legend=False)
# ax.set_xticks(y_pos, list(df1["words"]))
xticks = np.arange(len(df1["words"]))
ax.set_xticks(xticks)
ax.set_xticklabels(df1["words"])
ax.set_ylabel('Number of occurences')

if flag_save_fig:
    fig_spam.savefig(
        os.path.join(save_folder, 'wordcloud_spam.pdf'), **savefig_kwargs
    )
    fig_ham.savefig(
        os.path.join(save_folder, 'wordcloud_ham.pdf'), **savefig_kwargs
    )
    fig_freq.savefig(
        os.path.join(save_folder, 'freq_word.pdf'), **savefig_kwargs
    )
else:
    plt.show()
