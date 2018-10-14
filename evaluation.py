import os
import pandas as pd
import seaborn as sns
import pickle
import argparse
import pdb
import matplotlib.pylab as plt

sns.set_style('ticks')
parser = argparse.ArgumentParser()
parser.add_argument("--plot", help="Activate plot",type=bool, nargs='?',
                    const=True, default=False)
args = parser.parse_args()

if __name__ == '__main__':
    dir_path = os.getcwd()
    fig_path = dir_path + "/fig"

    # check if fig_path exists
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)

    df = pd.read_csv(dir_path +"/df_res.csv")

    keep_columns = ["C_Nom","C_Prenom","C_Date","C_Mention"]
    df["Score"] = df[keep_columns].sum(axis=1).values
    df_red_col = df[keep_columns + ["Score"]]

    df_bar = df_red_col.groupby("Score").apply(
    lambda x: x["C_Nom"].count()) / df_red_col.shape[0]

    df_grouped_score = df_red_col.groupby("Score").apply(lambda x:x.sum(axis=0))
    df_ext = pd.concat((df.Ext, df_red_col.Score), axis=1)

    # retain examples with errors
    for i in range(4):
        idx_retained = df_red_col[df_red_col.Score==i].index
        df_retained = df.iloc[idx_retained]
        file_list_crnn = df_retained.FileName.tolist()

        with open(dir_path +'/retained_file_score_%d' %i, 'wb') as fp:
            pickle.dump(file_list_crnn, fp)

    # To read the pickle file
    # with open(dir_path +'/retained_file_score_3', 'rb') as fp:
    #    list = pickle.load(fp)
    if bool(args.plot):
        fig, ax = plt.subplots(figsize=(4, 4))
        df_bar.plot.bar(ax=ax)
        title = "Score distribution"
        ax.set_title(title)
        ax.set_xticklabels(df_bar.index, rotation=0)
        ax.set_ylabel("Fraction")
        sns.despine()
        plt.tight_layout()
        fig.savefig("%s/%s.pdf" %(fig_path, title))

        fig, ax = plt.subplots(figsize=(4,4))
        df_error_rate = (df_red_col.shape[0] - df_red_col[keep_columns].sum(axis=0))/df_red_col.shape[0]
        df_error_rate.plot.bar(ax=ax)
        title = "Error distribution by columns"
        ax.set_title(title)
        ax.set_xticklabels(df_error_rate.index, rotation=0)
        ax.set_ylabel("Fraction")
        sns.despine()
        plt.tight_layout()
        fig.savefig("%s/%s.pdf" %(fig_path, title))

        fig, axes = plt.subplots(1,3, figsize=(10,3))
        for i, ax in enumerate(axes.flatten()):
            (df_red_col[df_red_col.Score==i+1].shape[0] - df_grouped_score.loc[i+1][keep_columns]).plot.bar(ax=ax)
            ax.set_xticklabels(df_grouped_score.loc[i+1].index[:-1], rotation=0)
            ax.set_title("Score %d" %(i+1))
            ax.set_ylabel("Missed count")
        plt.tight_layout()
        sns.despine()
        title = "missed_count_per_column"
        fig.savefig("%s/%s.pdf" %(fig_path, title))

        fig, ax = plt.subplots(figsize=(4,4))
        sns.barplot(data=df_ext, x='Ext', y='Score', ax=ax)
        ax.set_title("Score distribution vs ext")
        ax.set_ylabel('Mean Score')
        sns.despine()
        plt.tight_layout()
        title = "score_ditrib"
        fig.savefig("%s/%s.pdf" %(fig_path, title))
