import pandas as pd

class FileHandling():
    """
    class that handles the file system
    """

    def __init__(self, df_path = None):
        self.df_path = df_path

    def load_df(self):
        if self.df_path is None:
            from config import df_prod
            self.df = df_prod
        else:
            self.df = pd.read_csv(self.df_path)

        self.keep_cols = self.df.columns.tolist()[:3]
        self.touched = pd.Series([False]*self.df.shape[0],
                                 index = self.df.index)
        self.touched.name = 'touched'

    def lookup(self, entry):
        """
        look up the database if the entry has already been analysed
        Parameters:
        -----------
        entry: pandas dataframe
        """
        index_entry = self.df[self.df[self.keep_cols].
                            isin(entry).all(axis=1)].index
        self.touched.loc[index_entry] = True
        return self.df[self.keep_cols].isin(entry).all(axis=1).any()

    def save(self):
        """
        save df to save path in csv
        """

        self.df.to_csv(self.df_path, index=False)

    def add(self, df_temp):
        """
        add rows to database
        """
        self.df = self.df.append(df_temp)
        # index of self.touched is the last one we added
        self.touched = self.touched.append(pd.Series([True],
                                index=[self.df.index[-1]]))

    def delete(self):
        """
        delete row from database
        """
        index_touched =  self.touched[self.touched].index
        self.df = self.df.loc[index_touched]
        self.df = self.df.drop_duplicates()
