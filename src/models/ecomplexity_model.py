import numpy as np
import pandas as pd


class EconomicDataModel:
    def __init__(self, data):
        self.data = {}
        for year in data.time.unique():
            self.data[year] = EconomicDataModel.make_dataframe(data, year)


    def rca(self):
        if not hasattr(self, 'rca_data'):
            self.rca_data = {}
        for year, year_data in self.data.items():
            loc_n_vals = len(year_data.index.levels[0])
            prod_n_vals = len(year_data.index.levels[1])
            data_np = year_data.values.reshape((loc_n_vals, prod_n_vals))

            with np.errstate(divide='ignore', invalid='ignore'):
                num = (data_np / np.nansum(data_np, axis=1)[:, np.newaxis])
                loc_total = np.nansum(data_np, axis=0)[np.newaxis, :]
                world_total = np.nansum(loc_total, axis=1)[:, np.newaxis]
                den = loc_total / world_total
                self.rca_data[year] = num / den


    def mcp(self, threshold=1):
        if not hasattr(self, 'mcp_data'):
            self.mcp_data = {}

        for year, year_data in self.rca_data.items():
            x = np.nan_to_num(year_data)
            x = np.where( x>= threshold, 1, 0)
            self.mcp_data[year] = x


    def diversity(self):
        if not hasattr(self, 'diversity_data'):
            self.diversity_data = {}
        for year, data in self.mcp_data.items():
            self.diversity_data[year] = np.nansum(data, axis=1)

    def ubiquity(self):
        if not hasattr(self, 'ubiquity_data'):
            self.ubiquity_data  = {}
        for year, year_data in self.mcp_data.items():
            self.ubiquity_data[year] = np.nansum(year_data, axis=0)

    def eci_pci(self):
        self.eci_data = {}
        self.pci_data = {}
        self.MCC = {}
        self.MPP = {}
        self.eigen_kc = {}
        self.eigen_kp = {}
        self.eigen = {}
        for year, year_data in self.mcp_data.items():
            mcp1 = (year_data / self.diversity_data[year][:, np.newaxis])
            mcp2 = (year_data / self.ubiquity_data[year][np.newaxis, :])
            # These matrix multiplication lines are very slow
            Mcc = mcp1 @ mcp2.T
            Mpp = mcp2.T @ mcp1

            self.MCC[year] = Mcc
            self.MPP[year] = Mpp

            # Calculate eigenvectors
            eigvals, eigvecs = np.linalg.eig(Mpp)
            eigvecs = np.real(eigvecs)
            # Get eigenvector corresponding to second largest eigenvalue
            eig_index = eigvals.argsort()[-2]
            kp = eigvecs[:, eig_index]
            kc = mcp1 @ kp

            self.eigen_kc[year] = kc
            self.eigen_kp[year] = kp

            # Adjust sign of ECI and PCI so it makes sense, as per book
            s1 = np.sign(np.corrcoef(self.diversity_data[year], kc)[0, 1])
            eci_t = s1 * kc
            pci_t = s1 * kp

            self.eci_data[year] = (eci_t - eci_t.mean()) / eci_t.std()
            self.pci_data[year] = (pci_t - pci_t.mean()) / pci_t.std()

    def proximity(self):
        self.proximity_data = {}
        for year, year_mcp_data in self.mcp_data.items():
            self.proximity_data[year] = year_mcp_data.T @ year_mcp_data / self.ubiquity_data[year][np.newaxis, :]


    def country_proximity(self):
        self.country_proximity_data = {}
        for year, year_mcp_data in self.mcp_data.items():
            self.country_proximity_data[year] = year_mcp_data @ year_mcp_data.T / self.diversity_data[year][np.newaxis, :]


    def density(self, country, year):
        pass

    @staticmethod
    def make_dataframe(data:pd.DataFrame, year:int):
        yeardata = data[data.time == year].copy().drop('time', axis=1)

        #yeardata = yeardata.drop('index',axis=1)
        diversity_check = yeardata .reset_index().groupby(
            ['loc'])['val'].sum().reset_index()
        ubiquity_check = yeardata .reset_index().groupby(
            ['prod'])['val'].sum().reset_index()
        diversity_check = diversity_check[diversity_check.val != 0]
        ubiquity_check = ubiquity_check[ubiquity_check.val != 0]
        #yeardata = yeardata .reset_index()
        yeardata = yeardata .merge(
            diversity_check[['loc']], on='loc', how='right')
        yeardata = yeardata .merge(
            ubiquity_check[['prod']], on='prod', how='right')
        yeardata .set_index(['loc','prod'], inplace=True)
        data_index = pd.MultiIndex.from_product(
            yeardata .index.levels, names=yeardata .index.names)
        yeardata = yeardata .reindex(data_index, fill_value=0)
        return yeardata



    @classmethod
    def getContrainer(cls, data):
        trade_cols = {'time': 'year', 'loc': 'origin', 'prod': 'hs07', 'val': 'export_val'}
        data = data[['origin', 'hs07', 'export_val', 'year']].copy()

        data.origin = data.origin.astype(str)
        data.hs07 = data.hs07.astype(str)
        data.export_val = data.export_val.astype(float)
        data.year = data.year.astype(int)

        cols_map_inv = {v: k for k, v in trade_cols.items()}
        data = data.rename(columns=cols_map_inv)
        data.val = pd.to_numeric(data[['time', 'loc', 'prod', 'val']].val, errors='raise')
        data.val.fillna(0, inplace=True)
        data = data.groupby(['time', 'loc', 'prod']).val.sum().reset_index()
        return cls(data)




