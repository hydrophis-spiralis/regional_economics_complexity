import numpy as np
import pandas as pd


class ComplexDataModel:


    def __init__(self, cp_table: pd.DataFrame):
        self.__cp_table = cp_table
        self.__rca = None
        self.__mcp = None
        self.__diversity = None
        self.__ubiquity = None
        self.__eci = None
        self.__pci = None
        self.__density = None
        self.__proximity = None

    def __compute_rca(self):
        cp_values = self.__cp_table.values
        sum_p = cp_values.sum(axis=1, keepdims=True)
        sum_c = cp_values.sum(axis=0, keepdims=True)
        return (cp_values / sum_c) / (sum_p / cp_values.sum())

    def __compute_mcp(self):
        rca = self.rca
        return np.where(rca >= 1, 1, 0)

    def __compute_diversity(self):
        return np.nansum(self.mcp, axis=1)

    def __compute_ubiquity(self):
        return np.nansum(self.mcp, axis=0)

    def __compute_eci_pci(self):
        kc0 = self.diversity
        kp0 = self.ubiquity

        mcp1 = self.mcp / kc0[:, np.newaxis]
        mcp2 = self.mcp / kp0[np.newaxis, :]

        mcc = np.dot(mcp1, mcp2.T)
        mpp = np.dot(mcp2.T, mcp1)
        self._mcc = mcc
        self._mpp = mpp

        eigvals, eigvec = np.linalg.eig(mpp)
        eigidx = eigvals.argsort()[-2]
        pci = np.real(eigvec[eigidx])
        pci = (pci - pci.mean())/np.nanstd(pci)

        eigvals, eigvec = np.linalg.eig(mcc)
        eigidx = eigvals.argsort()[-2]
        eci = np.real(eigvec[eigidx])
        eci = (eci - eci.mean())/np.nanstd(eci)

        return eci, pci

    def __compute_density(self):

        x = np.where(self.rca > 1, 1, 0)
        prox = self.proximity

        density = x@ prox / np.nansum(prox, axis=1)

        #density = np.sum(xtile * prox, axis=1) / np.sum(prox, axis=1)

        return density

    def __compute_proximity(self):
        mat = self.mcp.T @ self.mcp
        u = self.ubiquity
        u1 = np.tile(u[np.newaxis, :],(len(u), 1))
        u2 = np.tile(u[:, np.newaxis], (1, len(u)))
        norm = np.maximum(u1,u2)
        return mat / norm

    @property
    def rca(self):
        if self.__rca is None:
            self.__rca = self.__compute_rca()
        return self.__rca

    @property
    def mcp(self):
        if self.__mcp is None:
            self.__mcp = self.__compute_mcp()
        return self.__mcp


    @property
    def diversity(self):
        if self.__diversity is None:
            self.__diversity = self.__compute_diversity()
        return self.__diversity

    @property
    def ubiquity(self):
        if self.__ubiquity is None:
            self.__ubiquity = self.__compute_ubiquity()
        return self.__ubiquity

    @property
    def eci(self):
        if self.__eci is None:
            self.__eci, self.__pci = self.__compute_eci_pci()
        return self.__eci

    @property
    def pci(self):
        if self.__pci is None:
            self.__eci, self.__pci = self.__compute_eci_pci()
        return self.__pci

    @property
    def proximity(self):
        if self.__proximity is None:
            self.__proximity = self.__compute_proximity()
        return self.__proximity

    @property
    def density(self):
        if self.__density is None:
            self.__density = self.__compute_density()
        return self.__density


    @classmethod
    def from_country_product_table(cls, table: pd.DataFrame):
        return cls(table)



# if __name__ == '__main__':
#
#
#     model = ComplexDataModel.from_country_product_table(None)
#     print(model.rca)