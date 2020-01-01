import os
from pathlib import Path
import pandas as pd
from src.models.ecomplexity_model import EconomicDataModel
import pickle as pkl
import numpy as np

project_dir = Path(__file__).resolve().parents[1]
raw_dir = os.path.join(project_dir, 'data', 'raw')
interim_dir = os.path.join(project_dir, 'data', 'interim')
external_data_dir = os.path.join(project_dir, 'data', 'external')
processed_data_dir = os.path.join(project_dir, 'data', 'processed')


if __name__ == '__main__':
    world_region_data = pd.read_csv(os.path.join(raw_dir, 'fake_tbl.csv'))
    processing_container = EconomicDataModel.getContrainer(world_region_data)
    processing_container.rca()
    processing_container.mcp()
    processing_container.ubiquity()
    processing_container.diversity()
    processing_container.proximity()
    processing_container.eci_pci()

    # print(processing_container.mcp_data[2019])

    #np.savetxt('mcc_eigen_vecs.csv', np.linalg.eig(processing_container.MCC[2019])[1], fmt='%.5f',delimiter=',')

    np.savetxt('mcc.csv', processing_container.MCC[2019],fmt='%.5f', delimiter=',')
    np.savetxt('mpp.csv', processing_container.MPP[2019],fmt='%.5f', delimiter=',')
    #np.savetxt('eigen.csv', processing_container.eigen_kp[2019],fmt='%.5f', delimiter=',')
    mpp = processing_container.MPP[2019]
    mcc = processing_container.MCC[2019]
    eigenvals, eigenvec = np.linalg.eig(mcc)

    np.savetxt('mcc_eigenvec.csv', eigenvec, fmt='%.5f', delimiter=',')
    np.savetxt('mcc_eigenval.csv', eigenvals, fmt='%.5f', delimiter=',')
    np.savetxt('mpp_eigenvec.csv', np.linalg.eig(mpp)[1][1], fmt='%.5f', delimiter=',')
    np.savetxt('mpp_eigenval.csv', np.linalg.eig(mpp)[0], fmt='%.5f', delimiter=',')

    res_tbl = {}

    c_prox = processing_container.country_proximity()
    with open(project_dir/ 'data' / 'cprox.pkl', 'wb') as f:
        pkl.dump(processing_container.country_proximity_data, f)

    

