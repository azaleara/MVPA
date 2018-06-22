#!/home/inb/lconcha/fmrilab_software/miniconda2/bin/python
# coding: utf-8
from mvpa2.suite import *
import os.path


data_path = '/misc/giora/azalea/MVPA/data/verbose/data'
subject = 1
mask_name = 'aza_LM1'
mask_folder = 'orig'
runs_to_train = [1, 2, 3, 4]
runs_to_test = [5, 6, 7, 8]
hemo_cor = 8
model = 1
task = 1
dhandle = OpenFMRIDataset(datapath)
subjString = 'sub' + '%03d' % subject

run_datasets = []
for run_id in runs_to_train:
    mask_fname = os.path.join(datapath, 'sub001', 'masks', 'orig', mask_name + '.nii.gz')

    run_events = dhandle.get_bold_run_model(model, subj, run_id)
    # carga la corrida pero solamente el volumen comprendido dentro de la mascara
    run_ds = dhandle.get_bold_run_dataset(subj, task, run_id, chunks=run_id -1, mask=mask_fname)
    #correccion hemodinamica
    if HemoCor != 0:
        for i, ar in enumerate(run_events):
            run_events[i]['onset'] = run_events[i]['onset'] + HemoCor
    # convierte run_events y lo asigna a targets
    run_ds.sa['targets'] = events2sample_attr(
                run_events, run_ds.sa.time_coords, noinfolabel='rest')
    #asigna del dataset al array que contiene todos
    run_datasets.append(run_ds)
fdsTrain = vstack(run_datasets, a=0)


# # Cargar los datos para probar al clasificador

# In[ ]:

run_datasets2 = [] #array donde se meteran los datos
for run_id in runsToTest: #Se mueve corrida por corrida para cargar cada dato
    # genera path de la mascara
    mask_fname = os.path.join(datapath, subjString, 'masks', maskFolder, maskName + '.nii.gz')
    # carga los vectores
    run_events = dhandle.get_bold_run_model(model, subj, run_id)
    # carga la corrida pero solamente el volumen comprendido dentro de la mascara
    run_ds = dhandle.get_bold_run_dataset(subj, task, run_id, chunks=run_id -1, mask=mask_fname)
    #correccion hemodinamica
    if HemoCor != 0:
        for i, ar in enumerate(run_events):
            run_events[i]['onset'] = run_events[i]['onset'] + HemoCor
    # convierte run_events y lo asigna a targets
    run_ds.sa['targets'] = events2sample_attr(
                run_events, run_ds.sa.time_coords, noinfolabel='rest')
    #asigna del dataset al array que contiene todos
    run_datasets2.append(run_ds)
fdsTest = vstack(run_datasets2, a=0)


# # Procesado previo a la clasificación

# In[ ]:

#Correccion
detrender = poly_detrend(fdsTrain, polyord=1, chunks_attr='chunks')

#conversion a puntaje z
zscore(fdsTrain, param_est=('targets', ['rest']))

#Remueve los volumenes asociados a la linea base
fdsTrain = fdsTrain[fds.sa.targets != 'rest']

#promedia los targets en cada chunk
run_averager = mean_group_sample(['targets', 'chunks'])
fdsTrain = fdsTrain.get_mapped(run_averager)


# In[ ]:

#Correccion
poly_detrend(fdsTest, polyord=1, chunks_attr='chunks')

#conversion a puntaje z
zscore(fdsTest, param_est=('targets', ['rest']))

#Remueve los volumenes asociados a la linea base
fdsTest = fdsTest[fdsTest.sa.targets != 'rest']

#promedia los targets en cada chunk
run_averager = mean_group_sample(['targets', 'chunks'])
fdsTest = fdsTest.get_mapped(run_averager)


# # Clasificación

# In[ ]:

#Crea un objeto del clasificador
clf = LinearCSVMC()
clf.train(fdsTrain)
err = clf(ds_split1)
print np.asscalar(err)
