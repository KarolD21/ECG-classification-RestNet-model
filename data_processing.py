import pandas as pd
import numpy as np
import os
import wfdb as ecg
from data_input_v2 import df_ptbxl_net, df_ptbxl_statements
from sklearn.preprocessing import MinMaxScaler

path = "path to database files"
rate = 500
name = '500_normalized'

# ADDITIONAL FUNCTIONS
# Getting plot and info about specyfic signal choosen by ecg_id index
def id_getting_info(df, ecg_id, path, plot = 'no', sampling_rate = 100):
    if sampling_rate == 500:
        rate = 'filename_hr'
    else:
        rate = 'filename_lr'
    direct = path + df.loc[ecg_id, rate]
    signal, meta = ecg.rdsamp(direct, warn_empty=True)
    print('Info about the signal:\n\n', meta)

    record = ecg.rdrecord(direct, warn_empty=True)
    # annotation = ecg.rdann(comp_p, 'hea')
    
    if plot == 'yes':
        ecg.plot_wfdb(record, plot_sym=True, time_units='seconds', ann_style=meta['sig_name'],
                      figsize=(10,20), ecg_grids='all')

# exemple: id_getting_info(df_ptbxl_net, 1, path)

# Getting data from base format to numpy 3d array
def load_raw_data(df, path, sampling_rate, print_info = 'no'):
    if sampling_rate == 100:
        data = [ecg.rdsamp(path+f) for f in df.filename_lr] # read the signal and metadata from each file
    else:
        data = [ecg.rdsamp(path+f) for f in df.filename_hr]
    data = np.array([signal for signal, meta in data])
    data = data.transpose(0, 2, 1)
    if print_info == 'yes':
        print('Data info: ')
        print(type(data))
        print(data.shape, '\n')
    return data

# exemple: load_raw_data(df, path, sampling_rate=rate)

# MAIN WORKFLOW
# Step 1: aggregation of malfunction tags
def class_agregation(df, class_type_df):
    class_list = []
    for val in df.scp_codes:
        if isinstance(val, dict):  # ensure val is a dictionary
            # creates a new dictionary valid_keys containing only those key-value 
            # pairs where the key is present in type_df['malfunction_name'].values.
            valid_keys = {key: val[key] for key in val if key in class_type_df['malfunction_name'].values}
            if valid_keys:
                keymax = max(valid_keys, key=valid_keys.get)
            else:
                keymax = 'NN'
            class_list.append(keymax)

    return class_list

# Step 2: aggregation of malfunction index from malfunction tags
def index_finder(lista):
    idx_arr = []
    for m in lista:
        if m in df_ptbxl_statements['malfunction_name'].values:
            idx = df_ptbxl_statements.loc[(df_ptbxl_statements == m).any(axis=1)].index[0]
            idx_arr.append(idx)
    idx_arr = np.array(idx_arr).reshape(1, len(lista))

    return idx_arr

# Step 3: cnoversion from NumPy array to Pandas DataFrame, cleaning and arragement of data and classes
def class_insert(df, rate, class_num_1 = None, class_num_2 = None, class_num_3 = None):
    t_data = load_raw_data(df, path, sampling_rate=rate)

    layer = t_data.shape[0]
    row = t_data.shape[1]
    col = t_data.shape[2]
    age = df.age.to_numpy().reshape(1, layer)
    sex = df.sex.to_numpy().reshape(1, layer)

    scaler = MinMaxScaler(feature_range=(-1, 1))

    # normalization of data across layers
    # for i in range(row):
    #     data = t_data[:, i, :]
    #     flat_data = data.reshape(-1, 1)
    #     normalized_data = scaler.fit_transform(flat_data).reshape(data.shape)
    #     normalized_data = np.around(normalized_data, 3)
    #     t_data[:, i, :] = normalized_data

    # normalization of data across rows
    for i in range(layer):
        for e in range(row):
            row_data = t_data[i, e, :].reshape(-1, 1)
            normalized_data = scaler.fit_transform(row_data).flatten()
            normalized_data = np.around(normalized_data, 3)
            t_data[i, e, :] = normalized_data

    all_class = np.concatenate((age, sex, class_num_1, class_num_2, class_num_3), axis=0)
    ecg_data = np.zeros((layer, row + all_class.shape[0], col))
    ecg_data[:,:row,:] = t_data
    ecg_data[:, row:row + all_class.shape[0], 0] = all_class.T

    # flattening 3D array into 2D => (layer * (row + 1), col)
    # example: (10 * 13, 1000) => (130, 1000)
    flat_ecg_data = ecg_data.reshape((layer * (row + 5), col))
    main_df = pd.DataFrame(flat_ecg_data)

    # creates an array where each layer index is repeated (row + x) times.
    main_df['layer'] = np.repeat(np.arange(layer), row + 5)
    # creates an array where the sequence from 0 to row is repeated for each layer
    main_df['row'] = np.tile(np.arange(row + 5), layer)
    main_df = main_df[['layer', 'row'] + list(main_df.columns[:-2])]

    for e in range(row, row + all_class.shape[0]):
        idx = main_df[main_df['row'] == e].index
        if idx.all():
            main_df.loc[idx, 0] = main_df.loc[idx, 0].astype(int)
            # main_df.loc[idx, 1:] = np.nan
            main_df.loc[idx, 1:] = 0

    if (all_class[:, 0] == main_df.iloc[np.linspace(row, 
                    row + all_class.shape[0], len(all_class)), 2]).all():
        raise ValueError('Position of the classes in the dataframe is incorrect, check the dataframe for class positioning failure')
    
    return main_df

# Dividing df with malfunction labels to smaller class df by parts of columns with
# tags of diffrent malfunction
diagnostic_col = df_ptbxl_statements[df_ptbxl_statements.diagnostic == 1]
form_col = df_ptbxl_statements[df_ptbxl_statements.form == 1]
rhythm_col = df_ptbxl_statements[df_ptbxl_statements.rhythm == 1]

# Data sets sorted and gruped by values in 'strat_fold' column
training = df_ptbxl_net[df_ptbxl_net.strat_fold.isin(range(1,9))].copy()
validation = df_ptbxl_net[df_ptbxl_net.strat_fold == 9].copy()
testing = df_ptbxl_net[df_ptbxl_net.strat_fold == 10].copy()

# Malfunction names/labels agregate from dictionaries to list
diagnostic_class_training = class_agregation(training, diagnostic_col)
form_class_training = class_agregation(training, form_col)
rhythm_class_training = class_agregation(training, rhythm_col)

diagnostic_class_validation = class_agregation(validation, diagnostic_col)
form_class_validation = class_agregation(validation, form_col)
rhythm_class_validation = class_agregation(validation, rhythm_col)

diagnostic_class_testing = class_agregation(testing, diagnostic_col)
form_class_testing = class_agregation(testing, form_col)
rhythm_class_testing = class_agregation(testing, rhythm_col)

# Malfunction names/labels indexes agregate from data frame to list
diagnostic_idx_training = index_finder(diagnostic_class_training)
form_idx_training = index_finder(form_class_training)
rhythm_idx_training = index_finder(rhythm_class_training)

diagnostic_idx_validation = index_finder(diagnostic_class_validation)
form_idx_validation = index_finder(form_class_validation)
rhythm_idx_validation = index_finder(rhythm_class_validation)

diagnostic_idx_testing = index_finder(diagnostic_class_testing)
form_idx_testing = index_finder(form_class_testing)
rhythm_idx_testing = index_finder(rhythm_class_testing)

# Conversion from NumPy array to Pandas DataFrame, cleaning and arragement of data and classes
df_ecg_training = class_insert(df=training, rate=rate, class_num_1=form_idx_training, class_num_2=rhythm_idx_training, class_num_3=diagnostic_idx_training)
df_ecg_validation = class_insert(df=validation, rate=rate, class_num_1=form_idx_validation, class_num_2=rhythm_idx_validation, class_num_3=diagnostic_idx_validation)
df_ecg_testing = class_insert(df=testing, rate=rate, class_num_1=form_idx_testing, class_num_2=rhythm_idx_testing, class_num_3=diagnostic_idx_testing)

# Printing the number of classes in each set
print('Number of classes for diagnostic: ', len(df_ptbxl_statements[df_ptbxl_statements.diagnostic == 1]))
print('Number of classes for rhythm: ', len(df_ptbxl_statements[df_ptbxl_statements.rhythm == 1]))
print('Number of classes for form: ', len(df_ptbxl_statements[df_ptbxl_statements.form == 1]))

# Function for making map of classes is applied in order to avoid any misunderstanding in
# PyTorch nn.Module where the number of neurons in output layer is setted by
# ecg malfunction class/label index number
def mapping_class(df, num_row):
    key = df[df.row == num_row]
    key = key.iloc[:,2].unique().tolist()
    key.sort()
    value = np.arange(len(key)).tolist()
    map_dict = {}
    for k in key:
        for v in value:
            map_dict[k] = v
            value.remove(v)
            break
    return map_dict

# Function for swapping orginal classes with map
def map_class_swap(df, row_num, map_name):
    df.loc[df.row == row_num, df.columns[2]] = df.loc[df.row == row_num, df.columns[2]].replace(map_name)

# Making map of classes for each set and type of malfunction category
diagnostic_map = mapping_class(df_ecg_training, 16)
rhythm_map = mapping_class(df_ecg_training, 15)
form_map = mapping_class(df_ecg_training, 14)

# All maps are make on the basis of df_ecg_training dataframe because
# there is maximum number of classes from all others

# Saving maps
np.save(f'maps/diagnostic_map{name}.npy', diagnostic_map)
np.save(f'maps/rhythm_map{name}.npy', rhythm_map) 
np.save(f'maps/form_map{name}.npy', form_map) 

# Appling map - swapping class in main dfs
map_class_swap(df_ecg_training, 16, diagnostic_map)
map_class_swap(df_ecg_validation, 16, diagnostic_map)
map_class_swap(df_ecg_testing, 16, diagnostic_map)

map_class_swap(df_ecg_training, 15, rhythm_map)
map_class_swap(df_ecg_validation, 15, rhythm_map)
map_class_swap(df_ecg_testing, 15, rhythm_map)

map_class_swap(df_ecg_training, 14, form_map)
map_class_swap(df_ecg_validation, 14, form_map)
map_class_swap(df_ecg_testing, 14, form_map)

# Saving dfs to .gzip file
df_ecg_training.columns = df_ecg_training.columns.astype(str)
df_ecg_validation.columns = df_ecg_validation.columns.astype(str)
df_ecg_testing.columns = df_ecg_testing.columns.astype(str)

df_ecg_training.to_parquet(f'gzip_data/df_ecg_training{name}.gzip', engine='fastparquet', compression='gzip')
df_ecg_validation.to_parquet(f'gzip_data/df_ecg_validation{name}.gzip', engine='fastparquet', compression='gzip')
df_ecg_testing.to_parquet(f'gzip_data/df_ecg_testing{name}.gzip', engine='fastparquet', compression='gzip')