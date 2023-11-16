import pandas as pd
import numpy as np

sub_array = ['', 'NONE',
'ALCOHOL',
'COCAINE/CRACK',
'MARIJUANA/HASHISH',
'HEROIN',
'NON-PRESCRIPTION METHADONE',
'OTHER OPIATES AND SYNTHETICS',
'PCP',
'OTHER HALLUCINOGENS',
'METHAMPHETAMINE',
'OTHER AMPHETAMINES', 'OTHER STIMULANTS',
'BENZODIAZEPINES', 'OTHER NON - BENZODIAZEPINE',
'TRANQUILIZERS','BARBITURATES',
'OTHER NON - BARBITURATE SEDATIVES',
'HYPNOTICS', 'INHALANTS',
'OVER - THE - COUNTER MEDICATIONS'
             ]

route_of_admin_array = ['', 'ORAL'
'SMOKING',
'INHALATION',
'INJECTION (IV OR INTRAMUSCULAR)']

flags = [    'ALCFLG',
    'COKEFLG',
    'MARFLG',
    'HERFLG',
    'METHFLG',
    'OPSYNFLG',

 #   'PCPFLG',
 #   'HALLFLG',
 #   'MTHAMFLG',
 #   'AMPHFLG',
 #   'STIMFLG',
 #   'BENZFLG',
 #   'TRNQFLG',
 #   'BARBFLG',
 #   'SEDHPFLG',
 #   'INHFLG',
 #   'OTCFLG'
 ]

flag_values = [
'ALCOHOL',
'COCAINE/CRACK',
'MARIJUANA/HASHISH',
'HEROIN',
'NON-RX METHADONE',
'OTHER OPIATES/SYNTHETICS'


]

ouput_data = []
global_list = []

df = pd.read_csv("TEDSA_PUF_2020_reduced.csv")
for i in range(len(df)):
    input_list = []
    sub1_n = df.loc[i, "SUB1"]
    if sub1_n>0:
        df.loc[i, "SUB1"] = sub_array[sub1_n]
        input_list.append(sub_array[sub1_n])
    else:
        df.loc[i, "SUB1"] = 'invalid'
        input_list.append('invalid')

    sub1_n = df.loc[i, "SUB2"]
    if sub1_n > 0:
        df.loc[i, "SUB2"] = sub_array[sub1_n]
        input_list.append(sub_array[sub1_n])
    else:
        df.loc[i, "SUB2"] = 'invalid'
        input_list.append('invalid')

    ra1_n = df.loc[i, "ROUTE1"]
    if ra1_n>0 and ra1_n< 4 :
        df.loc[i, "ROUTE1"] = route_of_admin_array[ra1_n]
        input_list.append(sub_array[ra1_n])
    else:
        df.loc[i, "ROUTE1"] = 'invalid'
        input_list.append('invalid')

    ra1_n = df.loc[i, "ROUTE2"]
    if ra1_n > 0 and ra1_n < 4:
        df.loc[i, "ROUTE2"] = route_of_admin_array[ra1_n]
        input_list. append(sub_array[sub1_n])
    else:
        df.loc[i, "ROUTE2"] = 'invalid'
        input_list.append('invalid')

    ra1_n = df.loc[i, "ROUTE3"]
    if ra1_n > 0 and ra1_n < 4:
        df.loc[i, "ROUTE3"] = route_of_admin_array[ra1_n]
        input_list. append( sub_array[sub1_n])
    else:
        df.loc[i, "ROUTE3"] = 'invalid'
        input_list.append('invalid')

    count = 0
    for f in flags:
        if df.loc[i, f] > 0:
            input_list.append(flag_values[count])
        count+=1

    sub3_n = df.loc[i, "SUB3"]
    check = isinstance(sub3_n, np.integer)
    if isinstance(sub3_n, np.integer) and sub3_n>0 and sub3_n<4:
        target = flag_values[sub3_n]
    else:
        target = 'invalid'

    input_list = list(set(input_list))
    input_line = ", ".join(input_list)

    ouput_data.append((input_line.lower(), target.lower()))
    if i>100:
        break
    global_list +=  input_list
dataframe_io = pd.DataFrame(ouput_data, columns=['Input','Output'])
dataframe_io.to_csv('treatment_sequence_data.csv', index = False)

global_list = list(set(global_list))
list_pr  = str(global_list ).lower()
print(list_pr )