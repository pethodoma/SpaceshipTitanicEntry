import pandas as pd
import numpy as np


def preproc(filename):

    df = pd.read_csv(filename)
    print(df['Destination'].unique())
    # replacing nan in numeric values simply with median
    numeric_columns = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    [df[col].fillna(df[col].median(), inplace=True) for col in numeric_columns]

    # replacing nan with mode
    df['Cabin'].fillna(df['Cabin'].mode()[0], inplace=True)
 
    # in case of the cabin column, the cabin number, the first letter and the last letter will be extracted, as they all contain information    
    df['Cabin_num']  = df['Cabin'].str.extract(r'/(\d+)').astype(int)
  
    # first letters
    for letter in ['A','B','C','D','E','F','G','T']:
        df[f'Cabin_{letter}']=(df['Cabin'].str.extract(r'(\w)/').astype(str)==letter).astype(int)
   
    # can be either S or T
    df['Cabin_last_letter'] = df['Cabin'].str.extract(r'/(\w)$').astype(str)
    
    # replacing nan in bool values
    if(filename == 'train.csv'):
        bool_columns = ['CryoSleep', 'VIP', 'Transported']
    else:
        bool_columns = ['CryoSleep', 'VIP']
    [df[col].fillna(df[col].mode()[0], inplace=True) for col in bool_columns]

    # converting bool values to numeric
    df['CryoSleep'] = df['CryoSleep'].astype(np.float32)
    df['VIP'] = df['VIP'].astype(np.float32)
    if(filename == 'train.csv'):
        df['Transported'] = df['Transported'].astype(np.float32)

    # passenger id will be split to two new columns: group(group id) and group number (their number inside the group)
    df['Group'] = df['PassengerId'].str.extract(r'(\w{4})')
    df['Group'] = df['Group'].astype(str)
    # zeros will be removed from the start of the string
    df['Group'] = df['Group'].str.replace('^0+', '', regex=True)
    df['Group'] = df['Group'].astype(np.float32)

    # passengers number in the group
    df['GroupNumber'] = df['PassengerId'].str.extract(r'_(\d{2})').astype(int)
    df['GroupNumber'] = df['GroupNumber'].astype(str)
    df['GroupNumber'] = df['GroupNumber'].str.replace('^0+', '', regex=True)
    df['GroupNumber'] = df['GroupNumber'].astype(np.float32)
    
    # now both PassengerId and Cabin columns can be dropped
    df.drop(columns=['PassengerId'], inplace=True)
    df.drop(columns=['Cabin'], inplace=True)

    # also dropping the name column - it doesn't contain any useful information
    df.drop(columns=['Name'], inplace=True)

    # one-hot encoding for categorical values
    cat_columns = ['HomePlanet', 'Destination','Cabin_last_letter']
    df = pd.get_dummies(df, columns=cat_columns)
    for col in df.columns:
        if col.startswith('HomePlanet_') or col.startswith('Cabin_') or col.startswith('Destination_'):
            df[col] = df[col].astype(np.float32)

    return df


