def df_desc(df, shape_only=False):
    ''' Describe the structure of a dataframe
    '''
    import pandas as pd
    print('Shape of the dataframe:')
    print('{:<10} {:>10,} | {:<10} {:>10,}'.format('Rows:', df.shape[0], 'Columns:', df.shape[1]))
    
    if not shape_only:
        desc = pd.DataFrame({'dtype': df.dtypes,
                             'NAs': df.isna().sum(),
                             '% Missing': np.floor((df.isna().sum() / len(df))*1000)/10,
                             'Boolean': df.apply(lambda column: column == 0).sum() + df.apply(lambda column: column == 1).sum() == len(df),
                             'Numerical': (df.dtypes != 'object') & (df.dtypes != 'datetime64[ns]') & (df.dtypes != 'datetime64[ns, UTC]') & (df.apply(lambda column: column == 0).sum() + df.apply(lambda column: column == 1).sum() != len(df)),
                             'Date': (pd.core.dtypes.common.is_datetime64_any_dtype(df.dtypes)) | (df.dtypes.isin(['datetime64[ns]', 'datetime64[D]', 'datetime64[ns, UTC]', '<M8[ns]', 'M8[ns]', '<M8[D]', 'M8[D]'])) | (np.issubdtype(df.dtypes, np.datetime64)) | (df.dtypes == np.datetime64),
                             'Categorical': df.dtypes == 'object',
                            })

        date_dtype = [pd.core.dtypes.common.is_datetime64_any_dtype(df[col]) for col in df.columns]            
        desc.Date = date_dtype

        desc.Boolean = desc.Boolean.map({True: 'YES', False: '-'})
        desc.Numerical = desc.Numerical.map({True: 'YES', False: '-'})
        desc.Date = desc.Date.map({True: 'YES', False: '-'})
        desc.Categorical = desc.Categorical.map({True: 'YES', False: '-'})

        desc['Count of Categories'] = [len(df.loc[:,col].unique()) if desc.loc[col,'Categorical'] == 'YES' else '-' for col in df.columns]
        desc['Categories'] = [sorted(list(df.loc[:,col].unique().astype('str'))) if desc.loc[col,'Categorical'] == 'YES' else '-' for col in df.columns]
        
        
        return desc
