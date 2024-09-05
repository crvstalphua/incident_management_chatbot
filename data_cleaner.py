import pandas as pd
import re

excel_path = 'data/agency_names.xlsx'
'''
Given an excel file with three columns, 'Agency Name' (full name of Agency), 'Agency' (acronym of Agency) and 'Masking'/

encrypt() masks any detected values of 'Agency Name' or 'Agency' into 'Masking'
decrypt() masks any detected values of 'Masking' into 'Agency Name'
'''

df = pd.read_excel(excel_path)
encryption_full_dict = pd.Series(df.Masking.values, index=df['Agency Name']).to_dict()
encryption_short_dict = pd.Series(df.Masking.values, index=df['Agency']).to_dict()
encryption_dict = {**encryption_full_dict, **encryption_short_dict}
decryption_dict = pd.Series(df['Agency Name'].values, index=df['Masking']).to_dict()

def encrypt(document):

    pattern = re.compile('|'.join(re.escape(key) for key in encryption_dict.keys()))
    
    def replace_match(match):
        return encryption_dict[match.group(0)]
    
    return pattern.sub(replace_match, document)

def decrypt(document):

    pattern = re.compile('|'.join(re.escape(key) for key in decryption_dict.keys()))
    
    def replace_match(match):
        return decryption_dict[match.group(0)]
    
    return pattern.sub(replace_match, document)

