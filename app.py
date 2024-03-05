import streamlit as st
import pandas as pd
import recordlinkage
from recordlinkage.preprocessing import clean

# Function to upload datasets
def upload_datasets():
    st.subheader("Upload Datasets")
    file1 = st.file_uploader("Upload first dataset (CSV format)", type=["csv"])
    file2 = st.file_uploader("Upload second dataset (CSV format)", type=["csv"])
    
    if file1 is not None and file2 is not None:
        df1 = pd.read_csv(file1)
        df2 = pd.read_csv(file2)
        
        st.write("Preview of first dataset:")
        st.write(df1.head())
        
        st.write("Preview of second dataset:")
        st.write(df2.head())
        
        return df1, df2
    else:
        return None, None

# Function for blocking configuration
def configure_blocking(df1, df2):
    st.subheader("Blocking Configuration")
    blocking_attributes = st.multiselect("Select attributes for blocking", df1.columns)
    st.write("Selected blocking attributes:", blocking_attributes)

# Function for comparison configuration
def configure_comparison(df1, df2):
    # st.subheader("Comparison Configuration")
    
    # Data preprocessing
    df1['firstname'] = clean(df1['firstname'].astype(str))
    df1['lastname'] = clean(df1['lastname'].astype(str))
    df1['dob'] = pd.to_datetime(df1['dob'], errors='coerce')

    df2['firstname'] = clean(df2['firstname'].astype(str))
    df2['lastname'] = clean(df2['lastname'].astype(str))
    df2['dob'] = pd.to_datetime(df2['dob'], errors='coerce')

    features_csv1 = df1[['firstname', 'lastname', 'dob']]
    features_csv2 = df2[['firstname', 'lastname', 'dob']]
    
    # Indexing
    indexer = recordlinkage.Index()
    indexer.full()
    pairs = indexer.index(features_csv1, features_csv2)

    # Comparison
    compare = recordlinkage.Compare()

    # Adding comparison methods
    compare.string('firstname', 'firstname', method='jarowinkler', threshold=0.85, label='firstname_jw')
    compare.string('lastname', 'lastname', method='jarowinkler', threshold=0.85, label='lastname_jw')
    compare.date('dob', 'dob', label='dob')

    features_comparison = compare.compute(pairs, features_csv1, features_csv2)

    # Classification
    matches = features_comparison[features_comparison.sum(axis=1) >= 2]

    # Extracting indices of matched records
    matched_records = []
    for match in matches.index:
        df1_index, df2_index = match
        matched_records.append((df1_index, df2_index))

    print("Matched Records:")

    # Empty DataFrame for the matched records
    matched_df = pd.DataFrame()

    for csv1_index, csv2_index in matched_records:
        # Extract the matching records from both DataFrames
        record_csv1 = df1.loc[[csv1_index]].reset_index(drop=True)
        record_csv2 = df2.loc[[csv2_index]].reset_index(drop=True)

        matched_record = pd.concat([record_csv1, record_csv2], axis=1)
        matched_df = pd.concat([matched_df, matched_record], ignore_index=True)

    column_names = [f"csv1_{col}" if col in df2.columns else col for col in df1.columns] + \
                  [f"csv2_{col}" if col in df1.columns else col for col in df2.columns]

    matched_df.columns = column_names

    matched_df.columns = [col.replace('csv1', 'facility') if 'csv1' in col else col.replace('csv2', 'hdss') for col in matched_df.columns]
    columns = [
    'facility_recnr','hdss_recnr',
    'facility_firstname', 'facility_lastname',
    'hdss_firstname', 'hdss_lastname',
    'facility_petname', 'hdss_petname',
    'facility_dob', 'hdss_dob',
    'facility_sex', 'hdss_sex',
    'facility_nationalid','hdss_nationalid',
    'patientid','hdssid',
    'hdsshhid', 'visitdate'
    ]

    matched_df = matched_df[columns]

    return matched_df

# Main function
def main():
    st.title("Record Linkage System")
    st.write("This system helps in linking hdss records to hospital/treatment centre records.")

    #  Upload datasets
    df1, df2 = upload_datasets()

    if df1 is not None and df2 is not None:
        blocks = configure_blocking(df1, df2)
        matches = configure_comparison(df1, df2)
        st.subheader("Results")
        st.write("Matched Records.")
        # Display matched DataFrame
        st.write(matches)

if __name__ == "__main__":
    main()
