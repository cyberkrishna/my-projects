import streamlit as st
import pandas as pd
import plotly.express as px

def load_data(file):
    df = pd.read_excel(file)
    return df

def filter_data_by_date_range(df, start_date, end_date, date_column):
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    filtered_df = df.loc[(df[date_column] >= start_date) & (df[date_column] <= end_date)]
    return filtered_df

def display_summary_stats(df, selected_columns):
    st.subheader("Summary Statistics")
    st.write(df[selected_columns].describe())

def display_data_table(df):
    st.subheader("Interactive Data Table")
    st.write(df)

def display_chart(chart_type, df, x_axis_column, data_columns):
    st.subheader(f"{chart_type} Chart")
    if chart_type == "Bar":
        fig = px.bar(df, x=x_axis_column, y=data_columns, title=f"{chart_type} Chart")
    elif chart_type == "Line":
        fig = px.line(df, x=x_axis_column, y=data_columns, title=f"{chart_type} Chart")
    elif chart_type == "Pie":
        if len(data_columns) == 1:
            fig = px.pie(df, names=x_axis_column, values=data_columns[0], title=f"{chart_type} Chart")
        else:
            st.warning("Please select exactly one column for the pie chart.")
            return
    elif chart_type == "Scatter":
        fig = px.scatter(df, x=x_axis_column, y=data_columns, title=f"{chart_type} Chart")
    elif chart_type == "Box":
        if len(data_columns) == 1:
            fig = px.box(df, x=x_axis_column, y=data_columns[0], title=f"{chart_type} Chart")
        else:
            st.warning("Please select exactly one column for the box plot.")
            return

    st.plotly_chart(fig)

def main():
    st.set_page_config(page_title="Excel Analyzer", page_icon=":1234:", layout="wide")
    st.title(":1234: Excel Analyzer")
    st.markdown('<style>div.block-container{padding-top:1rem;}</style>', unsafe_allow_html=True)

    # File Upload
    uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx", "xls"])

    if uploaded_file is not None:
        df = load_data(uploaded_file)
        st.write(f"Uploaded file: {uploaded_file.name}")

        # Automatic Date Column Detection
        date_columns = [col for col in df.columns if "date" in col.lower()]
        if date_columns:
            date_column = date_columns[0]
        else:
            st.warning("No date column found. Please specify the date column.")

        # Data Analysis
        st.subheader("Data Analysis")
        st.write("Data Preview:")
        st.write(df.head())

        # Instructions
        st.info("Instructions:")
        st.markdown("1. Upload an Excel file containing your data.")
        st.markdown("2. Choose a chart type from the dropdown.")
        st.markdown("3. Select a category column for the X-axis.")
        st.markdown("4. Select one or more numerical data columns for the Y-axis.")
        st.markdown("5. Ensure the selected columns are of the appropriate data type for the chosen chart.")
        st.markdown("6. Set the time period using the date range selector.")
        st.markdown("7. Click on the chart type to generate and display the chart.")
        
        st.subheader("Select Time Period")
        start_date = st.date_input("Start Date", min_value=df[date_column].min(), max_value=df[date_column].max(), value=df[date_column].min())
        end_date = st.date_input("End Date", min_value=df[date_column].min(), max_value=df[date_column].max(), value=df[date_column].max())

        # Filter Data by Date Range
        df_filtered = filter_data_by_date_range(df, start_date, end_date, date_column)

        # Chart Selection
        st.subheader("Select a Chart Type")
        chart_type = st.selectbox("Choose a chart type", ["Bar", "Line", "Pie", "Scatter", "Box"])

        # X-Axis Selection (Category Column)
        st.subheader("Select Category ")
        x_axis_column_options = [col for col in df.columns if pd.api.types.is_categorical_dtype(df[col]) or pd.api.types.is_object_dtype(df[col])]
        x_axis_column = st.selectbox("Choose a category column for the X-axis", x_axis_column_options, index=0)

        # Y-Axis Selection (Numerical Data Columns)
        st.subheader("Select  Data ")
        y_axis_column_options = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
        data_columns = st.multiselect("Select numerical data columns for the Y-axis", y_axis_column_options)

        if not data_columns:
            st.warning("Please select at least one numerical data column.")
            return

        # Generate the selected chart
        display_chart(chart_type, df_filtered, x_axis_column, data_columns)

        st.sidebar.title("Additional Features")
        
        if st.sidebar.checkbox("Display Summary Statistics"):
            display_summary_stats(df_filtered, data_columns)

if __name__ == "__main__":
    main()
