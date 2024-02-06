import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

def load_data(file):
    df = pd.read_excel(file)
    return df

def filter_data_by_date_range(df, start_date, end_date, date_column):
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    filtered_df = df.loc[(df[date_column] >= start_date) & (df[date_column] <= end_date)]
    return filtered_df

def calculate_total_profit(df, x_axis_column, data_column):
    total_profit_df = df.groupby(x_axis_column)[data_column].sum().reset_index()
    return total_profit_df

def display_line_chart(df, x_axis_column, data_columns):
    st.subheader("Line Chart")

    fig = px.line(df, x=x_axis_column, y=data_columns, title="Line Chart")
    st.plotly_chart(fig)

def display_bar_chart(df, x_axis_column, data_columns):
    st.subheader("Bar Chart")

    fig = px.bar(df, x=x_axis_column, y=data_columns, title="Bar Chart")
    st.plotly_chart(fig)

def display_pie_chart(df, data_columns):
    st.subheader("Pie Chart")

    fig = px.pie(df, names=data_columns[0], values=data_columns[1], title="Pie Chart")
    st.plotly_chart(fig)

def display_data_table(df):
    st.subheader("Data Table")
    st.write(df)

def display_summary_stats(df, selected_columns):
    st.subheader("Summary Statistics")
    st.write(df[selected_columns].describe())

def display_heatmap(df, color_scale):
    st.subheader("Correlation Heatmap")
    corr = df.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap=color_scale, fmt=".2f")
    st.pyplot()

def main():
    st.set_page_config(page_title="Streamlit Power BI", layout="wide")
    st.title("Streamlit Power BI Dashboard")

    uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx", "xls"])

    if uploaded_file is not None:
        df = load_data(uploaded_file)

        st.sidebar.subheader("Select Time Period")
        start_date = st.sidebar.date_input("Start Date", min_value=df[date_column].min(), max_value=df[date_column].max(), value=df[date_column].min())
        end_date = st.sidebar.date_input("End Date", min_value=df[date_column].min(), max_value=df[date_column].max(), value=df[date_column].max())

        df_filtered = filter_data_by_date_range(df, start_date, end_date, date_column)

        st.sidebar.subheader("Select Chart Type")
        chart_type = st.sidebar.selectbox("Choose a chart type", ["Line Chart", "Bar Chart", "Pie Chart"])

        if chart_type == "Line Chart":
            st.sidebar.subheader("Select Line Chart Options")
            x_axis_column_options = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]
            x_axis_column = st.sidebar.selectbox("Choose X-axis (Time) Column", x_axis_column_options, index=0)
            y_axis_column_options = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
            data_columns = st.sidebar.multiselect("Choose Y-axis (Data) Columns", y_axis_column_options)

            if data_columns:
                display_line_chart(df_filtered, x_axis_column, data_columns)

        elif chart_type == "Bar Chart":
            st.sidebar.subheader("Select Bar Chart Options")
            x_axis_column_options = [col for col in df.columns if pd.api.types.is_object_dtype(df[col])]
            x_axis_column = st.sidebar.selectbox("Choose X-axis (Category) Column", x_axis_column_options, index=0)
            y_axis_column_options = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
            data_columns = st.sidebar.multiselect("Choose Y-axis (Data) Columns", y_axis_column_options)

            if data_columns:
                display_bar_chart(df_filtered, x_axis_column, data_columns)

        elif chart_type == "Pie Chart":
            st.sidebar.subheader("Select Pie Chart Options")
            category_column_options = [col for col in df.columns if pd.api.types.is_object_dtype(df[col])]
            category_column = st.sidebar.selectbox("Choose Category Column", category_column_options, index=0)
            numeric_column_options = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
            value_column = st.sidebar.selectbox("Choose Numeric Value Column", numeric_column_options, index=0)

            if category_column and value_column:
                display_pie_chart(df_filtered, [category_column, value_column])

        st.sidebar.subheader("Additional Features")
        if st.sidebar.checkbox("Display Data Table"):
            display_data_table(df_filtered)

        if st.sidebar.checkbox("Display Summary Statistics"):
            selected_columns = st.sidebar.multiselect("Select columns for summary statistics", df.columns)
            if selected_columns:
                display_summary_stats(df_filtered, selected_columns)

        if st.sidebar.checkbox("Display Correlation Heatmap"):
            color_scale_options = ["viridis", "plasma", "inferno", "magma", "cividis"]
            color_scale = st.sidebar.selectbox("Choose color scale", color_scale_options, index=0)
            display_heatmap(df_filtered, color_scale)

if __name__ == "__main__":
    main()
