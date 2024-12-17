import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import box
from PIL import Image
import io
import imageio
from matplotlib.colors import LinearSegmentedColormap, Normalize
import plotly.graph_objects as go
import os
from google.cloud import bigquery
from storm_functions import detect_storms

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./key.json"
client = bigquery.Client()

st.set_page_config(page_title='Weather data', layout='wide', page_icon=':ambulance:')

shape = gpd.read_file('world-administrative-boundaries').to_crs("EPSG:4326")
bbox = box(102, 8, 112, 24)
cropped_shape = shape.clip(bbox)

page = st.sidebar.radio("Go to", ["Yearly Analysis", "Monthly Analysis","Daily Analysis"])

# Available attributes for selection
attributes = {
    "Temperature": "temperature_celsius",
    "Relative Humidity": "relative_humidity",
    "Mean Sea Level Pressure": "mean_sea_level_pressure",
    "Wind speed": "wind_speed",
    "Sea Surface Temperature": "sea_surface_temperature_celsius",
    "Surface Pressure": "surface_pressure",
    "Total Cloud Cover": "total_cloud_cover",
    "Total Precipitation": "total_precipitation"
}

if page == "Daily Analysis":
    st.title("Daily Weather Data Analysis")

    # Attribute selection
    selected_attr = st.selectbox("Select Attribute for Heatmap", options=list(attributes.keys()))
    selected_column = attributes[selected_attr]

    selected_date = st.date_input("Select Date", value=pd.to_datetime("2024-01-01"))
    selected_date_str = selected_date.strftime("%Y-%m-%d")

    colors = ["purple", "blue", "cyan", "green", "yellow", "orange", "red", "white"]
    custom_cmap = LinearSegmentedColormap.from_list("custom_purple_red", colors)
    data_dict ={}

    if 'data_dict' not in st.session_state or st.session_state.selected_date != selected_date:
        # Query all attributes for the selected date
        for attribute in attributes.values():
            QUERY = f'''
                SELECT
                    {attribute}
                FROM
                    strong-ward-437213-j6.bigdata_20241.dashboard_main
                WHERE
                    valid_time >= '{selected_date_str} 00:00:00 UTC'
                    AND valid_time <= '{selected_date_str} 23:00:00 UTC'
                ORDER BY
                    valid_time, latitude DESC, longitude
            '''
            
            # Execute the query and get results
            query_job = client.query(QUERY)
            rows = query_job.result()

            # Convert results to a numpy array
            data = [row[0] for row in rows]

            # Process data and create a dictionary of 3D arrays for each attribute
            data_dict[attribute] =  np.reshape(data, (24, 65, 41)) 

        msl = data_dict["mean_sea_level_pressure"]
        lon = np.arange(102, 112.25, 0.25)
        lat = np.arange(24, 7.75,-0.25)
        res = 0.25 
        order = 'topdown' 
        Npix_min = 9
        Npix_max = 6000  
        rel_amp_thresh = 100  
        d_thresh = 2500  
        cyc = 'cyclonic'  
        cut_lon = 1  #
        cut_lat = 1  
        globe = False  

        lon_storms = []
        lat_storms = []
        amp_storms = []
        area_storms = []
        regions = []
        for i in range(msl.shape[0]):
            lon_storm, lat_storm, amp_storm, area_storm, region = detect_storms(
                msl[i, :, :], lon, lat, res, order, Npix_min, Npix_max, rel_amp_thresh, 
                d_thresh, cyc, cut_lon, cut_lat, globe
            )
            lon_storms.append(lon_storm)
            lat_storms.append(lat_storm)
            amp_storms.append(amp_storm)
            area_storms.append(area_storm)
            regions.append(regions)
        

        storm_data = {
            "Time": [],
            "Longitude": [],
            "Latitude": [],
            "Amplitude": [],
            "Area": [],
        }

        # Flag to check if any storms are detected
        any_storms_detected = False

        # Populate the lists with data from each time step
        for i in range(len(lon_storms)):
            storm_data["Time"].append(f"{i + 1}:00")
            
            if lon_storms[i] and lat_storms[i] and amp_storms[i] and area_storms[i] and regions[i]:
                # Add storm data if a storm is detected
                storm_data["Longitude"].append(lon_storms[i][0])
                storm_data["Latitude"].append(lat_storms[i][0])
                storm_data["Amplitude"].append(amp_storms[i][0])
                storm_data["Area"].append(area_storms[i][0])
                any_storms_detected = True  # Mark as storm detected
            else:
                # Indicate no storm if data arrays are empty
                storm_data["Longitude"].append("No storm")
                storm_data["Latitude"].append("No storm")
                storm_data["Amplitude"].append("No storm")
                storm_data["Area"].append("No storm")

        # Convert the dictionary to a DataFrame for display
        storm_df = pd.DataFrame(storm_data)


        # Store data and selected date in session state
        st.session_state.data_dict = data_dict
        st.session_state.selected_date = selected_date
        st.session_state.storm_df = storm_df
        st.session_state.any_storm_decteted = any_storms_detected
    else:
        data_dict = st.session_state.data_dict
        storm_df = st.session_state.storm_df
        any_storms_detected = st.session_state.any_storm_decteted


    # Use selected attribute data
    data_array = data_dict[selected_column]

    # GIF creation (only if new data is loaded)
    if 'gif_bytes' not in st.session_state or st.session_state.selected_attr != selected_attr:
        data_array = data_dict[selected_column]
        frames = []
        for i in range(24):
            fig, ax = plt.subplots(figsize=(12, 10))  
            intensity = data_array[i][::-1]
            
            # Plot the geographic boundary and the heatmap
            cropped_shape.boundary.plot(ax=ax, color='black', linewidth=2)
            img = ax.imshow(intensity, cmap=custom_cmap, interpolation='lanczos', extent=[102, 112, 8, 24], origin='lower')
            
            ax.set_title(f"{selected_attr} - Frame {i + 1}", fontsize=14)
            plt.axis("off")
            
            fig.tight_layout()
            
            # Save frame to in-memory buffer
            buf = io.BytesIO()
            fig.savefig(buf, dpi=100)
            buf.seek(0)
            frames.append(Image.open(buf))
            plt.close(fig)

        # Save GIF to session state
        gif_bytes_io = io.BytesIO()
        with imageio.get_writer(gif_bytes_io, format='GIF', duration=0.5, loop=0) as writer:
            for frame in frames:
                writer.append_data(frame)
        st.session_state.gif_bytes = gif_bytes_io.getvalue()
        st.session_state.selected_attr = selected_attr

    col1, col2 = st.columns(2)

    with col1:
        vmin, vmax = np.min(data_array), np.max(data_array)
        fig_colorbar, ax_colorbar = plt.subplots(figsize=(8, 0.5))
        
        # Colorbar
        img = ax_colorbar.imshow(np.linspace(vmin, vmax, 256).reshape(1, -1), cmap=custom_cmap, aspect="auto")
        ax_colorbar.set_yticks([])
        tick_labels = np.linspace(vmin, vmax, 7, endpoint=True, dtype=int)
        ax_colorbar.set_xticklabels(tick_labels)
        
        st.pyplot(fig_colorbar)
        
        gif_bytes = st.session_state.gif_bytes
        st.image(gif_bytes, caption="Heatmap Animation", use_column_width=True, output_format="GIF")

    # Line Chart and Sliders
    with col2:
        lat_slider = st.slider("Select Latitude", 8.0, 24.0, 16.0, step=0.25) 
        lon_slider = st.slider("Select Longitude", 102.0, 112.0, 106.0, step=0.25) 
        
        line_chart_placeholder = st.empty()

        # Map latitude and longitude from slider
        lat_min, lat_max = 8, 24
        lon_min, lon_max = 102, 112
        
        lat_idx = int((lat_max - lat_slider) / 0.25)
        lon_idx = int((lon_slider - lon_min) / 0.25)

        line_data = data_array[:, lat_idx, lon_idx]

        with line_chart_placeholder.container():
            fig_line = go.Figure()

            fig_line.add_trace(go.Scatter(
                x=list(range(24)), 
                y=line_data, 
                mode='lines+markers', 
                marker=dict(color='blue'), 
                name=f'{selected_attr} Trend'
            ))

            fig_line.update_layout(
                title=f"{selected_attr} Trend",
                xaxis_title="Hour of the Day",
                yaxis_title=selected_attr,
                template='plotly'
            )

            st.plotly_chart(fig_line)


        
    st.subheader("Detected Storm Information")
    col3, col4 = st.columns([2, 1])

    # Column 1: Display the storm data table
    with col3:
        if any_storms_detected:
            st.table(storm_df)
        else:
            st.write("No storms detected for the selected date.")

    # Column 2: Plot the storm coordinates on the map with a line connecting them
    with col4:
        if any_storms_detected:
        
            # Giảm kích thước đồ thị
            fig, ax = plt.subplots(figsize=(3, 2.5), dpi=300)
            
            # Vẽ bản đồ đã cắt
            cropped_shape.boundary.plot(ax=ax, color='black', linewidth=1)

            # Vẽ heatmap (giả sử `msl1` có sẵn)
            msl1 = [[0]*100 for _ in range(100)]  # Example placeholder for msl
            plt.imshow(msl1, cmap=custom_cmap, interpolation='lanczos', extent=[102, 112, 8, 24], alpha=0, origin='lower')

            # Vẽ các điểm bão và nối các điểm
            for i in range(1, 24):
                if lon_storms[i] and lat_storms[i] and lon_storms[i-1] and lat_storms[i-1]:
                    # Vẽ đường nối giữa các điểm bão
                    ax.plot([lon_storms[i-1][0], lon_storms[i][0]], 
                            [lat_storms[i-1][0], lat_storms[i][0]], 'b-', marker='o', markersize=1, label=f"Time Step {i}")

            # Vẽ các điểm bão với kích thước nhỏ hơn
            ax.scatter(lon_storms, lat_storms, color='red', marker='x', label='Storm Centers', s=10)  # Giảm kích thước điểm (s=30)

            # Tùy chỉnh biểu đồ
            ax.set_title("Storm Path Coordinates", fontsize=10)
            ax.set_xlabel("Longitude", fontsize=8)
            ax.set_ylabel("Latitude", fontsize=8)
            
            # Tắt grid và axis
            ax.grid(False)  # Tắt grid
            plt.axis("off")  # Tắt trục để bản đồ nhìn rõ hơn

            # Hiển thị bản đồ trong Streamlit
            st.pyplot(fig)

elif page == "Yearly Analysis":
    st.title("Yearly Weather Data Analysis")

    years = list(range(2022, 2025)) 

    selected_year = st.selectbox("Select Year", years)
    lat, lon = st.columns([1,1])
    with lat:
        lat_slider = st.slider("Select Latitude", 8.0, 24.0, 16.0, step=0.25) 
    with lon:
        lon_slider = st.slider("Select Longitude", 102.0, 112.0, 106.0, step=0.25) 

    QUERY = f"""
            SELECT 
                FORMAT_TIMESTAMP('%Y-%m', TIMESTAMP(day)) AS month, 
                MAX(max_temperature) as max_temperature,
                MIN(min_temperrator) as min_temperature,
                MAX(daily_difference) AS max_daily_difference ,
                SUM(total_rain) as rain

            FROM (
                SELECT 
                    DATE(TIMESTAMP(valid_time)) AS day, 
                    MAX(temperature_celsius) - MIN(temperature_celsius) AS daily_difference, 
                    MAX(temperature_celsius) as max_temperature,
                    MIN(temperature_celsius) as min_temperrator,
                    SUM(total_precipitation) as total_rain
                FROM `strong-ward-437213-j6.bigdata_20241.dashboard_main`
                WHERE
                    latitude = {lat_slider}
                    AND longitude = {lon_slider}
                    AND FORMAT_TIMESTAMP('%Y', TIMESTAMP(valid_time)) = '{selected_year}'

                GROUP BY day
                )
            GROUP BY month
            ORDER BY month;

             """
    query_job = client.query(QUERY)
    rows = query_job.result()
    df = {"month":[], "max_tem":[], "min_tem":[], "max_dif_tem":[], "tp":[]}

    for row in rows:
        df["month"].append(row[0])
        df['max_tem'].append(row[1])
        df["min_tem"].append(row[2])
        df["max_dif_tem"].append(row[3])
        df["tp"].append(row[4]*1000)

    fig = go.Figure()

# Vẽ các biểu đồ đường
    fig.add_trace(go.Scatter(
        x=df["month"], 
        y=df["max_tem"], 
        mode='lines+markers', 
        name='Max Temperature',
        line=dict(color='red')
    ))

    fig.add_trace(go.Scatter(
        x=df["month"], 
        y=df["min_tem"], 
        mode='lines+markers', 
        name='Min Temperature',
        line=dict(color='blue')
    ))

    fig.add_trace(go.Scatter(
        x=df["month"], 
        y=df["max_dif_tem"], 
        mode='lines+markers', 
        name='Max Difference in Temperature',
        line=dict(color='green')
    ))

    # Cuối cùng thêm trace của đồ thị cột
    fig.add_trace(go.Bar(
        x=df["month"], 
        y=df["tp"], 
        name='Total Precipitation',
        marker=dict(color='blue'),
        yaxis='y2'  # Dùng trục y thứ hai
    ))

    # Cấu hình biểu đồ
    fig.update_layout(
        title="Temperature and Rainfall Analysis",
        xaxis=dict(
            title="Month",
            tickvals=df["month"]
        ),
        yaxis_title="Temperature (°C)",
        yaxis=dict(
            title="Temperature (°C)",
            titlefont=dict(color="black"),
            tickfont=dict(color="black")
        ),
        yaxis2=dict(
            title="Rainfall (mm)",
            titlefont=dict(color="blue"),
            tickfont=dict(color="blue"),
            overlaying='y',
            side='right'
        ),
        legend=dict(
            x=0.05, y=1.5, orientation='h'
        ),
        template="plotly_white",
        barmode='relative'  # Nếu có nhiều đồ thị cột, tránh chúng chồng lên nhau
    )


    pie_chart = go.Figure(
        go.Pie(
            labels=df["month"],  # Tháng
            values=df["tp"],  # Lượng mưa
            hole=0.3,  # Nếu bạn muốn biểu đồ tròn dạng donut, điều chỉnh giá trị này
            textinfo='label+percent',  # Hiển thị nhãn và phần trăm
            # marker=dict(colors=px.colors.sequential.Blues)  # Tùy chỉnh màu sắc
        )
    )
    pie_chart.update_layout(
        title="Rainfall Distribution by Month"
    )
    col1, col2 = st.columns([2,1])
    with col1:
        st.plotly_chart(fig)
    with col2:
        st.plotly_chart(pie_chart)
        
else:
    st.title("Monthly Weather Data Analysis")

    years = list(range(2022, 2025)) 

    selected_year = st.selectbox("Select Year", years)
    selected_month = st.selectbox("Select Month", range(1, 13), format_func=lambda x: str(x).zfill(2))

    lat, lon = st.columns([1, 1])
    with lat:
        lat_slider = st.slider("Select Latitude", 8.0, 24.0, 16.0, step=0.25)
    with lon:
        lon_slider = st.slider("Select Longitude", 102.0, 112.0, 106.0, step=0.25)

    QUERY = f"""
                SELECT
                    DATE(TIMESTAMP(valid_time)) AS day,
                    MAX(temperature_celsius) AS max_temperature,
                    MIN(temperature_celsius) AS min_temperature,
                    SUM(total_precipitation) AS total_precipitation
                FROM
                    `strong-ward-437213-j6.bigdata_20241.dashboard_main`
                WHERE
                    latitude = {lat_slider}
                    AND longitude = {lon_slider}
                    AND FORMAT_TIMESTAMP('%Y-%m', TIMESTAMP(valid_time)) = '{selected_year}-{selected_month:02d}'
                GROUP BY day
                ORDER BY day;
                """
    query_job = client.query(QUERY)
    rows = query_job.result()
    df = {"day":[], "max_tem":[], "min_tem":[], "tp":[]}
    for row in rows:
        df["day"].append(row[0])
        df['max_tem'].append(row[1])
        df["min_tem"].append(row[2])
        df["tp"].append(row[3]*1000)

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=df["day"], 
        y=df["tp"], 
        name='Total Precipitation',
        marker=dict(color='blue'),
        yaxis='y2', 
        offsetgroup=1
    ))


    fig.add_trace(go.Scatter(
        x=df["day"], 
        y=df["max_tem"], 
        mode='lines+markers', 
        name='Max Temperature',
        line=dict(color='red'),
        offsetgroup=2
    ))

    fig.add_trace(go.Scatter(
        x=df["day"], 
        y=df["min_tem"], 
        mode='lines+markers', 
        name='Min Temperature',
        line=dict(color='green'),
        offsetgroup=2
    ))


    
    # Cấu hình biểu đồ
    fig.update_layout(
        title="Temperature and Rainfall Analysis",
        xaxis=dict(
            title="Day",
            tickvals=df["day"]
        ),
        yaxis_title="Temperature (°C)",
        yaxis=dict(
            title="Temperature (°C)",
            titlefont=dict(color="black"),
            tickfont=dict(color="black")
        ),
        yaxis2=dict(
            title="Rainfall (mm)",
            titlefont=dict(color="blue"),
            tickfont=dict(color="blue"),
            overlaying='y',
            side='right'
        ),
        legend=dict(
            x=0.05, y=1.5, orientation='h'
        ),
        template="plotly_white",
        barmode='relative'  
    )

    pie_chart = go.Figure(
        go.Pie(
            labels=df["day"],  
            values=df["tp"],
            hole=0.3, 
            textinfo='label+percent',  
        )
    )
    pie_chart.update_layout(
        title="Rainfall Distribution by Day"
    )

    col1, col2 = st.columns([2,1])
    with col1:
        st.plotly_chart(fig)
    with col2:
        st.plotly_chart(pie_chart)


