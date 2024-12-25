import pyarrow as pa
import pyarrow.parquet as pq
import os
import pandas as pd

if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/home/src/strong-ward-437213-j6-c3ae16d10e5f.json"
project_id = "strong-ward-437213-j6"
bucket_name = 'weather_ds_20241'
table_name = "other_data"
root_path=f'{bucket_name}/{table_name}'


@data_exporter
def export_data(data, *args, **kwargs):
    if data is None:
        raise ValueError('Data is missing!')
    
    if not pd.api.types.is_datetime64_any_dtype(data['time']):
        data['time'] = pd.to_datetime(data['time'])

    data["year"] = data["time"].dt.year
    data["month"] = data["time"].dt.month
    data["day"] = data["time"].dt.day  # Thêm phân vùng theo ngày

    table = pa.Table.from_pandas(data)
    gcs = pa.fs.GcsFileSystem()
    pq.write_to_dataset(
        table,
        root_path=root_path,
        partition_cols=["year", "month", "day"],  # Phân vùng theo năm, tháng và ngày
        filesystem=gcs
    )
