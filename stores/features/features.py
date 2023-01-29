from datetime import datetime, timedelta

from feast import Entity, FeatureView, Field, FileSource, ValueType
from feast.data_format import ParquetFormat
from feast.types import String

# Read data
START_TIME = "2020-02-17"
project_details = FileSource(
    path="data/features.parquet",
    event_timestamp_column="created_on",
    file_format=ParquetFormat(),
)

# Define an entity for the project
project = Entity(
    name="id",
    value_type=ValueType.INT64,
    description="project id",
)

# Define a Feature View for each project
# Can be used for fetching historical data and online serving
project_details_view = FeatureView(
    name="project_details",
    entities=[project],
    ttl=timedelta(days=(datetime.today() - datetime.strptime(START_TIME, "%Y-%m-%d")).days),
    schema=[
        Field(name="text", dtype=String),
        Field(name="tag", dtype=String),
    ],
    online=True,
    source=project_details,
)
