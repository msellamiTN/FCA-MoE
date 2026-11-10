import pandas as pd
from moe_fca.trace_io import ParquetTraceWriter

def test_parquet_writer_resume(tmp_path):
    config = {"model":"unit-test","experts":32}
    writer = ParquetTraceWriter(tmp_path, config)
    frame = pd.DataFrame({"sequence_id":[0,0],"token_position":[0,1],"layer_id":[0,0]})
    writer.write_chunk(0, frame)
    resumed = ParquetTraceWriter(tmp_path, config)
    assert resumed.next_chunk_id == 1
    resumed.write_chunk(1, frame)
    resumed.finish()
    assert resumed.manifest["completed_chunks"] == 2
