CREATE TABLE IF NOT EXISTS bwg.agent_events (
  session_id STRING,
  agent_name STRING,
  latency_ms FLOAT64,
  tokens INT64,
  status STRING,
  brief STRING,
  model_id STRING,
  key_visual_uri STRING,
  nodes_executed ARRAY<STRING>,
  project_id STRING,
  dataset STRING,
  recorded_at TIMESTAMP
);
