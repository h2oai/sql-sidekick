sample_values = """
Column Name: id
Column Type: uuid PRIMARY KEY

Column Name: ts
Column Type:  TIMESTAMP WITH TIME ZONE NOT NULL

Column Name: kind
Column Type: TEXT NOT NULL, -- or int?
Sample Values: ['EVENT']

Column Name: user_id
Column Type: TEXT

Column Name: user_name
Column Type: TEXT

Column Name: resource_type
Column Type: TEXT NOT NULL, -- or int?
Sample Values: ['FEATURE_STORE', 'PROJECT', 'MLOPS_EXPERIMENT', 'APP', 'APP_INSTANCE', 'MLOPS_DEPLOYMENT',
'MLOPS_DATASET', 'MLOPS_USER', 'RESOURCE_TYPE_UNSPECIFIED', 'SCORING', 'DAI_ENGINE', 'MLOPS_MODEL']

Column Name: resource_id
Column Type: TEXT

Column Name: stream
Column Type: TEXT NOT NULL
Sample Values: ['ai/h2o/cloud/mlops/deployment/created', 'ai/h2o/cloud/appstore/instance/gauge/running',
'ai/h2o/cloud/mlops/project/unshared',
'ai/h2o/cloud/mlops/gauge/project',
'ai/h2o/cloud/appstore/user/event/login',
'ai/h2o/cloud/mlops/gauge/registered-model-version',
'ai/h2o/cloud/appstore/instance/event/started',
'ai/h2o/cloud/mlops/deployment/deleted',
'ai/h2o/cloud/mlops/gauge/dataset',
'ai/h2o/cloud/fs/job/running',
'ai/h2o/engine/event/paused',
'ai/h2o/cloud/mlops/project/deleted',
'ai/h2o/engine/event/deleting',
'ai/h2o/engine/event/pausing',
'ai/h2o/cloud/mlops/gauge/deployment',
'ai/h2o/cloud/usage/global/gauge/resources',
'ai/h2o/cloud/mlops/gauge/registered-model',
'ai/h2o/cloud/appstore/instance/event/suspended',
'ai/h2o/cloud/usage/namespace/gauge/resources',
'ai/h2o/cloud/mlops/registered-model-version/created'],
'ai/h2o/cloud/mlops/project/created',
'ai/h2o/cloud/mlops/project/shared',
'ai/h2o/cloud/mlops/experiment/created',
'ai/h2o/cloud/mlops/dataset/created',
'ai/h2o/cloud/appstore/app/event/created',
'ai/h2o/cloud/appstore/instance/event/terminated',
'ai/h2o/cloud/mlops/gauge/user',
'ai/h2o/engine/event/starting',
'ai/h2o/cloud/mlops/event/scoring-result/created',
'ai/h2o/engine/event/running',
'ai/h2o/cloud/fs/job/submitted',
'ai/h2o/cloud/mlops/registered-model/created',
'ai/h2o/cloud/mlops/gauge/experiment',
'ai/h2o/document/ai/proxy',
'ai/h2o/cloud/mlops/experiment/unlinked',
'ai/h2o/cloud/fs/job/finished',
'ai/h2o/cloud/appstore/app/event/deleted',
'ai/h2o/cloud/appstore/instance/event/resumed']

Column Name: source
Column Type: TEXT NOT NULL

Column Name: payload
Column Type: jsonb NOT NULL
Sample Values:
[{'engineEvent': {'pausing': {'engine': {'cpu': '1', 'memory': '1', 'gpu': '0'}}}}]
]
"""

# For few shot prompting
samples_queries = [
    """
# query: Total number of CPUs used?
# answer:
SELECT sum((payload->'engineEvent'-> 'pausing' -> 'engine'->> 'cpu')::integer) AS total_cpus_used
FROM telemetry
WHERE payload->'engineEvent'-> 'pausing' -> 'engine'->> 'cpu' IS NOT NULL;
""",
    """
# query: Find the number of AI units for each user using stream for each resource type (overall)
# answer:
SELECT user_id, user_name, resource_type, date_trunc('day', ts) as start_day,
       sum(AI_units) as AI_units FROM (
      SELECT user_id, user_name, resource_type, ts,
             extract(epoch from ts - lag(ts) over (partition by user_id, resource_type order by ts)) / 3600 AS AI_units
      FROM telemetry
      WHERE stream = 'running'
     ) sub GROUP BY user_id, user_name, resource_type, start_day
ORDER BY start_day DESC NULLS LAST;
""",
    """
# query: Compute global usage over time
# answer:
SELECT
  ts AS time_interval,
  GREATEST((GREATEST((ram_gi / 64.0), (cpu / 8.0)) - gpu), 0) + (gpu * 4.0) as ai_units
FROM (
    SELECT
      -- This is a gauge stream, meaning multiple sources are exporting duplicate entries during the same hour interval
      ts,
      -- RAM usage in Gi
      COALESCE(((payload->'usageGauge'->'billingResources'->>'paddedMemoryReservationBytes')::bigint/1024.0/1024.0/1024.0), 0) AS ram_gi,
      -- CPU usage in vCPU
      COALESCE(((payload->'usageGauge'->'billingResources'->'paddedCpuReservationMillicpu')::int/1000.0), 0) AS cpu,
      -- GPU usage in number of GPUs
      COALESCE(((payload->'usageGauge'->'billingResources'->'gpuCount')::int), 0) AS gpu
    FROM telemetry
    WHERE stream = 'gauage_resources'
) AS internal
ORDER BY 1, 2 DESC;
""",
]
