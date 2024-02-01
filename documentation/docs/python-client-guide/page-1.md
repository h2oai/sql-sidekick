import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Page 1

<!-- Add code blocks in tabs: example -->

  ```mdx-code-block
  <Tabs>
  <TabItem value="Constants">
  ```

  ```python title="StandardFlowExample.py"
  ### Constants
  MLOPS_API_URL = "https://api.mlops.my.domain"
  TOKEN_ENDPOINT_URL="https://mlops.keycloak.domain/auth/realms/[fill-in-realm-name]"
  REFRESH_TOKEN="<your-refresh-token>"
  ```

  ```mdx-code-block
  </TabItem>
  <TabItem value="Response">
  ```

  ```python title="StandardFlowExample.py"
  Deployment drift metrics: {'count_feature_drift': 13, 'feature_frequency': 
  [{'categorical': {'description': '','name': '','point': [],
  ```

  ```mdx-code-block
  </TabItem>
  </Tabs>
  ```

