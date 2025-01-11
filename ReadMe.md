## Customer Churn Prediction Model Pipeline

#### IF4054 - Pengoperasian Perangkat Lunak

### How to Run

1. `git clone https://github.com/nathaniacalista01/XOps.git`
2. Run `docker compose up -d` in `root` directory
3. Make sure all containers are running

### Steps for Airflow

1. Open `localhost:8080` to view airflow dashboard
2. Login using the credentials below
   ```bash
   username: admin
   ```
   for password, you can see it from `airflow/standalone_admin_password.txt` after building the container
3. Go to `admin` Menu bar and select `connections`
4. Edit the connection to:
   ```bash
   connection id: spark_default
   host: spark://spark-master
   port: 7077
   ```

### Ports List

1. Spark: `localhost:7077`
2. Model deploy: `localhost:5555`
3. MlFlow: `localhost:5000`
4. Minio: `localhost:9000`
5. Airflow: `localhost:8080`

### Team Member

13521100 - Alexander Jason
13521139 - Nathania Calista
13521170 - Haziq Abiyyu Mahdy
