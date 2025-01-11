def branch_based_on_drift(**kwargs):
    # Extract the result of detect_drift
    ti = kwargs['ti']
    drift_detected = ti.xcom_pull(task_ids='detect_drift', key="drift")
    print("========DRIFT DETECTED===================")
    print(drift_detected)
    # Branch to the appropriate task
    if drift_detected:
        return 'ingest_clean_data'
    else:
        return 'skip_training'