"""Record supplied measured metrics in MLflow without invented constants."""
import argparse
import json
from pathlib import Path
import mlflow

def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('metrics', help='JSON object containing numeric measured metrics')
    parser.add_argument('--name', default='treponema_run')
    parser.add_argument('--uri', default='sqlite:///mlflow.db')
    parser.add_argument('--artifact', action='append', default=[])
    args = parser.parse_args()
    metrics = json.loads(Path(args.metrics).read_text())
    if not isinstance(metrics, dict) or not metrics:
        raise ValueError('A nonempty numeric JSON object is required')
    if any(type(v) not in (int, float) for v in metrics.values()):
        raise ValueError('Each metric must be a number')
    mlflow.set_tracking_uri(args.uri)
    mlflow.set_experiment('treponema_lab4')
    with mlflow.start_run(run_name=args.name) as run:
        mlflow.log_param('metrics_source', str(Path(args.metrics).resolve()))
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(args.metrics)
        for artifact in args.artifact:
            mlflow.log_artifact(artifact)
        print('Recorded run:', run.info.run_id)

if __name__ == '__main__':
    main()
